use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use opencv::{
    core::{self, Mat, Vector},
    imgcodecs, imgproc,
    prelude::*,
};

// Reusable hand: screen capture. Zero screenshot files left behind.
// Strategy, measured on SM-E146B (not guessed):
//   screencap -p stdout .. 318ms (PNG encode dominates)
//   screencap raw stdout . 638ms (10MB pipe copy dominates)
//   screencap raw file .... 186ms flash / 163ms tmpfs (capture itself!)
//   2x raw-file parallel  188ms total = ~94ms/frame (composer parallelizes)
// So: a ring of in-flight `screencap <tmpfsfile>` processes, round-robin
// wait. Steady state yields a fresh frame every ~100ms. Falls back to the
// PNG-stdout path where tmpfs is not writable.
const SLOTS: usize = 2;
const TMPDIR: &str = "/dev";
const WAIT_TIMEOUT: Duration = Duration::from_millis(3000);

struct Slot {
    path: String,
    child: Option<Child>,
}

fn spawn_screencap(path: &str) -> anyhow::Result<Child> {
    let _ = std::fs::remove_file(path);
    Command::new("screencap")
        .arg(path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| anyhow::anyhow!("spawn screencap: {}", e))
}

fn wait_child(child: &mut Child) -> anyhow::Result<()> {
    let t0 = Instant::now();
    loop {
        match child.try_wait()? {
            Some(status) => {
                if status.success() {
                    return Ok(());
                }
                anyhow::bail!("screencap exit {}", status);
            }
            None => {
                if t0.elapsed() > WAIT_TIMEOUT {
                    let _ = child.kill();
                    anyhow::bail!("screencap hung");
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
}

/// Raw RGBA bytes (12- or 16-byte header, Android 15 has 16) -> BGR Mat.
fn raw_to_bgr(b: &[u8]) -> anyhow::Result<Mat> {
    if b.len() < 16 {
        anyhow::bail!("raw too short ({} bytes)", b.len());
    }
    let w = u32::from_le_bytes(b[0..4].try_into().unwrap()) as i32;
    let h = u32::from_le_bytes(b[4..8].try_into().unwrap()) as i32;
    let fmt = u32::from_le_bytes(b[8..12].try_into().unwrap());
    if w < 200 || h < 200 || w > 5000 || h > 5000 {
        anyhow::bail!("bad dims {}x{}", w, h);
    }
    if fmt != 1 {
        anyhow::bail!("not RGBA_8888 fmt={}", fmt);
    }
    let stride = (w as usize) * (h as usize) * 4usize;
    let off = if b.len() >= 16 + stride && b.len() < 16 + stride + 16 {
        16
    } else if b.len() >= 12 + stride {
        12
    } else {
        anyhow::bail!("truncated raw ({} bytes, need {})", b.len(), 16 + stride);
    };
    let pixels = &b[off..off + stride];
    let mut rgba = Mat::zeros(h, w, core::CV_8UC4)?.to_mat()?;
    {
        let dst = rgba.data_bytes_mut()?;
        if dst.len() != pixels.len() {
            anyhow::bail!("stride mismatch");
        }
        dst.copy_from_slice(pixels);
    }
    let mut color = Mat::default();
    imgproc::cvt_color_def(&rgba, &mut color, imgproc::COLOR_RGBA2BGR)?;
    if color.empty() {
        anyhow::bail!("cvt empty");
    }
    Ok(color)
}

pub struct ScreenPump {
    slots: Vec<Slot>,
    cursor: usize,
    legacy: bool,
}

impl ScreenPump {
    pub fn new() -> Self {
        // Probe: can we round-trip 1 byte through tmpfs?
        let probe = format!("{}/sniper_probe", TMPDIR);
        let legacy = std::fs::write(&probe, [1u8])
            .and_then(|_| std::fs::read(&probe))
            .map(|v| v != vec![1u8])
            .unwrap_or(true);
        let _ = std::fs::remove_file(&probe);
        if legacy {
            eprintln!("capture: tmpfs unavailable, PNG-stdout fallback");
            return Self {
                slots: Vec::new(),
                cursor: 0,
                legacy: true,
            };
        }
        let mut slots = Vec::with_capacity(SLOTS);
        for i in 0..SLOTS {
            let path = format!("{}/sniper_fb{}.raw", TMPDIR, i);
            let child = spawn_screencap(&path).ok();
            slots.push(Slot { path, child });
        }
        eprintln!("capture: {}x raw-tmpfs ring", SLOTS);
        Self {
            slots,
            cursor: 0,
            legacy: false,
        }
    }

    /// Next color frame: wait the oldest in-flight capture, respawn it.
    pub fn next_color(&mut self) -> anyhow::Result<Mat> {
        if self.legacy {
            return capture_png_color();
        }
        let idx = self.cursor % self.slots.len();
        self.cursor += 1;
        let slot = &mut self.slots[idx];
        let mut child = slot
            .child
            .take()
            .ok_or_else(|| anyhow::anyhow!("no capture process"))?;
        let r = (|| -> anyhow::Result<Mat> {
            wait_child(&mut child)?;
            let bytes = std::fs::read(&slot.path)?;
            let _ = std::fs::remove_file(&slot.path);
            raw_to_bgr(&bytes)
        })();
        // Always keep the ring full, even on bad frames.
        slot.child = spawn_screencap(&slot.path).ok();
        r
    }
}

pub fn to_gray(color: &Mat) -> anyhow::Result<Mat> {
    let mut gray = Mat::default();
    imgproc::cvt_color_def(color, &mut gray, imgproc::COLOR_BGR2GRAY)?;
    if gray.empty() {
        anyhow::bail!("gray empty");
    }
    Ok(gray)
}

fn capture_png_color() -> anyhow::Result<Mat> {
    let out = Command::new("screencap")
        .arg("-p")
        .output()
        .map_err(|e| anyhow::anyhow!("spawn screencap -p: {}", e))?;
    if out.stdout.is_empty() {
        anyhow::bail!("empty png stdout");
    }
    let buf = Vector::<u8>::from_slice(&out.stdout);
    let m = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR)?;
    if m.empty() {
        anyhow::bail!("imdecode empty ({} bytes in)", out.stdout.len());
    }
    Ok(m)
}
