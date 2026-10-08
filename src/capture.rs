use std::process::Command;

use opencv::{
    core::{self, Mat, Vector},
    imgcodecs, imgproc,
    prelude::*,
};

// Reusable hand: screen capture. Zero disk I/O, RAM only.
// Primary PNG (318ms on SM-E146B), fallback RAW (638ms).
pub fn capture_gray() -> anyhow::Result<Mat> {
    match capture_png_gray() {
        Ok(m) => Ok(m),
        Err(png_err) => capture_raw_gray()
            .map_err(|raw_err| anyhow::anyhow!("png: {:#} | raw: {:#}", png_err, raw_err)),
    }
}

fn capture_png_gray() -> anyhow::Result<Mat> {
    let out = Command::new("screencap")
        .arg("-p")
        .output()
        .map_err(|e| anyhow::anyhow!("spawn screencap -p: {}", e))?;
    if out.stdout.is_empty() {
        anyhow::bail!("empty png stdout");
    }
    let buf = Vector::<u8>::from_slice(&out.stdout);
    let m = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_GRAYSCALE)?;
    if m.empty() {
        anyhow::bail!("imdecode empty ({} bytes in)", out.stdout.len());
    }
    Ok(m)
}

// Android 15 = 16-byte header (w,h,fmt,colorspace), older = 12.
fn capture_raw_gray() -> anyhow::Result<Mat> {
    let out = Command::new("screencap")
        .output()
        .map_err(|e| anyhow::anyhow!("spawn screencap: {}", e))?;
    let b = &out.stdout;
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
    let mut gray = Mat::default();
    // _def form: portable across OpenCV 4.5 (4 args) and 4.11+ (5th AlgorithmHint).
    imgproc::cvt_color_def(&rgba, &mut gray, imgproc::COLOR_RGBA2GRAY)?;
    if gray.empty() {
        anyhow::bail!("cvt empty");
    }
    Ok(gray)
}
