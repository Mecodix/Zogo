use std::process::Command;

use opencv::{
    core::{self, Mat, Vector},
    imgcodecs, imgproc,
    prelude::*,
};

// Reusable hand: screen capture. Zero disk I/O, RAM only.
// Primary PNG (318ms on SM-E146B), fallback RAW (638ms). Color kept so the
// detector can run Klick'r's second gate: HSV mean check on candidates.
pub fn capture_color() -> anyhow::Result<Mat> {
    match capture_png_color() {
        Ok(m) => Ok(m),
        Err(png_err) => capture_raw_color()
            .map_err(|raw_err| anyhow::anyhow!("png: {:#} | raw: {:#}", png_err, raw_err)),
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

// Android 15 = 16-byte header (w,h,fmt,colorspace), older = 12.
fn capture_raw_color() -> anyhow::Result<Mat> {
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
    let mut color = Mat::default();
    imgproc::cvt_color_def(&rgba, &mut color, imgproc::COLOR_RGBA2BGR)?;
    if color.empty() {
        anyhow::bail!("cvt empty");
    }
    Ok(color)
}
