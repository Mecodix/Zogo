use opencv::{
    core::{self, Mat, Point, Rect, Size},
    imgcodecs,
    imgproc::{self, TM_CCOEFF_NORMED},
    prelude::*,
};

// Reusable hand: edge-template vision. No screen logic here.
// Tuned for SM-E146B 1080x2408: X icons are ~48-80px, hence 64px base.
pub const SCALES: [f64; 5] = [0.85, 0.92, 1.0, 1.08, 1.15];
pub const CANNY_LOW: f64 = 50.0;
pub const CANNY_HIGH: f64 = 150.0;
pub const MATCH_THRESH: f64 = 0.55;
pub const STRONG_HIT: f64 = 0.78;
pub const TRACK_SIZE: i32 = 140;

pub struct Tpl {
    pub edges: Mat,
    pub w: i32,
    pub h: i32,
}

pub struct Hit {
    pub x: i32,
    pub y: i32,
    pub score: f64,
    pub region: &'static str,
}

pub struct NamedRegion {
    pub name: &'static str,
    pub rect: Rect,
}

pub fn load_template_gray(path: &str) -> anyhow::Result<Mat> {
    if let Ok(m) = imgcodecs::imread(path, imgcodecs::IMREAD_GRAYSCALE) {
        if !m.empty() {
            return Ok(m);
        }
    }
    // Synthetic X fallback so binary runs with zero assets.
    let mut m = Mat::zeros(64, 64, core::CV_8UC1)?.to_mat()?;
    let white = core::Scalar::all(255.0);
    imgproc::line(
        &mut m,
        Point::new(10, 10),
        Point::new(54, 54),
        white,
        8,
        imgproc::LINE_8,
        0,
    )?;
    imgproc::line(
        &mut m,
        Point::new(54, 10),
        Point::new(10, 54),
        white,
        8,
        imgproc::LINE_8,
        0,
    )?;
    Ok(m)
}

pub fn canny(gray: &Mat) -> anyhow::Result<Mat> {
    let mut edges = Mat::default();
    imgproc::canny(gray, &mut edges, CANNY_LOW, CANNY_HIGH, 3, false)?;
    Ok(edges)
}

// Precompute once at startup: 5x less work per frame.
pub fn build_pyramid(base: &Mat) -> anyhow::Result<Vec<Tpl>> {
    let mut out = Vec::with_capacity(SCALES.len());
    for &s in &SCALES {
        let mut resized = Mat::default();
        imgproc::resize(
            base,
            &mut resized,
            Size::new(0, 0),
            s,
            s,
            imgproc::INTER_LINEAR,
        )?;
        if resized.cols() < 10 || resized.rows() < 10 {
            continue;
        }
        let edges = canny(&resized)?;
        out.push(Tpl {
            w: edges.cols(),
            h: edges.rows(),
            edges,
        });
    }
    Ok(out)
}

pub fn match_roi(
    roi_edges: &Mat,
    pyramid: &[Tpl],
) -> anyhow::Result<Option<(Point, i32, i32, f64)>> {
    let mut best_score = 0.0;
    let mut best_loc = Point::new(0, 0);
    let mut best_w = 0;
    let mut best_h = 0;
    for t in pyramid {
        if t.w > roi_edges.cols() || t.h > roi_edges.rows() {
            continue;
        }
        let mut result = Mat::default();
        imgproc::match_template(
            roi_edges,
            &t.edges,
            &mut result,
            TM_CCOEFF_NORMED,
            &core::no_array(),
        )?;
        let mut max_val = 0.0;
        let mut max_loc = Point::new(0, 0);
        core::min_max_loc(
            &result,
            None,
            Some(&mut max_val),
            None,
            Some(&mut max_loc),
            &core::no_array(),
        )?;
        if max_val > best_score {
            best_score = max_val;
            best_loc = max_loc;
            best_w = t.w;
            best_h = t.h;
            if best_score >= STRONG_HIT {
                break;
            }
        }
    }
    if best_score >= MATCH_THRESH {
        Ok(Some((best_loc, best_w, best_h, best_score)))
    } else {
        Ok(None)
    }
}

pub fn clamp_rect(x: i32, y: i32, w: i32, h: i32, cols: i32, rows: i32) -> Option<Rect> {
    if cols < 10 || rows < 10 {
        return None;
    }
    let x = x.clamp(0, cols - 1);
    let y = y.clamp(0, rows - 1);
    let w = w.min(cols - x).max(10);
    let h = h.min(rows - y).max(10);
    if x + w > cols || y + h > rows {
        return None;
    }
    Some(Rect::new(x, y, w, h))
}

// Coverage for this phone: ad X lives top corners, Play Store popups mid-right.
// Order = likelihood. No bottom: never seen there. Fixed array, zero alloc.
pub fn default_regions(cols: i32, rows: i32) -> [NamedRegion; 4] {
    let s = ((cols as f32 * 0.32) as i32).clamp(240, 420);
    let rw = s.min(cols);
    let rh = s.min(rows);
    let mid_y = ((rows as f32 * 0.35) as i32).clamp(0, (rows - rh).max(0));
    [
        NamedRegion {
            name: "top_right",
            rect: Rect::new((cols - rw).max(0), 0, rw, rh),
        },
        NamedRegion {
            name: "top_left",
            rect: Rect::new(0, 0, rw, rh),
        },
        NamedRegion {
            name: "mid_right",
            rect: Rect::new((cols - rw).max(0), mid_y, rw, rh),
        },
        NamedRegion {
            name: "mid_left",
            rect: Rect::new(0, mid_y, rw, rh),
        },
    ]
}
