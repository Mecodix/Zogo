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
// Dual gate ported from Klick'r TemplateMatcher: shape confidence ANDed
// with HSV color distance. Their int threshold T means conf > (100-T)/100
// and color <= T; T=25 here -> conf > 0.75, color <= 25.
pub const MATCH_CONF: f64 = 0.75;
pub const COLOR_MAX: f64 = 25.0;
pub const STRONG_CONF: f64 = 0.88;
pub const MAX_CANDIDATES: i32 = 3;
pub const TRACK_SIZE: i32 = 140;

pub struct Tpl {
    pub edges: Mat,
    pub w: i32,
    pub h: i32,
    pub tpl: usize,
    pub hsv: [f64; 3],
}

pub struct RawTemplate {
    pub gray: Mat,
    pub color: Mat,
}

pub struct Hit {
    pub x: i32,
    pub y: i32,
    pub score: f64,
    pub region: &'static str,
    pub tpl: usize,
}

pub struct NamedRegion {
    pub name: &'static str,
    pub rect: Rect,
}

/// (top-left of best match, template w/h, normalized score, template id)
pub type ScoredLoc = Option<(Point, i32, i32, f64, usize)>;

// All x_template*.png crops are loaded (x_template.png, x_template1..3).
// Synthetic X fallback only if none exist.
pub fn load_templates() -> anyhow::Result<Vec<RawTemplate>> {
    let mut v = Vec::new();
    for name in [
        "x_template.png",
        "x_template1.png",
        "x_template2.png",
        "x_template3.png",
    ] {
        // exists() first: imread_ logs a scary WARN for missing files.
        if !std::path::Path::new(name).exists() {
            continue;
        }
        let gray = imgcodecs::imread(name, imgcodecs::IMREAD_GRAYSCALE)?;
        let color = imgcodecs::imread(name, imgcodecs::IMREAD_COLOR)?;
        if !gray.empty() && !color.empty() {
            eprintln!("template {}: {}x{}", name, gray.cols(), gray.rows());
            v.push(RawTemplate { gray, color });
        }
    }
    if v.is_empty() {
        eprintln!("no template files, synthetic fallback");
        let gray = load_template_gray("x_template.png")?;
        let mut color = Mat::zeros(64, 64, core::CV_8UC3)?.to_mat()?;
        let white = core::Scalar::new(255.0, 255.0, 255.0, 0.0);
        imgproc::line(
            &mut color,
            Point::new(10, 10),
            Point::new(54, 54),
            white,
            8,
            imgproc::LINE_8,
            0,
        )?;
        imgproc::line(
            &mut color,
            Point::new(54, 10),
            Point::new(10, 54),
            white,
            8,
            imgproc::LINE_8,
            0,
        )?;
        v.push(RawTemplate { gray, color });
    }
    Ok(v)
}

// Mean HSV of a BGR patch, channels as [H 0..180, S, V].
pub fn hsv_mean(bgr: &Mat) -> anyhow::Result<[f64; 3]> {
    let mut hsv = Mat::default();
    imgproc::cvt_color_def(bgr, &mut hsv, imgproc::COLOR_BGR2HSV)?;
    let m = core::mean(&hsv, &core::no_array())?;
    Ok([m[0], m[1], m[2]])
}

// Klick'r getColorDiff: circular H distance, linear S/V, mean on 0..100.
pub fn color_diff(a: [f64; 3], b: [f64; 3]) -> f64 {
    let mut h = (a[0] - b[0]).abs();
    if h > 90.0 {
        h = 180.0 - h;
    }
    let s = (a[1] - b[1]).abs();
    let v = (a[2] - b[2]).abs();
    ((h / 90.0) + (s / 255.0) + (v / 255.0)) * (100.0 / 3.0)
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

// Precompute once at startup: 5 scales per template, best wins per frame.
pub fn build_pyramid(raw: &RawTemplate, tpl_id: usize) -> anyhow::Result<Vec<Tpl>> {
    let mut out = Vec::with_capacity(SCALES.len());
    for &s in &SCALES {
        let mut resized_gray = Mat::default();
        let mut resized_color = Mat::default();
        imgproc::resize(
            &raw.gray,
            &mut resized_gray,
            Size::new(0, 0),
            s,
            s,
            imgproc::INTER_LINEAR,
        )?;
        imgproc::resize(
            &raw.color,
            &mut resized_color,
            Size::new(0, 0),
            s,
            s,
            imgproc::INTER_LINEAR,
        )?;
        if resized_gray.cols() < 10 || resized_gray.rows() < 10 {
            continue;
        }
        let edges = canny(&resized_gray)?;
        let hsv = hsv_mean(&resized_color)?;
        out.push(Tpl {
            w: edges.cols(),
            h: edges.rows(),
            tpl: tpl_id,
            hsv,
            edges,
        });
    }
    Ok(out)
}

pub fn match_roi(roi_edges: &Mat, roi_color: &Mat, pyramid: &[Tpl]) -> anyhow::Result<ScoredLoc> {
    // Klick'r parseMatchingResult port: per scale, take the best peak; if the
    // HSV color gate rejects it, suppress that area and try the NEXT peak
    // (up to MAX_CANDIDATES) instead of trusting the global max blindly.
    let mut best: ScoredLoc = None;
    let mut best_score = 0.0;
    'outer: for t in pyramid {
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
        for _ in 0..MAX_CANDIDATES {
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
            if max_val < MATCH_CONF {
                break;
            }
            let r = Rect::new(max_loc.x, max_loc.y, t.w, t.h);
            let pass = match Mat::roi(roi_color, r) {
                Ok(v) => match hsv_mean(&v.clone_pointee()) {
                    Ok(m) => color_diff(m, t.hsv) <= COLOR_MAX,
                    Err(_) => false,
                },
                Err(_) => false,
            };
            if pass {
                if max_val > best_score {
                    best_score = max_val;
                    best = Some((max_loc, t.w, t.h, max_val, t.tpl));
                }
                if max_val >= STRONG_CONF {
                    break 'outer;
                }
                break;
            }
            // Reject: paint peak away, look at the next one in this scale.
            imgproc::rectangle(
                &mut result,
                r,
                core::Scalar::all(-1.0),
                -1,
                imgproc::LINE_8,
                0,
            )?;
        }
    }
    Ok(best)
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
    // Upper-middle: Play Store popup X sits above screen middle on the right.
    // 0.22 keeps continuity with the top boxes (which end ~345px).
    let mid_y = ((rows as f32 * 0.22) as i32).clamp(0, (rows - rh).max(0));
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
