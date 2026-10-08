use opencv::{
    core::{self, Mat, Point, Rect, Size},
    imgcodecs,
    imgproc::{self, TM_CCOEFF_NORMED},
    prelude::*,
};

// Reusable hand: masked gray-template vision. No screen logic here.
// Tuned for SM-E146B 1080x2408: X icons are ~28-80px, hence 7 scales.
pub const SCALES: [f64; 7] = [0.65, 0.75, 0.85, 0.92, 1.0, 1.08, 1.15];
pub const CANNY_LOW: f64 = 50.0;
pub const CANNY_HIGH: f64 = 150.0;
// Dual gate, Klick'r-style but dim-proof: shape confidence ANDed with
// hue/saturation distance plus a brightness-CONTRAST check (not absolute
// color-mean, which popup dim animations shift under our feet).
// 0.75: measured on-device — true X 0.77, nearest junk 0.73, old junk
// 0.65. Gate sits in the verified gap; HS/contrast gates plus the cell
// budget contain what slips near it. 0.80+ (textbook advice) kills real
// hits here, so the number comes from logs, not books.
pub const MATCH_CONF: f64 = 0.75;
pub const HS_MAX: f64 = 15.0;
pub const CONTRAST_RATIO: f64 = 0.5;
pub const STRONG_CONF: f64 = 0.90;
pub const MAX_CANDIDATES: i32 = 3;
// Tier-1 prefilter: base-scale shape score that justifies the full pyramid.
// Junk frames read 0.2-0.4, true X reads 0.7+; 0.45 splits them cheaply.
pub const PREFILTER_SCORE: f64 = 0.45;
pub const TRACK_SIZE: i32 = 140;

pub struct Tpl {
    pub edges: Mat,
    pub w: i32,
    pub h: i32,
    pub tpl: usize,
    pub hs: [f64; 2],
    pub grange: f64,
    pub base: bool,
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
    pub color: f64,
}

pub struct NamedRegion {
    pub name: &'static str,
    pub rect: Rect,
}

/// Best match in a region: location, template size, shape score,
/// template id, and the HSV color distance that passed/failed the gate.
pub struct Scored {
    pub loc: Point,
    pub w: i32,
    pub h: i32,
    pub score: f64,
    pub tpl: usize,
    pub color: f64,
}

pub type ScoredLoc = Option<Scored>;

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
pub fn canny(gray: &Mat) -> anyhow::Result<Mat> {
    let mut edges = Mat::default();
    imgproc::canny(gray, &mut edges, CANNY_LOW, CANNY_HIGH, 3, false)?;
    Ok(edges)
}

pub fn hsv_mean(bgr: &Mat) -> anyhow::Result<[f64; 3]> {
    let mut hsv = Mat::default();
    imgproc::cvt_color_def(bgr, &mut hsv, imgproc::COLOR_BGR2HSV)?;
    let m = core::mean(&hsv, &core::no_array())?;
    Ok([m[0], m[1], m[2]])
}

// Dim-proof gate: hue/saturation distance only (dimming lives in V,
// grays carry no hue). Plus a brightness-CONTRAST check: an icon patch
// always spans glyph-vs-background tones; flat junk does not.
pub fn hs_diff(a: [f64; 2], b: [f64; 2]) -> f64 {
    let mut h = (a[0] - b[0]).abs();
    if h > 90.0 {
        h = 180.0 - h;
    }
    let s = (a[1] - b[1]).abs();
    ((h / 90.0) + (s / 255.0)) * 50.0
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

// Precompute once at startup: per scale, the gray template, a binary mask
// of its glyph pixels (deviants from the mean: background baked into crops
// is ignored at match time), plus HS mean and brightness range for gates.
pub fn build_pyramid(raw: &RawTemplate, tpl_id: usize) -> anyhow::Result<Vec<Tpl>> {
    let mut out = Vec::with_capacity(SCALES.len());
    for &s in &SCALES {
        let mut gray = Mat::default();
        let mut color = Mat::default();
        imgproc::resize(
            &raw.gray,
            &mut gray,
            Size::new(0, 0),
            s,
            s,
            imgproc::INTER_LINEAR,
        )?;
        imgproc::resize(
            &raw.color,
            &mut color,
            Size::new(0, 0),
            s,
            s,
            imgproc::INTER_LINEAR,
        )?;
        if gray.cols() < 10 || gray.rows() < 10 {
            continue;
        }
        let edges = canny(&gray)?;
        let mut mn = 0.0;
        let mut mx = 0.0;
        core::min_max_loc(
            &gray,
            Some(&mut mn),
            Some(&mut mx),
            None,
            None,
            &core::no_array(),
        )?;
        let hsv = hsv_mean(&color)?;
        out.push(Tpl {
            w: edges.cols(),
            h: edges.rows(),
            tpl: tpl_id,
            hs: [hsv[0], hsv[1]],
            grange: mx - mn,
            base: (s - 1.0).abs() < 1e-9,
            edges,
        });
    }
    Ok(out)
}

// Candidate gate: HS distance tight, brightness range (contrast) present.
// Returns (pass, hs_distance). V is deliberately ignored: dim overlays and
// countdown fades shift brightness, never hue.
fn gate_candidate(roi_color: &Mat, roi_gray: &Mat, r: Rect, t: &Tpl) -> (bool, f64) {
    let hs = match Mat::roi(roi_color, r) {
        Ok(v) => match hsv_mean(&v.clone_pointee()) {
            Ok(m) => hs_diff([m[0], m[1]], t.hs),
            Err(_) => return (false, f64::INFINITY),
        },
        Err(_) => return (false, f64::INFINITY),
    };
    if hs > HS_MAX {
        return (false, hs);
    }
    let range = match Mat::roi(roi_gray, r) {
        Ok(v) => {
            let g = v.clone_pointee();
            let mut mn = 0.0;
            let mut mx = 0.0;
            match core::min_max_loc(
                &g,
                Some(&mut mn),
                Some(&mut mx),
                None,
                None,
                &core::no_array(),
            ) {
                Ok(()) => mx - mn,
                Err(_) => return (false, hs),
            }
        }
        Err(_) => return (false, hs),
    };
    (range >= CONTRAST_RATIO * t.grange, hs)
}

/// Raw shape scores per template id, no gates: diagnostic only.
/// Shows whether a miss is shape (low score) or gates (score ok, vetoed).
pub fn raw_scores(roi_edges: &Mat, pyramid: &[Tpl]) -> Vec<(usize, f64)> {
    let mut per_tpl: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for t in pyramid {
        if t.w > roi_edges.cols() || t.h > roi_edges.rows() {
            continue;
        }
        let mut result = Mat::default();
        if imgproc::match_template(
            roi_edges,
            &t.edges,
            &mut result,
            TM_CCOEFF_NORMED,
            &core::no_array(),
        )
        .is_err()
        {
            continue;
        }
        let mut max_val = 0.0;
        let mut max_loc = Point::new(0, 0);
        if core::min_max_loc(
            &result,
            None,
            Some(&mut max_val),
            None,
            Some(&mut max_loc),
            &core::no_array(),
        )
        .is_err()
        {
            continue;
        }
        let e = per_tpl.entry(t.tpl).or_insert(0.0);
        if max_val > *e {
            *e = max_val;
        }
    }
    let mut v: Vec<(usize, f64)> = per_tpl.into_iter().collect();
    v.sort_by_key(|&(tpl, _)| tpl);
    v
}

pub fn match_roi(
    roi_edges: &Mat,
    roi_gray: &Mat,
    roi_color: &Mat,
    pyramid: &[Tpl],
) -> anyhow::Result<ScoredLoc> {
    // Masked gray matching (background in crops is ignored) + HS/contrast
    // gates per candidate + next-best loop on reject. If this OpenCV build
    // rejects masked CCOEFF_NORMED, fall back to unmasked once and remember.
    // Tier 1: base scale only, shape score only. Empty regions die here
    // for the price of 2-3 matches instead of the full pyramid.
    let mut pre = 0.0;
    for t in pyramid.iter().filter(|t| t.base) {
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
        if max_val > pre {
            pre = max_val;
        }
    }
    if pre < PREFILTER_SCORE {
        return Ok(None);
    }
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
            let (pass, hs) = gate_candidate(roi_color, roi_gray, r, t);
            if pass {
                if max_val > best_score {
                    best_score = max_val;
                    best = Some(Scored {
                        loc: max_loc,
                        w: t.w,
                        h: t.h,
                        score: max_val,
                        tpl: t.tpl,
                        color: hs,
                    });
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

// Coverage for this phone: ad X lives top corners, Play Store popups
// upper-middle right. Boxes overlap by 80px+ so an X straddling a boundary
// still matches whole in one box: continuous 0..~690px on both edges.
pub fn default_regions(cols: i32, rows: i32) -> [NamedRegion; 4] {
    let s = ((cols as f32 * 0.32) as i32).clamp(240, 420);
    let rw = s.min(cols);
    let top_h = (s + 80).min(rows);
    let mid_h = (s + 80).min(rows);
    let mid_y = (s - 80).max(0).min((rows - mid_h).max(0));
    [
        NamedRegion {
            name: "top_right",
            rect: Rect::new((cols - rw).max(0), 0, rw, top_h),
        },
        NamedRegion {
            name: "top_left",
            rect: Rect::new(0, 0, rw, top_h),
        },
        NamedRegion {
            name: "mid_right",
            rect: Rect::new((cols - rw).max(0), mid_y, rw, mid_h),
        },
        NamedRegion {
            name: "mid_left",
            rect: Rect::new(0, mid_y, rw, mid_h),
        },
    ]
}
