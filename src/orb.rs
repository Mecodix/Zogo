use opencv::{
    core::{self, Mat, Point, Vector, NORM_HAMMING},
    features2d::{BFMatcher, DMatch, DescriptorMatcherTraitConst, Feature2DTrait, KeyPoint, ORB},
    prelude::*,
};

use crate::detect;

// ORB hand: feature matching for the X target (no template matching).
// Template descriptors are extracted once at startup. Each frame, every ROI
// gets ORB keypoints, KNN k=2 against the template, Lowe ratio test 0.75,
// and the tap point is the centroid of the good screen-side keypoints.
//
// Why ORB and not SIFT: binary X icons have few keypoints either way, ORB's
// Hamming + brute-force matcher is ~10x cheaper on a phone CPU, and ORB is
// patent-free. Flann is skipped on purpose: brute force over a few hundred
// 32-byte descriptors is microseconds; Flann-LSH tuning buys nothing here.
pub const LOWE_RATIO: f32 = 0.75;
pub const MIN_GOOD: usize = 6;
pub const STRONG_GOOD: usize = 12;

pub struct OrbMatcher {
    orb: ORB,
    matcher: BFMatcher,
    tpl_desc: Mat,
}

impl OrbMatcher {
    pub fn new() -> anyhow::Result<Self> {
        // Raw gray on purpose: ORB needs intensity texture (FAST corners +
        // BRIEF). Edge maps would destroy exactly what it measures.
        let base = detect::load_template_gray("x_template.png")?;
        let mut orb = ORB::new(1000, 1.2, 8, 31, 0, 2, 0, 31, 20)?;
        let mut kps = Vector::<KeyPoint>::new();
        let mut desc = Mat::default();
        orb.detect_and_compute(&base, &core::no_array(), &mut kps, &mut desc, false)?;
        // A bare synthetic X yields few keypoints; a real cropped screenshot
        // (rounded rect, shadow, bg) yields far more. Prefer the real one.
        if desc.empty() || kps.len() < MIN_GOOD {
            anyhow::bail!("template too featureless ({} kps, need {})", kps.len(), MIN_GOOD);
        }
        eprintln!("orb template: {} kps", kps.len());
        let matcher = BFMatcher::new(NORM_HAMMING, false)?;
        Ok(Self {
            orb,
            matcher,
            tpl_desc: desc,
        })
    }

    /// Gray ROI in, tap point in ROI-local coords + good-match count out.
    pub fn match_roi(&mut self, roi_gray: &Mat) -> anyhow::Result<Option<(Point, usize)>> {
        let mut kps = Vector::<KeyPoint>::new();
        let mut desc = Mat::default();
        self.orb
            .detect_and_compute(roi_gray, &core::no_array(), &mut kps, &mut desc, false)?;
        if desc.empty() || kps.len() == 0 {
            return Ok(None);
        }
        let mut knn = Vector::<Vector<DMatch>>::new();
        self.matcher
            .knn_train_match_def(&self.tpl_desc, &desc, &mut knn, 2)?;
        let mut sx = 0f64;
        let mut sy = 0f64;
        let mut n = 0usize;
        for i in 0..knn.len() {
            let pair = knn.get(i)?;
            if pair.len() < 2 {
                continue;
            }
            let m = pair.get(0)?;
            let q = pair.get(1)?;
            if m.distance < LOWE_RATIO * q.distance {
                let kp = kps.get(m.train_idx as usize)?;
                sx += kp.pt.x as f64;
                sy += kp.pt.y as f64;
                n += 1;
            }
        }
        if n >= MIN_GOOD {
            Ok(Some((
                Point::new((sx / n as f64) as i32, (sy / n as f64) as i32),
                n,
            )))
        } else {
            Ok(None)
        }
    }
}
