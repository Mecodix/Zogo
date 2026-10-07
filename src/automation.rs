use opencv::{core::Mat, core::Point, prelude::*};

use crate::{
    detect::{self, Hit, Tpl},
    orb::{self, OrbMatcher},
};

// One trait per screen job. Hands (capture/detect/tap/orb) are shared,
// each automation holds only its own template + logic.
pub trait ScreenAutomation {
    fn name(&self) -> &str;
    fn step(&mut self, gray: &Mat) -> anyhow::Result<Option<Hit>>;
    fn cooldown_ms(&self) -> u64;
}

// Parked alternative: ORB features (KNN k=2, Lowe 0.75, centroid tap).
// Kept for A/B: benchmarks say edge-template wins on flat X icons.
#[allow(dead_code)]
pub struct OrbXCloser {
    orb: OrbMatcher,
    last_hit: Option<Point>,
    misses: u32,
}

#[allow(dead_code)]
impl OrbXCloser {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            orb: OrbMatcher::new()?,
            last_hit: None,
            misses: 0,
        })
    }
}

#[allow(dead_code)]
impl ScreenAutomation for OrbXCloser {
    fn name(&self) -> &str {
        "orb_x"
    }

    fn cooldown_ms(&self) -> u64 {
        1200
    }

    fn step(&mut self, gray: &Mat) -> anyhow::Result<Option<Hit>> {
        let cols = gray.cols();
        let rows = gray.rows();

        // Fast path: box around last hit, raw gray (ORB needs texture).
        if let Some(p) = self.last_hit {
            if let Some(r) = detect::clamp_rect(
                p.x - detect::TRACK_SIZE / 2,
                p.y - detect::TRACK_SIZE / 2,
                detect::TRACK_SIZE,
                detect::TRACK_SIZE,
                cols,
                rows,
            ) {
                if let Ok(roi_view) = Mat::roi(gray, r) {
                    let roi: Mat = roi_view.clone_pointee();
                    if let Ok(Some((pt, n))) = self.orb.match_roi(&roi) {
                        let hit = Hit {
                            x: r.x + pt.x,
                            y: r.y + pt.y,
                            score: n as f64,
                            region: "track",
                        };
                        self.last_hit = Some(Point::new(hit.x, hit.y));
                        self.misses = 0;
                        return Ok(Some(hit));
                    }
                }
            }
        }

        let mut best: Option<Hit> = None;
        for region in detect::default_regions(cols, rows) {
            let roi: Mat = match Mat::roi(gray, region.rect) {
                Ok(v) => v.clone_pointee(),
                Err(_) => continue,
            };
            match self.orb.match_roi(&roi) {
                Ok(Some((pt, n))) => {
                    let cand = Hit {
                        x: region.rect.x + pt.x,
                        y: region.rect.y + pt.y,
                        score: n as f64,
                        region: region.name,
                    };
                    let better = best
                        .as_ref()
                        .map(|b: &Hit| cand.score > b.score)
                        .unwrap_or(true);
                    if better {
                        let strong = n >= orb::STRONG_GOOD;
                        best = Some(cand);
                        if strong {
                            break;
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => eprintln!("orb err {}: {:#}", region.name, e),
            }
        }

        match &best {
            Some(h) => {
                self.last_hit = Some(Point::new(h.x, h.y));
                self.misses = 0;
            }
            None => {
                // Miss-based decay: ~5 misses at 3fps = track dies in ~1.5s.
                self.misses += 1;
                if self.misses >= 5 {
                    self.last_hit = None;
                    self.misses = 0;
                }
            }
        }
        Ok(best)
    }
}

// Active accuracy pick: Canny multi-scale template matcher. Benchmarks and
// UI-icon literature agree edge-shape beats ORB on tiny low-texture X icons.
pub struct AdCloser {
    pyramid: Vec<Tpl>,
    last_hit: Option<Point>,
    misses: u32,
}

impl AdCloser {
    pub fn new() -> anyhow::Result<Self> {
        let base = detect::load_template_gray("x_template.png")?;
        let pyramid = detect::build_pyramid(&base)?;
        if pyramid.is_empty() {
            anyhow::bail!("empty pyramid");
        }
        Ok(Self {
            pyramid,
            last_hit: None,
            misses: 0,
        })
    }
}

impl ScreenAutomation for AdCloser {
    fn name(&self) -> &str {
        "ad_closer"
    }

    fn cooldown_ms(&self) -> u64 {
        1200
    }

    fn step(&mut self, gray: &Mat) -> anyhow::Result<Option<Hit>> {
        let cols = gray.cols();
        let rows = gray.rows();

        if let Some(p) = self.last_hit {
            if let Some(r) = detect::clamp_rect(
                p.x - detect::TRACK_SIZE / 2,
                p.y - detect::TRACK_SIZE / 2,
                detect::TRACK_SIZE,
                detect::TRACK_SIZE,
                cols,
                rows,
            ) {
                if let Ok(roi_view) = Mat::roi(gray, r) {
                    let roi: Mat = roi_view.clone_pointee();
                    if let Ok(re) = detect::canny(&roi) {
                        if let Ok(Some((loc, w, h, s))) = detect::match_roi(&re, &self.pyramid) {
                            let hit = Hit {
                                x: r.x + loc.x + w / 2,
                                y: r.y + loc.y + h / 2,
                                score: s,
                                region: "track",
                            };
                            self.last_hit = Some(Point::new(hit.x, hit.y));
                            self.misses = 0;
                            return Ok(Some(hit));
                        }
                    }
                }
            }
        }

        let mut best: Option<Hit> = None;
        for region in detect::default_regions(cols, rows) {
            let roi: Mat = match Mat::roi(gray, region.rect) {
                Ok(v) => v.clone_pointee(),
                Err(_) => continue,
            };
            let re = match detect::canny(&roi) {
                Ok(m) => m,
                Err(_) => continue,
            };
            match detect::match_roi(&re, &self.pyramid) {
                Ok(Some((loc, w, h, s))) => {
                    let cand = Hit {
                        x: region.rect.x + loc.x + w / 2,
                        y: region.rect.y + loc.y + h / 2,
                        score: s,
                        region: region.name,
                    };
                    let better = best
                        .as_ref()
                        .map(|b: &Hit| cand.score > b.score)
                        .unwrap_or(true);
                    if better {
                        let strong = cand.score >= detect::STRONG_HIT;
                        best = Some(cand);
                        if strong {
                            break;
                        }
                    }
                }
                Ok(None) => {}
                Err(e) => eprintln!("match err {}: {:#}", region.name, e),
            }
        }

        match &best {
            Some(h) => {
                self.last_hit = Some(Point::new(h.x, h.y));
                self.misses = 0;
            }
            None => {
                self.misses += 1;
                if self.misses >= 5 {
                    self.last_hit = None;
                    self.misses = 0;
                }
            }
        }
        Ok(best)
    }
}

// Add job #2 here later, e.g. PlayStorePopupCloser / SkipButton, same trait.
