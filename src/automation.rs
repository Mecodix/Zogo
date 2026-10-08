use std::collections::HashMap;

use opencv::{core::Mat, core::Point, prelude::*};

use crate::{
    detect::{self, Hit, Tpl},
    orb::{self, OrbMatcher},
};

// One trait per screen job. Hands (capture/detect/tap/orb) are shared,
// each automation holds only its own template + logic.
pub struct Frame {
    pub gray: Mat,
    pub color: Mat,
}

pub trait ScreenAutomation {
    fn name(&self) -> &str;
    fn step(&mut self, frame: &Frame) -> anyhow::Result<Option<Hit>>;
    fn cooldown_ms(&self) -> u64;
    /// Called by main after a tap actually fired, so jobs can budget spots.
    fn note_tapped(&mut self, _x: i32, _y: i32) {}
}

// One grid cell of the screen: after MAX_TAPS_PER_CELL taps that change
// nothing (dead countdown-X, or app chrome shaped like an X), the sniper
// leaves that spot alone for a while instead of spamming the app UI.
const CELL: i32 = 70;
const MAX_TAPS_PER_CELL: u32 = 3;
const BLACKLIST_FRAMES: u32 = 25; // ~8s at 3fps

#[derive(Default)]
struct SpotBudget {
    cells: HashMap<(i32, i32), (u32, u32)>, // cell -> (taps, banned_frames_left)
}

impl SpotBudget {
    fn cell_of(x: i32, y: i32) -> (i32, i32) {
        (x.div_euclid(CELL), y.div_euclid(CELL))
    }

    fn banned(&self, x: i32, y: i32) -> bool {
        self.cells
            .get(&Self::cell_of(x, y))
            .map(|&(_, b)| b > 0)
            .unwrap_or(false)
    }

    fn tick(&mut self) {
        self.cells.retain(|_, v| {
            if v.1 > 0 {
                v.1 -= 1;
            }
            v.0 > 0 || v.1 > 0
        });
        if self.cells.len() > 64 {
            self.cells.clear();
        }
    }

    fn note_tapped(&mut self, x: i32, y: i32) {
        let e = self.cells.entry(Self::cell_of(x, y)).or_insert((0, 0));
        e.0 += 1;
        if e.0 >= MAX_TAPS_PER_CELL {
            e.0 = 0;
            e.1 = BLACKLIST_FRAMES;
        }
    }
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

    fn step(&mut self, frame: &Frame) -> anyhow::Result<Option<Hit>> {
        let cols = frame.gray.cols();
        let rows = frame.gray.rows();

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
                if let Ok(roi_view) = Mat::roi(&frame.gray, r) {
                    let roi: Mat = roi_view.clone_pointee();
                    if let Ok(Some((pt, n))) = self.orb.match_roi(&roi) {
                        let hit = Hit {
                            x: r.x + pt.x,
                            y: r.y + pt.y,
                            score: n as f64,
                            region: "track",
                            tpl: 0,
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
            let roi: Mat = match Mat::roi(&frame.gray, region.rect) {
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
                        tpl: 0,
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
    budget: SpotBudget,
}

impl AdCloser {
    pub fn new() -> anyhow::Result<Self> {
        let mut pyramid = Vec::new();
        for (i, base) in detect::load_templates()?.iter().enumerate() {
            pyramid.extend(detect::build_pyramid(base, i)?);
        }
        if pyramid.is_empty() {
            anyhow::bail!("empty pyramid");
        }
        Ok(Self {
            pyramid,
            last_hit: None,
            misses: 0,
            budget: SpotBudget {
                cells: HashMap::new(),
            },
        })
    }
}

impl AdCloser {
    /// Offline diagnosis: best gated match per region, for `sniper test`.
    pub fn diagnose(&mut self, frame: &Frame) -> Vec<(String, f64, usize, f64, i32, i32)> {
        let cols = frame.gray.cols();
        let rows = frame.gray.rows();
        let mut out = Vec::new();
        for region in detect::default_regions(cols, rows) {
            let entry = match (
                Mat::roi(&frame.gray, region.rect),
                Mat::roi(&frame.color, region.rect),
            ) {
                (Ok(g), Ok(c)) => {
                    let roi = g.clone_pointee();
                    let croi = c.clone_pointee();
                    match detect::canny(&roi) {
                        Ok(re) => match detect::match_roi(&re, &croi, &self.pyramid) {
                            Ok(Some(m)) => (
                                region.name.to_string(),
                                m.score,
                                m.tpl,
                                m.color,
                                region.rect.x + m.loc.x + m.w / 2,
                                region.rect.y + m.loc.y + m.h / 2,
                            ),
                            _ => (region.name.to_string(), 0.0, 99, 999.0, -1, -1),
                        },
                        Err(_) => (region.name.to_string(), 0.0, 99, 999.0, -1, -1),
                    }
                }
                _ => (region.name.to_string(), 0.0, 99, 999.0, -1, -1),
            };
            out.push(entry);
        }
        out
    }
}

impl ScreenAutomation for AdCloser {
    fn name(&self) -> &str {
        "ad_closer"
    }

    fn cooldown_ms(&self) -> u64 {
        800
    }

    fn note_tapped(&mut self, x: i32, y: i32) {
        self.budget.note_tapped(x, y);
        // Freshly rested cell + stale track = instant re-tap loop on app
        // chrome. Drop tracking so the next hit must re-prove itself.
        if self.budget.banned(x, y) {
            self.last_hit = None;
        }
    }

    fn step(&mut self, frame: &Frame) -> anyhow::Result<Option<Hit>> {
        let cols = frame.gray.cols();
        let rows = frame.gray.rows();
        self.budget.tick();

        if let Some(p) = self.last_hit {
            if let Some(r) = detect::clamp_rect(
                p.x - detect::TRACK_SIZE / 2,
                p.y - detect::TRACK_SIZE / 2,
                detect::TRACK_SIZE,
                detect::TRACK_SIZE,
                cols,
                rows,
            ) {
                if let (Ok(gview), Ok(cview)) =
                    (Mat::roi(&frame.gray, r), Mat::roi(&frame.color, r))
                {
                    let roi: Mat = gview.clone_pointee();
                    let croi: Mat = cview.clone_pointee();
                    if let Ok(re) = detect::canny(&roi) {
                        if let Ok(Some(m)) = detect::match_roi(&re, &croi, &self.pyramid) {
                            let hit = Hit {
                                x: r.x + m.loc.x + m.w / 2,
                                y: r.y + m.loc.y + m.h / 2,
                                score: m.score,
                                region: "track",
                                tpl: m.tpl,
                            };
                            if !self.budget.banned(hit.x, hit.y) {
                                self.last_hit = Some(Point::new(hit.x, hit.y));
                                self.misses = 0;
                                return Ok(Some(hit));
                            }
                        }
                    }
                }
            }
        }

        let mut best: Option<Hit> = None;
        for region in detect::default_regions(cols, rows) {
            let roi: Mat = match Mat::roi(&frame.gray, region.rect) {
                Ok(v) => v.clone_pointee(),
                Err(_) => continue,
            };
            let croi: Mat = match Mat::roi(&frame.color, region.rect) {
                Ok(v) => v.clone_pointee(),
                Err(_) => continue,
            };
            let re = match detect::canny(&roi) {
                Ok(m) => m,
                Err(_) => continue,
            };
            match detect::match_roi(&re, &croi, &self.pyramid) {
                Ok(Some(m)) => {
                    let cand = Hit {
                        x: region.rect.x + m.loc.x + m.w / 2,
                        y: region.rect.y + m.loc.y + m.h / 2,
                        score: m.score,
                        region: region.name,
                        tpl: m.tpl,
                    };
                    let better = best
                        .as_ref()
                        .map(|b: &Hit| cand.score > b.score)
                        .unwrap_or(true);
                    if better {
                        let strong = cand.score >= detect::STRONG_CONF;
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

        if let Some(h) = &best {
            if self.budget.banned(h.x, h.y) {
                best = None;
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
