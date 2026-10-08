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
    fn verbose_scan(&mut self, _frame: &Frame) -> String {
        String::new()
    }
    fn name(&self) -> &str;
    fn step(&mut self, frame: &Frame) -> anyhow::Result<Option<Hit>>;
    fn cooldown_ms(&self) -> u64;
    /// Called by main after a tap actually fired, so jobs can budget spots.
    fn note_tapped(&mut self, _x: i32, _y: i32) {}
}

// One grid cell of the screen: after MAX_TAPS_PER_CELL taps that change
// nothing (dead countdown-X, or app chrome shaped like an X), the sniper
// leaves that spot alone for a while instead of spamming the app UI.
// Weak first sightings on a new spot must repeat next frame (kills
// mid-animation taps that land on the ad body and open browsers).
const STABILITY_CONF: f64 = 0.85;
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
                            color: 0.0,
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
                        color: 0.0,
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
    pending: Option<(i32, i32)>,
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
            pending: None,
        })
    }

    /// One ROI through the full pipeline: edges, gated match, tap point.
    /// Collapses the track/regions/diagnose triple copy into short lines.
    fn scan(&self, gray: &Mat, color: &Mat, ox: i32, oy: i32, region: &'static str) -> Option<Hit> {
        let re = detect::canny(gray).ok()?;
        let m = detect::match_roi(&re, gray, color, &self.pyramid).ok()??;
        Some(Hit {
            x: ox + m.loc.x + m.w / 2,
            y: oy + m.loc.y + m.h / 2,
            score: m.score,
            region,
            tpl: m.tpl,
            color: m.color,
        })
    }
}

impl AdCloser {
    /// Offline diagnosis: best gated match per region, for `sniper test`.
    pub fn diagnose(&mut self, frame: &Frame) -> Vec<(String, f64, usize, f64, i32, i32, String)> {
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
                    let raw = detect::canny(&roi)
                        .map(|re| detect::raw_scores(&re, &self.pyramid))
                        .unwrap_or_default()
                        .iter()
                        .map(|(t, v)| format!("t{}:{:.2}", t, v))
                        .collect::<Vec<_>>()
                        .join(" ");
                    match self.scan(&roi, &croi, region.rect.x, region.rect.y, region.name) {
                        Some(m) => (
                            region.name.to_string(),
                            m.score,
                            m.tpl,
                            m.color,
                            m.x,
                            m.y,
                            raw,
                        ),
                        None => (region.name.to_string(), 0.0, 99, 999.0, -1, -1, raw),
                    }
                }
                _ => (
                    region.name.to_string(),
                    0.0,
                    99,
                    999.0,
                    -1,
                    -1,
                    String::new(),
                ),
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

    /// Live debug: same numbers test mode prints, per frame.
    fn verbose_scan(&mut self, frame: &Frame) -> String {
        self.diagnose(frame)
            .iter()
            .map(|(region, score, tpl, cdiff, x, y, raw)| {
                format!(
                    "  {:11} score={:.3} tpl{} color={:6.1} tap={},{} raw=[{}]",
                    region, score, tpl, cdiff, x, y, raw
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn cooldown_ms(&self) -> u64 {
        800
    }

    fn note_tapped(&mut self, x: i32, y: i32) {
        self.budget.note_tapped(x, y);
        self.pending = None;
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
                    if let Some(hit) = self.scan(
                        &gview.clone_pointee(),
                        &cview.clone_pointee(),
                        r.x,
                        r.y,
                        "track",
                    ) {
                        let rested = self.budget.banned(hit.x, hit.y);
                        if !rested || hit.score >= detect::STRONG_CONF {
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
            let roi: Mat = match Mat::roi(&frame.gray, region.rect) {
                Ok(v) => v.clone_pointee(),
                Err(_) => continue,
            };
            let croi: Mat = match Mat::roi(&frame.color, region.rect) {
                Ok(v) => v.clone_pointee(),
                Err(_) => continue,
            };
            if let Some(cand) = self.scan(&roi, &croi, region.rect.x, region.rect.y, region.name) {
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
        }

        // Arbitration, cheapest proof first: rested cells stay resting
        // unless very confident (a real X may spawn where junk lived);
        // weak first sightings on a new spot must repeat next frame.
        let mut fire: Option<Hit> = None;
        if let Some(h) = best {
            let tracked = self
                .last_hit
                .map(|p| (p.x - h.x).abs() < 40 && (p.y - h.y).abs() < 40)
                .unwrap_or(false);
            if self.budget.banned(h.x, h.y) && h.score < detect::STRONG_CONF {
                self.misses += 1;
            } else if !tracked && h.score < STABILITY_CONF {
                let seen = self
                    .pending
                    .map(|(px, py)| (px - h.x).abs() < 40 && (py - h.y).abs() < 40)
                    .unwrap_or(false);
                if seen {
                    self.pending = None;
                    self.last_hit = Some(Point::new(h.x, h.y));
                    self.misses = 0;
                    fire = Some(h);
                } else {
                    self.pending = Some((h.x, h.y));
                    self.misses += 1;
                }
            } else {
                self.last_hit = Some(Point::new(h.x, h.y));
                self.misses = 0;
                fire = Some(h);
            }
        } else {
            self.misses += 1;
        }
        if self.misses >= 5 {
            self.last_hit = None;
            self.misses = 0;
        }
        Ok(fire)
    }
}

// Add job #2 here later, e.g. PlayStorePopupCloser / SkipButton, same trait.
