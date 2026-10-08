mod automation;
mod capture;
mod detect;
// ORB parked: benchmarks + UI-icon literature say edge-template wins on
// low-texture X icons; ORB stays one line away in main.rs for A/B.
#[allow(dead_code)]
mod orb;
mod tap;

use std::time::Duration;

use automation::{AdCloser, Frame, ScreenAutomation};
use tap::Tapper;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    // Offline diagnosis, zero device writes: scores a saved screenshot so a
    // miss can be debugged with evidence instead of guesses.
    if args.len() > 1 && !args[1].starts_with('-') {
        return test_shot(&args[1]);
    }
    let verbose = args.iter().any(|a| a == "-v");
    if verbose {
        eprintln!("verbose: printing per-frame region scores (slower, debug only)");
    }
    let mut tapper = Tapper::new()?;
    // Multi-logic: add more automations here, first hit wins per frame.
    // Accuracy pick: Canny multi-scale template matcher. ORB is parked
    // (see top of file) after benchmarks showed it starves on flat icons.
    let mut jobs: Vec<Box<dyn ScreenAutomation>> = vec![Box::new(AdCloser::new()?)];

    println!("sniper v5 live: {} job(s), 4 regions", jobs.len());

    // Capture is the floor here (~318ms PNG on SM-E146B), so no frame budget:
    // each loop costs one screencap no matter what. Back off only when the
    // device stops giving frames at all.
    let mut bad_frames = 0u32;
    loop {
        let color = match capture::capture_color() {
            Ok(m) => {
                bad_frames = 0;
                m
            }
            Err(e) => {
                bad_frames += 1;
                eprintln!("bad frame x{}: {:#}", bad_frames, e);
                // 150ms -> 300 -> 600 -> cap 1000ms. Persistent failure
                // (locked screen, dead SurfaceFlinger) must not hot-spin.
                let wait = (150u64 * (1u64 << bad_frames.min(3))).min(1000);
                std::thread::sleep(Duration::from_millis(wait));
                continue;
            }
        };
        let gray = match capture::to_gray(&color) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("bad gray: {:#}", e);
                continue;
            }
        };
        let frame = Frame { gray, color };

        let mut fired_cooldown = 0u64;
        for job in jobs.iter_mut() {
            match job.step(&frame) {
                Ok(Some(h)) => {
                    eprintln!(
                        "hit [{}] {:.3} tpl{} c{:04.1} @ {},{} ({})",
                        job.name(),
                        h.score,
                        h.tpl,
                        h.color,
                        h.x,
                        h.y,
                        h.region
                    );
                    tapper.tap(h.x, h.y);
                    job.note_tapped(h.x, h.y);
                    fired_cooldown = job.cooldown_ms();
                    break; // one tap per frame
                }
                Ok(None) => {
                    if verbose {
                        let t = job.verbose_scan(&frame);
                        if !t.is_empty() {
                            eprintln!("{}", t);
                        }
                    }
                }
                Err(e) => eprintln!("job {} err: {:#}", job.name(), e),
            }
        }

        if fired_cooldown > 0 {
            std::thread::sleep(Duration::from_millis(fired_cooldown));
        }
    }
}

/// `sniper /path/shot.png`: load, scan, print per-region best
/// (score, template, color distance, tap point). No taps, no device writes.
fn test_shot(path: &str) -> anyhow::Result<()> {
    use opencv::{imgcodecs, prelude::*};
    let color = imgcodecs::imread(path, imgcodecs::IMREAD_COLOR)?;
    if color.empty() {
        anyhow::bail!("cannot read {}", path);
    }
    println!("shot: {}x{}", color.cols(), color.rows());
    let gray = capture::to_gray(&color)?;
    let frame = Frame { gray, color };
    let mut job = AdCloser::new()?;
    for (region, score, tpl, cdiff, x, y, raw) in job.diagnose(&frame) {
        println!(
            "{:11} score={:.3} tpl{} color={:6.1} tap={},{} raw=[{}]",
            region, score, tpl, cdiff, x, y, raw
        );
    }
    // Step twice: the stability gate deliberately holds back weak first
    // sightings, so prime it before reading the decision.
    let _ = job.step(&frame)?;
    match job.step(&frame)? {
        Some(h) => println!(
            "DECISION tap {:.3} tpl{} @ {},{} ({})",
            h.score, h.tpl, h.x, h.y, h.region
        ),
        None => println!("DECISION no tap"),
    }
    Ok(())
}
