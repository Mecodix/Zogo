mod automation;
mod capture;
// TM matcher parked (ORB-only runtime): most of detect/ is still live
// (Hit, regions, template load), this silences only the unused TM fns.
#[allow(dead_code)]
mod detect;
mod orb;
mod tap;

use std::time::Duration;

use automation::{OrbXCloser, ScreenAutomation};
use tap::Tapper;

fn main() -> anyhow::Result<()> {
    let mut tapper = Tapper::new()?;
    // Multi-logic: add more automations here, first hit wins per frame.
    // ORB-only runtime (no template matching): swap OrbXCloser for AdCloser
    // to A/B against the parked Canny matcher.
    let mut jobs: Vec<Box<dyn ScreenAutomation>> = vec![Box::new(OrbXCloser::new()?)];

    println!("sniper v4 live: {} job(s), 4 regions", jobs.len());

    // Capture is the floor here (~318ms PNG on SM-E146B), so no frame budget:
    // each loop costs one screencap no matter what. Back off only when the
    // device stops giving frames at all.
    let mut bad_frames = 0u32;
    loop {
        let gray = match capture::capture_gray() {
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

        let mut fired_cooldown = 0u64;
        for job in jobs.iter_mut() {
            match job.step(&gray) {
                Ok(Some(h)) => {
                    eprintln!(
                        "hit [{}] {:.3} @ {},{} ({})",
                        job.name(),
                        h.score,
                        h.x,
                        h.y,
                        h.region
                    );
                    tapper.tap(h.x, h.y);
                    fired_cooldown = job.cooldown_ms();
                    break; // one tap per frame
                }
                Ok(None) => {}
                Err(e) => eprintln!("job {} err: {:#}", job.name(), e),
            }
        }

        if fired_cooldown > 0 {
            std::thread::sleep(Duration::from_millis(fired_cooldown));
        }
    }
}
