use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::time::Duration;

// Reusable hand: tapping. Direct event injection first, su fallback.
// sec_touchscreen = /dev/input/event3, physical max 1080x2408 (1:1 portrait).
// Slot 9 (last of 0..9): real fingers use low slots, so we never steal yours.
const TOUCH_DEV: &str = "/dev/input/event3";
const TAP_SLOT: i32 = 9;
const MAX_X: i32 = 1080;
const MAX_Y: i32 = 2408;
const EV_SYN: u16 = 0;
const EV_KEY: u16 = 1;
const EV_ABS: u16 = 3;
const SYN_REPORT: u16 = 0;
const BTN_TOUCH: u16 = 330;
const ABS_MT_SLOT: u16 = 47;
const ABS_MT_TRACKING_ID: u16 = 57;
const ABS_MT_POSITION_X: u16 = 53;
const ABS_MT_POSITION_Y: u16 = 54;

fn ev_write(f: &mut File, typ: u16, code: u16, value: i32) -> std::io::Result<()> {
    let mut buf = [0u8; 24];
    buf[16..18].copy_from_slice(&typ.to_le_bytes());
    buf[18..20].copy_from_slice(&code.to_le_bytes());
    buf[20..24].copy_from_slice(&value.to_le_bytes());
    f.write_all(&buf)
}

struct DirectTap {
    f: File,
    next_id: i32,
}

impl DirectTap {
    fn open() -> anyhow::Result<Self> {
        let f = OpenOptions::new().write(true).open(TOUCH_DEV)?;
        Ok(Self { f, next_id: 1 })
    }

    fn tap(&mut self, x: i32, y: i32) -> anyhow::Result<()> {
        let x = x.clamp(0, MAX_X - 1);
        let y = y.clamp(0, MAX_Y - 1);
        let id = self.next_id;
        self.next_id = if self.next_id >= 60000 {
            1
        } else {
            self.next_id + 1
        };
        ev_write(&mut self.f, EV_ABS, ABS_MT_SLOT, TAP_SLOT)?;
        ev_write(&mut self.f, EV_ABS, ABS_MT_TRACKING_ID, id)?;
        ev_write(&mut self.f, EV_ABS, ABS_MT_POSITION_X, x)?;
        ev_write(&mut self.f, EV_ABS, ABS_MT_POSITION_Y, y)?;
        ev_write(&mut self.f, EV_KEY, BTN_TOUCH, 1)?;
        ev_write(&mut self.f, EV_SYN, SYN_REPORT, 0)?;
        self.f.flush()?;
        std::thread::sleep(Duration::from_millis(60));
        ev_write(&mut self.f, EV_ABS, ABS_MT_TRACKING_ID, -1)?;
        ev_write(&mut self.f, EV_KEY, BTN_TOUCH, 0)?;
        ev_write(&mut self.f, EV_SYN, SYN_REPORT, 0)?;
        self.f.flush()?;
        Ok(())
    }
}

fn spawn_su() -> anyhow::Result<(Child, BufWriter<ChildStdin>)> {
    let mut child = Command::new("su")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    let stdin = child.stdin.take().expect("su stdin");
    Ok((child, BufWriter::with_capacity(8192, stdin)))
}

pub struct Tapper {
    direct: Option<DirectTap>,
    su_child: Child,
    su_stdin: BufWriter<ChildStdin>,
}

impl Tapper {
    pub fn new() -> anyhow::Result<Self> {
        let direct = match DirectTap::open() {
            Ok(d) => {
                println!("tap: direct {} slot {} (~5ms)", TOUCH_DEV, TAP_SLOT);
                Some(d)
            }
            Err(e) => {
                eprintln!("direct open failed ({}), fallback su+input", e);
                None
            }
        };
        let (su_child, su_stdin) = spawn_su()?;
        Ok(Self {
            direct,
            su_child,
            su_stdin,
        })
    }

    fn ensure_su(&mut self) {
        // su died (OOM / update)? Respawn instead of logging "broken" forever.
        match self.su_child.try_wait() {
            Ok(Some(_)) | Err(_) => match spawn_su() {
                Ok((c, s)) => {
                    self.su_child = c;
                    self.su_stdin = s;
                    eprintln!("su respawned");
                }
                Err(e) => eprintln!("su respawn failed: {:#}", e),
            },
            Ok(None) => {}
        }
    }

    pub fn tap(&mut self, x: i32, y: i32) {
        if let Some(d) = self.direct.as_mut() {
            if d.tap(x, y).is_ok() {
                return;
            }
            eprintln!("direct tap failed, fallback");
        }
        self.ensure_su();
        if writeln!(self.su_stdin, "input tap {} {}", x, y).is_err()
            || self.su_stdin.flush().is_err()
        {
            eprintln!("su pipe broken");
        }
    }
}
