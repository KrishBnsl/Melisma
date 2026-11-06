//! `dhunn similar`, `dhunn session` and `dhunn play`.

use std::{
    collections::HashSet,
    fs::File,
    io::{IsTerminal, Write},
    path::Path,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    terminal,
};
use dhunn_core::TrackId;
use dhunn_index::{AspectWeights, SoundSpace};
use dhunn_steer::{Feedback, Pick, Reason, Session, SessionConfig};
use dhunn_store::{EventKind, SessionId, Store, User};

use crate::library::open;

/// Don't replay tracks heard in dhunn within this window.
const RECENT_WINDOW_MS: i64 = 36 * 3600 * 1000;

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// The store plus the sound space built from it. Kept as two values so a session can
/// borrow the space while the player writes events to the store.
fn load(db: &Path) -> Result<(Store, SoundSpace)> {
    let store = open(db)?;
    let space = SoundSpace::from_store(&store, &AspectWeights::default())?;
    if space.len() < 2 {
        bail!(
            "the library has {} analyzed tracks; run `dhunn analyze <folder>` first",
            space.len()
        );
    }
    Ok((store, space))
}

/// The named user, or the only user if there is exactly one.
fn resolve_user(store: &Store, name: Option<&str>) -> Result<Option<User>> {
    match name {
        Some(n) => match store.user_by_name(n)? {
            Some(u) => Ok(Some(u)),
            None => bail!("no user named {n:?}; add one with `dhunn user add {n}`"),
        },
        None => {
            let mut users = store.users()?;
            Ok(if users.len() == 1 { users.pop() } else { None })
        }
    }
}

fn find_track(store: &Store, space: &SoundSpace, query: &str) -> Result<TrackId> {
    let hits = store.search_tracks(query, 50)?;
    match hits.into_iter().find(|h| space.contains(h.track)) {
        Some(h) => Ok(h.track),
        None => bail!("no analyzed track matches {query:?}"),
    }
}

/// Seeds from queries; otherwise the user's most-listened tracks; otherwise a random track.
fn resolve_seeds(
    store: &Store,
    space: &SoundSpace,
    queries: &[String],
    user: Option<&User>,
) -> Result<Vec<TrackId>> {
    if !queries.is_empty() {
        return queries
            .iter()
            .map(|q| find_track(store, space, q))
            .collect();
    }
    if let Some(u) = user {
        let mut stats = store.track_play_stats(u.id)?;
        stats.retain(|s| space.contains(s.0) && s.1 > 0);
        stats.sort_by(|a, b| b.1.cmp(&a.1).then(b.3.cmp(&a.3)));
        if !stats.is_empty() {
            return Ok(stats.iter().take(3).map(|s| s.0).collect());
        }
    }
    let ids = space.ids();
    Ok(vec![ids[(now_ms() as usize / 1000) % ids.len()]])
}

fn name(store: &Store, t: TrackId) -> String {
    match store.track(t).ok().flatten() {
        Some(r) => match r.artists.first() {
            Some(a) => format!("{a} – {}", r.title),
            None => r.title,
        },
        None => format!("track {}", t.0),
    }
}

/// "92 BPM · A minor" from the stored feature summary.
fn facts(store: &Store, t: TrackId) -> String {
    let Some(json) = store
        .track_summary(t, "dsp", dhunn_dsp::DSP_VERSION)
        .ok()
        .flatten()
    else {
        return String::new();
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) else {
        return String::new();
    };
    let tempo = v["rhythm"]["tempo_bpm"]
        .as_f64()
        .map(|b| format!("{b:.0} BPM"));
    let key = v["harmony"]["key_name"].as_str().map(str::to_owned);
    [tempo, key]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" · ")
}

fn reason_text(store: &Store, r: &Reason) -> String {
    let aspects: Vec<&str> = r
        .shared
        .iter()
        .filter(|(_, s)| *s > 0.0)
        .map(|(a, _)| a.label())
        .collect();
    let what = match aspects.as_slice() {
        [] => "close overall".to_owned(),
        [a] => format!("similar {a}"),
        [a, b, ..] => format!("similar {a} and {b}"),
    };
    format!("{what} to {}", name(store, r.like))
}

fn session_config(
    store: &Store,
    user: Option<&User>,
    adventure: Option<f32>,
    rng_seed: u64,
) -> Result<SessionConfig> {
    let exclude = match user {
        Some(u) => store.recently_played(u.id, now_ms() - RECENT_WINDOW_MS)?,
        None => HashSet::new(),
    };
    Ok(SessionConfig {
        adventurousness: adventure.or(user.map(|u| u.adventurousness)).unwrap_or(0.5),
        seed: rng_seed,
        exclude,
        artist_of: store.primary_artists()?,
        song_of: song_keys(store)?,
        ..Default::default()
    })
}

/// One key per song (normalized primary artist + title), so different files or versions
/// of the same song aren't both played in a session.
fn song_keys(store: &Store) -> Result<std::collections::HashMap<TrackId, u64>> {
    use std::hash::{Hash, Hasher};
    let mut out = std::collections::HashMap::new();
    for e in store.library_entries()? {
        let artist = e
            .artists
            .first()
            .map(|a| dhunn_import::matcher::normalize_artist(&a.1))
            .unwrap_or_default();
        let title = dhunn_import::matcher::normalize_title(&e.title);
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (artist, title).hash(&mut h);
        out.insert(e.track, h.finish() & !(1 << 63));
    }
    Ok(out)
}

pub fn similar(db: &Path, query: &str, n: usize) -> Result<()> {
    let (store, space) = load(db)?;
    let seed = find_track(&store, &space, query)?;
    let v = space.vector(seed).expect("found in space").to_vec();
    println!("{}  {}", name(&store, seed), facts(&store, seed));
    println!();
    for (t, sim) in space.nearest(&v, n, |t| t != seed) {
        let shared: Vec<String> = space
            .aspect_similarities(seed, t)
            .iter()
            .take(2)
            .map(|(a, s)| format!("{} {:.0}%", a.label(), s * 100.0))
            .collect();
        println!(
            "  {:>4.0}%  {}  [{}]  {}",
            sim * 100.0,
            name(&store, t),
            facts(&store, t),
            shared.join(", ")
        );
    }
    Ok(())
}

pub fn session(
    db: &Path,
    seeds: &[String],
    user: Option<&str>,
    len: usize,
    adventure: Option<f32>,
    rng_seed: u64,
) -> Result<()> {
    let (store, space) = load(db)?;
    let user = resolve_user(&store, user)?;
    let seed_ids = resolve_seeds(&store, &space, seeds, user.as_ref())?;
    let cfg = session_config(&store, user.as_ref(), adventure, rng_seed)?;
    let mut sess =
        Session::new(&space, &seed_ids, cfg).context("seeds are not in the sound space")?;
    println!(
        "Seeds: {}",
        seed_ids
            .iter()
            .map(|&t| name(&store, t))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("Exploration {:.2}\n", sess.exploration());
    for i in 1..=len {
        let Some(p) = sess.next_pick() else {
            println!("(library exhausted)");
            break;
        };
        print_pick(&store, i, &p);
        sess.feedback(p.track, Feedback::Finished);
    }
    Ok(())
}

fn print_pick(store: &Store, i: usize, p: &Pick) {
    println!(
        "{i:>3}. {}  [{}]",
        name(store, p.track),
        facts(store, p.track)
    );
    if let Some(r) = &p.reason {
        println!("     {}", reason_text(store, r));
    }
}

/// Restores the terminal when dropped, even on error or panic.
struct RawMode;

impl RawMode {
    fn on() -> Result<Self> {
        terminal::enable_raw_mode().context("switching the terminal to raw mode")?;
        Ok(RawMode)
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
    }
}

/// Prints a line in raw mode (which needs explicit carriage returns).
fn line(s: &str) {
    if std::io::stdout().is_terminal() {
        print!("\r\x1b[2K{s}\r\n");
    } else {
        println!("{s}");
    }
    let _ = std::io::stdout().flush();
}

fn clock(d: Duration) -> String {
    let s = d.as_secs();
    format!("{}:{:02}", s / 60, s % 60)
}

enum After {
    Next,
    Quit,
}

struct Player<'a> {
    store: &'a mut Store,
    session: SessionId,
    user: Option<dhunn_core::UserId>,
    player: rodio::Player,
}

impl Player<'_> {
    fn log(&mut self, track: Option<TrackId>, kind: EventKind, pos: Option<Duration>) {
        let pos = pos.map(|d| d.as_millis() as i64);
        if let Err(e) = self
            .store
            .log_event(self.session, self.user, track, kind, pos, None)
        {
            line(&format!("  (could not log {}: {e})", kind.as_str()));
        }
    }
}

pub struct PlayOptions<'a> {
    pub seeds: &'a [String],
    pub user: Option<&'a str>,
    pub adventure: Option<f32>,
    /// Non-interactive check: play this many seconds of a few tracks, then stop.
    pub smoke: Option<u64>,
}

pub fn play(db: &Path, opts: PlayOptions<'_>) -> Result<()> {
    if opts.smoke.is_none() && !std::io::stdin().is_terminal() {
        bail!("`dhunn play` needs an interactive terminal");
    }
    let (mut store, space) = load(db)?;
    let user = resolve_user(&store, opts.user)?;
    let seed_ids = resolve_seeds(&store, &space, opts.seeds, user.as_ref())?;
    let cfg = session_config(&store, user.as_ref(), opts.adventure, now_ms() as u64)?;
    let mut sess =
        Session::new(&space, &seed_ids, cfg).context("seeds are not in the sound space")?;

    let mut device = rodio::DeviceSinkBuilder::open_default_sink()
        .context("opening the default audio output")?;
    device.log_on_drop(false);
    let params = serde_json::json!({
        "seeds": seed_ids.iter().map(|t| t.0).collect::<Vec<_>>(),
        "exploration": sess.exploration(),
    });
    let session_id =
        store.start_session(user.as_ref().map(|u| u.id), "drift", &params.to_string())?;
    let mut p = Player {
        store: &mut store,
        session: session_id,
        user: user.as_ref().map(|u| u.id),
        player: rodio::Player::connect_new(device.mixer()),
    };

    let who = user.as_ref().map(|u| u.name.as_str()).unwrap_or("guest");
    println!(
        "dhunn · {who} · from {} · exploration {:.2}",
        seed_ids
            .iter()
            .map(|&t| name(p.store, t))
            .collect::<Vec<_>>()
            .join(", "),
        sess.exploration()
    );
    if opts.smoke.is_none() {
        println!(
            "keys: space pause · n next · l like · d dislike · f further · b back · +/- volume · q quit\n"
        );
    }
    let _raw = if opts.smoke.is_none() {
        Some(RawMode::on()?)
    } else {
        None
    };

    let mut played = 0;
    while let Some(pick) = sess.next_pick() {
        if opts.smoke.is_some() && played >= 3 {
            break;
        }
        let t = pick.track;
        let Some(path) = p.store.track_path(t)? else {
            p.log(Some(t), EventKind::Error, None);
            continue;
        };
        let source = match File::open(&path)
            .map_err(anyhow::Error::from)
            .and_then(|f| rodio::Decoder::try_from(f).map_err(anyhow::Error::from))
        {
            Ok(s) => s,
            Err(e) => {
                line(&format!("  can't play {path}: {e}"));
                p.log(Some(t), EventKind::Error, None);
                continue;
            }
        };
        let duration = p
            .store
            .track(t)?
            .map(|r| Duration::from_secs_f64(r.duration_s))
            .unwrap_or_default();
        p.player.append(source);
        p.player.play();
        played += 1;
        p.log(Some(t), EventKind::Play, Some(Duration::ZERO));
        line(&format!("▶ {}  [{}]", name(p.store, t), facts(p.store, t)));
        if let Some(r) = &pick.reason {
            line(&format!("  {}", reason_text(p.store, r)));
        }

        let after = match opts.smoke {
            Some(secs) => {
                std::thread::sleep(Duration::from_secs(secs));
                let pos = p.player.get_pos();
                p.player.clear();
                p.log(Some(t), EventKind::Skip, Some(pos));
                line(&format!("  (smoke test: played {:.1}s)", pos.as_secs_f32()));
                After::Next
            }
            None => listen(&mut p, &mut sess, t, duration)?,
        };
        if let After::Quit = after {
            break;
        }
    }
    p.store.end_session(session_id)?;
    drop(_raw);
    println!("\nSession saved ({played} tracks).");
    Ok(())
}

/// Plays the current track until it ends or the listener acts.
fn listen(
    p: &mut Player<'_>,
    sess: &mut Session<'_>,
    t: TrackId,
    duration: Duration,
) -> Result<After> {
    let mut last_draw = Instant::now() - Duration::from_secs(1);
    loop {
        if p.player.empty() {
            p.log(Some(t), EventKind::Complete, Some(duration));
            sess.feedback(t, Feedback::Finished);
            return Ok(After::Next);
        }
        if last_draw.elapsed() >= Duration::from_millis(250) {
            let pos = p.player.get_pos();
            let state = if p.player.is_paused() { "paused" } else { "" };
            print!(
                "\r\x1b[2K  {} / {}  · exploration {:.2}  {state}",
                clock(pos),
                clock(duration),
                sess.exploration()
            );
            let _ = std::io::stdout().flush();
            last_draw = Instant::now();
        }
        if !event::poll(Duration::from_millis(100))? {
            continue;
        }
        let Event::Key(k) = event::read()? else {
            continue;
        };
        if k.kind != KeyEventKind::Press {
            continue;
        }
        let pos = p.player.get_pos();
        let at = if duration.is_zero() {
            0.0
        } else {
            (pos.as_secs_f32() / duration.as_secs_f32()).min(1.0)
        };
        let ctrl_c = k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL);
        match k.code {
            _ if ctrl_c => {
                p.log(Some(t), EventKind::Quit, Some(pos));
                p.player.clear();
                return Ok(After::Quit);
            }
            KeyCode::Char('q') | KeyCode::Esc => {
                p.log(Some(t), EventKind::Quit, Some(pos));
                p.player.clear();
                return Ok(After::Quit);
            }
            KeyCode::Char(' ') => {
                if p.player.is_paused() {
                    p.player.play();
                    p.log(Some(t), EventKind::Resume, Some(pos));
                } else {
                    p.player.pause();
                    p.log(Some(t), EventKind::Pause, Some(pos));
                }
            }
            KeyCode::Char('n') | KeyCode::Right => {
                p.player.clear();
                p.log(Some(t), EventKind::Skip, Some(pos));
                sess.feedback(t, Feedback::Skipped { at });
                line(&format!("  ⏭ skipped at {}", clock(pos)));
                return Ok(After::Next);
            }
            KeyCode::Char('l') => {
                p.log(Some(t), EventKind::Like, Some(pos));
                sess.feedback(t, Feedback::Like);
                line("  ♥ liked — more like this");
            }
            KeyCode::Char('d') => {
                p.player.clear();
                p.log(Some(t), EventKind::Dislike, Some(pos));
                sess.feedback(t, Feedback::Dislike);
                line("  ✕ disliked — won't play it again this session");
                return Ok(After::Next);
            }
            KeyCode::Char('f') => {
                p.log(Some(t), EventKind::Further, Some(pos));
                sess.feedback(t, Feedback::Further);
                line(&format!(
                    "  → going further (exploration {:.2})",
                    sess.exploration()
                ));
            }
            KeyCode::Char('b') => {
                p.log(Some(t), EventKind::Back, Some(pos));
                sess.feedback(t, Feedback::Back);
                line(&format!(
                    "  ← heading back (exploration {:.2})",
                    sess.exploration()
                ));
            }
            KeyCode::Char('+') | KeyCode::Char('=') | KeyCode::Up => {
                p.player.set_volume((p.player.volume() + 0.1).min(2.0));
            }
            KeyCode::Char('-') | KeyCode::Down => {
                p.player.set_volume((p.player.volume() - 0.1).max(0.0));
            }
            _ => {}
        }
    }
}
