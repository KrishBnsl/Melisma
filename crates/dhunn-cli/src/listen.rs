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
use dhunn_index::places::Places;
use dhunn_index::{AspectWeights, SoundSpace};
use dhunn_profile::{
    Friend, Profile, Retainer, SkipModel, beyond_me, friend_signal, meeting_point,
};
use dhunn_steer::{Feedback, Goal, Leg, Mode, Pick, Reason, Session, SessionConfig};
use dhunn_store::{EventKind, SessionId, Store, User};

use crate::library::open;

/// How a session is steered (shared by `dhunn session` and `dhunn play`).
#[derive(Debug, Clone, clap::Args)]
pub struct SteerArgs {
    /// Start from these songs (search words; repeatable). Default: your most-played.
    #[arg(long = "seed")]
    pub seeds: Vec<String>,
    /// Who is listening [default: $DHUNN_USER, or the only user].
    #[arg(long, short, env = "DHUNN_USER")]
    pub user: Option<String>,
    /// Override the user's adventurousness (0..1) for this session.
    #[arg(long)]
    pub adventure: Option<f32>,
    /// drift: wander gradually · usual: stay with what you play most · surprise: one big
    /// hop somewhere you haven't explored · new: travel to music you've never played.
    #[arg(long, value_enum, default_value_t = ModeArg::Drift)]
    pub mode: ModeArg,
    /// Travel toward this song or artist in small steps (search words).
    #[arg(long)]
    pub to: Option<String>,
    /// Travel toward a described sound, e.g. "rainy night drive" (needs `dhunn embed`).
    #[arg(long, conflicts_with = "to")]
    pub vibe: Option<String>,
    /// Travel gradually into the part of this friend's taste you don't share yet.
    #[arg(long, conflicts_with_all = ["to", "vibe"])]
    pub friend: Option<String>,
    /// Listening together (repeatable): picks must suit everyone, starting where your
    /// tastes meet.
    #[arg(long = "with")]
    pub with: Vec<String>,
    /// Ignore what friends love (no friend signal).
    #[arg(long)]
    pub solo: bool,
    /// Keep this session out of what friends' recommendations learn from.
    #[arg(long)]
    pub private: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ModeArg {
    Drift,
    Usual,
    Surprise,
    New,
}

/// Don't replay tracks heard in dhunn within this window.
const RECENT_WINDOW_MS: i64 = 36 * 3600 * 1000;

pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// The store plus the sound space built from it. Kept as two values so a session can
/// borrow the space while the player writes events to the store.
pub(crate) fn load(db: &Path) -> Result<(Store, SoundSpace)> {
    let store = open(db)?;
    let space = SoundSpace::from_store_blended(
        &store,
        &AspectWeights::default(),
        dhunn_index::DEFAULT_CLAP_SHARE,
    )?;
    if space.len() < 2 {
        bail!(
            "the library has {} analyzed tracks; run `dhunn analyze <folder>` first",
            space.len()
        );
    }
    Ok((store, space))
}

/// The named user, or the only user if there is exactly one.
pub(crate) fn resolve_user(store: &Store, name: Option<&str>) -> Result<Option<User>> {
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
pub(crate) fn resolve_seeds(
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

pub(crate) fn name(store: &Store, t: TrackId) -> String {
    match store.track(t).ok().flatten() {
        Some(r) => match r.artists.first() {
            Some(a) => format!("{a} – {}", r.title),
            None => r.title,
        },
        None => format!("track {}", t.0),
    }
}

fn summary(store: &Store, t: TrackId) -> Option<serde_json::Value> {
    let json = store
        .track_summary(t, "dsp", dhunn_dsp::DSP_VERSION)
        .ok()
        .flatten()?;
    serde_json::from_str(&json).ok()
}

/// "92 BPM · A minor" from the stored feature summary.
pub(crate) fn facts(store: &Store, t: TrackId) -> String {
    let Some(v) = summary(store, t) else {
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

pub(crate) fn reason_text(store: &Store, r: &Reason) -> String {
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

/// What dhunn knows about a listener, rebuilt from their history and logged sessions.
pub(crate) struct Known {
    pub profile: Profile,
    pub model: Option<SkipModel>,
}

/// Builds the user's profile and skip model (`None` without a user or any plays).
pub(crate) fn know(
    store: &Store,
    space: &SoundSpace,
    user: Option<&User>,
) -> Result<Option<Known>> {
    let Some(u) = user else { return Ok(None) };
    let sessions = crate::friends::sessions_of(store, u, true)?;
    if sessions.is_empty() {
        return Ok(None);
    }
    let profile = crate::friends::profile_from(store, space, &sessions)?;
    let model = SkipModel::train(space, &profile, &sessions);
    Ok(Some(Known { profile, model }))
}

/// A destination from search words: an artist or an album (the centre of its tracks) if
/// the words name one, otherwise the best-matching track.
pub(crate) fn destination(store: &Store, space: &SoundSpace, words: &str) -> Result<Mode> {
    let wanted = dhunn_core::normalize_name(words);
    let artists = store.primary_artists()?;
    if let Some((id, label)) = store
        .artists()?
        .into_iter()
        .find(|(_, n)| dhunn_core::normalize_name(n) == wanted)
    {
        let tracks: Vec<(TrackId, f32)> = artists
            .iter()
            .filter(|(_, a)| **a == id)
            .map(|(&t, _)| (t, 1.0))
            .collect();
        if let Some(target) = space.centroid(&tracks) {
            return Ok(Mode::Toward { target, label });
        }
    }
    // An album whose name contains the words ("sonny boy" finds the soundtrack).
    if wanted.chars().count() >= 3 {
        let mut albums: std::collections::HashMap<String, Vec<(TrackId, f32)>> = Default::default();
        for r in store.browse_rows()? {
            if let Some(a) = r.album
                && dhunn_core::normalize_name(&a).contains(&wanted)
                && space.contains(r.track)
            {
                albums.entry(a).or_default().push((r.track, 1.0));
            }
        }
        // The album with the most analyzed tracks, if several match.
        if let Some((label, tracks)) = albums.into_iter().max_by_key(|(a, t)| (t.len(), a.clone()))
            && let Some(target) = space.centroid(&tracks)
        {
            return Ok(Mode::Toward { target, label });
        }
    }
    let t = find_track(store, space, words)?;
    Ok(Mode::Toward {
        target: space.vector(t).expect("found in space").to_vec(),
        label: name(store, t),
    })
}

/// How many best-matching tracks define a vibe's destination.
const VIBE_TRACKS: usize = 10;

/// A described sound as a destination: the centre of the tracks CLAP thinks match it best,
/// with those matches (best first).
pub(crate) fn vibe_target(space: &SoundSpace, text: &str) -> Result<(Mode, Vec<(TrackId, f32)>)> {
    if !space.has_clap() {
        bail!("vibe search needs learned embeddings: run `dhunn embed` first");
    }
    let clap = crate::embed::text_clap()?;
    let q = clap.embed_text(text)?;
    let top = space.clap_matches(&q, VIBE_TRACKS, |_| true);
    // Weight by how much better than the 10th match each one is, so the best count most.
    let floor = top.last().map_or(0.0, |x| x.1);
    let weighted: Vec<(TrackId, f32)> = top.iter().map(|&(t, s)| (t, s - floor + 0.01)).collect();
    let target = space
        .centroid(&weighted)
        .context("no tracks match that description")?;
    Ok((
        Mode::Toward {
            target,
            label: format!("\"{text}\""),
        },
        top,
    ))
}

/// `dhunn vibe "<text>"`: the tracks that best match a description.
pub fn vibe(db: &Path, text: &str, n: usize) -> Result<()> {
    let (store, space) = load(db)?;
    if !space.has_clap() {
        bail!("vibe search needs learned embeddings: run `dhunn embed` first");
    }
    let clap = crate::embed::text_clap()?;
    let q = clap.embed_text(text)?;
    for (t, s) in space.clap_matches(&q, n, |_| true) {
        println!("{s:>6.3}  {}  [{}]", name(&store, t), facts(&store, t));
    }
    Ok(())
}

/// Friends whose taste this session may use (empty without a user).
pub(crate) fn session_friends(
    store: &Store,
    space: &SoundSpace,
    user: Option<&User>,
    known: Option<&Known>,
) -> Result<Vec<Friend>> {
    match user {
        Some(u) => crate::friends::friends_of(store, space, u, known.map(|k| &k.profile)),
        None => Ok(Vec::new()),
    }
}

pub(crate) fn session_config(
    store: &Store,
    space: &SoundSpace,
    user: Option<&User>,
    known: Option<&Known>,
    friends: &[Friend],
    args: &SteerArgs,
    rng_seed: u64,
) -> Result<SessionConfig> {
    let adventure = args.adventure;
    if user.is_none() && (args.friend.is_some() || !args.with.is_empty()) {
        bail!("who's listening? pass --user <name> or set DHUNN_USER (see `dhunn user list`)");
    }
    let co: Vec<&Friend> = args
        .with
        .iter()
        .map(|w| crate::friends::find_friend(store, friends, w))
        .collect::<Result<_>>()?;
    let mode = match (&args.to, args.mode) {
        _ if args.vibe.is_some() => vibe_target(space, args.vibe.as_deref().unwrap_or(""))?.0,
        (Some(words), _) => destination(store, space, words)?,
        _ if args.friend.is_some() => {
            let f =
                crate::friends::find_friend(store, friends, args.friend.as_deref().unwrap_or(""))?;
            let me = known.map(|k| k.profile.clone()).unwrap_or_default();
            Mode::Toward {
                target: beyond_me(&me, &f.profile).context("they have no taste profile yet")?,
                label: format!("{}'s taste", f.name),
            }
        }
        (None, ModeArg::Drift) if !co.is_empty() => {
            let me = known
                .map(|k| &k.profile)
                .context("a group session needs your own listening history first")?;
            let group: Vec<&Profile> = std::iter::once(me)
                .chain(co.iter().map(|f| &f.profile))
                .collect();
            let names: Vec<&str> = co.iter().map(|f| f.name.as_str()).collect();
            Mode::Toward {
                target: meeting_point(&group).context("no shared taste to start from")?,
                label: format!("where you and {} meet", names.join(" and ")),
            }
        }
        (None, ModeArg::Drift) => Mode::Drift,
        (None, ModeArg::Usual) => Mode::Usual,
        (None, ModeArg::Surprise) => Mode::Surprise,
        (None, ModeArg::New) => Mode::New,
    };
    let mut exclude = match user {
        Some(u) => store.recently_played(u.id, now_ms() - RECENT_WINDOW_MS)?,
        None => HashSet::new(),
    };
    if let Some(k) = known {
        exclude.extend(&k.profile.disliked);
    }
    Ok(SessionConfig {
        adventurousness: adventure.or(user.map(|u| u.adventurousness)).unwrap_or(0.5),
        seed: rng_seed,
        exclude,
        artist_of: store.primary_artists()?,
        song_of: song_keys(store)?,
        taste: known.map(|k| k.profile.centroids()).unwrap_or_default(),
        known: known
            .map(|k| k.profile.tracks.keys().copied().collect())
            .unwrap_or_default(),
        mode,
        friend_signal: if args.solo {
            Default::default()
        } else {
            friend_signal(friends)
        },
        co_listeners: co.iter().map(|f| f.profile.centroids()).collect(),
        ..Default::default()
    })
}

pub(crate) fn retainer<'a>(
    space: &'a SoundSpace,
    known: Option<&'a Known>,
) -> Option<Retainer<'a>> {
    let k = known?;
    Some(Retainer::new(k.model.as_ref()?, space, &k.profile))
}

/// A session, with the library map and (when there is one) the skip model attached.
pub(crate) fn start<'a>(
    space: &'a SoundSpace,
    places: &'a Places,
    seeds: &[TrackId],
    cfg: SessionConfig,
    retainer: Option<&'a Retainer<'a>>,
) -> Result<Session<'a>> {
    let s = Session::new(space, seeds, cfg)
        .context("seeds are not in the sound space")?
        .with_places(places);
    Ok(match retainer {
        Some(r) => s.with_retention(r),
        None => s,
    })
}

/// Where a pick is on a journey, as text ("" when not on one).
pub(crate) fn leg_text(l: Option<&Leg>) -> String {
    let Some(l) = l else {
        return String::new();
    };
    // Stop 0 is where the session started.
    let stop = format!("stop {} of {}", l.step, l.of);
    let arrived = l.step >= l.of;
    match &l.goal {
        Goal::Surprise if l.hop => {
            "surprise: a jump to a part of your library you haven't explored".into()
        }
        Goal::New if arrived && l.first_here => "arrived: music you've never played".into(),
        Goal::Destination(d) if arrived && l.first_here => format!("arrived near {d}"),
        _ if arrived => String::new(),
        Goal::New => format!("heading for music you've never played · {stop}"),
        Goal::Destination(d) => format!("on the way to {d} · {stop}"),
        Goal::Surprise => String::new(),
    }
}

fn mode_line(sess: &Session) -> String {
    match sess.mode() {
        Mode::Toward { label, .. } => format!("toward {label}"),
        m => m.name().to_owned(),
    }
}

/// "· 82% likely you keep listening", when the skip model is attached.
fn keep_text(p: &Pick) -> String {
    p.keep
        .map(|k| format!(" · {:.0}% likely you keep listening", k * 100.0))
        .unwrap_or_default()
}

/// One key per song (normalized primary artist + title), so different files or versions
/// of the same song aren't both played in a session.
pub(crate) fn song_keys(store: &Store) -> Result<std::collections::HashMap<TrackId, u64>> {
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

pub fn session(db: &Path, args: &SteerArgs, len: usize, rng_seed: u64) -> Result<()> {
    let (store, space) = load(db)?;
    let places = Places::build(&space);
    let user = resolve_user(&store, args.user.as_deref())?;
    let seed_ids = resolve_seeds(&store, &space, &args.seeds, user.as_ref())?;
    let known = know(&store, &space, user.as_ref())?;
    let friends = session_friends(&store, &space, user.as_ref(), known.as_ref())?;
    let cfg = session_config(
        &store,
        &space,
        user.as_ref(),
        known.as_ref(),
        &friends,
        args,
        rng_seed,
    )?;
    let retainer = retainer(&space, known.as_ref());
    let mut sess = start(&space, &places, &seed_ids, cfg, retainer.as_ref())?;
    println!(
        "Seeds: {}",
        seed_ids
            .iter()
            .map(|&t| name(&store, t))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "Mode {} · exploration {:.2}\n",
        mode_line(&sess),
        sess.exploration()
    );
    if let Some(v) = &args.vibe {
        let (_, top) = vibe_target(&space, v)?;
        let names: Vec<String> = top.iter().take(3).map(|&(t, _)| name(&store, t)).collect();
        println!("\"{v}\" sounds most like: {}\n", names.join(" · "));
    }
    for i in 1..=len {
        let Some(p) = sess.next_pick() else {
            println!("(library exhausted)");
            break;
        };
        print_pick(&store, &friends, i, &p);
        sess.feedback(p.track, Feedback::Finished);
    }
    Ok(())
}

fn print_pick(store: &Store, friends: &[Friend], i: usize, p: &Pick) {
    println!(
        "{i:>3}. {}  [{}]",
        name(store, p.track),
        facts(store, p.track)
    );
    let leg = leg_text(p.leg.as_ref());
    if !leg.is_empty() {
        println!("     {leg}");
    }
    if let Some(f) = p
        .friend
        .and_then(|(w, s)| crate::friends::friend_text(friends, w, s))
    {
        println!("     {f}");
    }
    if let Some(r) = &p.reason {
        println!("     {}{}", reason_text(store, r), keep_text(p));
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

/// `smoke`: non-interactive check — play this many seconds of a few tracks, then stop.
pub fn play(db: &Path, args: &SteerArgs, smoke: Option<u64>) -> Result<()> {
    if smoke.is_none() && !std::io::stdin().is_terminal() {
        bail!("`dhunn play` needs an interactive terminal");
    }
    let (mut store, space) = load(db)?;
    let places = Places::build(&space);
    let user = resolve_user(&store, args.user.as_deref())?;
    let seed_ids = resolve_seeds(&store, &space, &args.seeds, user.as_ref())?;
    let known = know(&store, &space, user.as_ref())?;
    let friends = session_friends(&store, &space, user.as_ref(), known.as_ref())?;
    let cfg = session_config(
        &store,
        &space,
        user.as_ref(),
        known.as_ref(),
        &friends,
        args,
        now_ms() as u64,
    )?;
    let retainer = retainer(&space, known.as_ref());
    let mut sess = start(&space, &places, &seed_ids, cfg, retainer.as_ref())?;

    let mut device = rodio::DeviceSinkBuilder::open_default_sink()
        .context("opening the default audio output")?;
    device.log_on_drop(false);
    let params = serde_json::json!({
        "seeds": seed_ids.iter().map(|t| t.0).collect::<Vec<_>>(),
        "exploration": sess.exploration(),
        "to": args.to,
        "vibe": args.vibe,
        "friend": args.friend,
        "with": args.with,
        "private": args.private,
    });
    // Automated audio checks are logged under their own mode so eval can ignore them.
    let mode = if smoke.is_some() {
        "smoke"
    } else {
        sess.mode().name()
    };
    let session_id = store.start_session(user.as_ref().map(|u| u.id), mode, &params.to_string())?;
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
    if smoke.is_none() {
        println!(
            "keys: space pause · n next · l like · d dislike · f further · b back · s surprise me · +/- volume · q quit\n"
        );
    }
    let _raw = if smoke.is_none() {
        Some(RawMode::on()?)
    } else {
        None
    };

    let mut played = 0;
    while let Some(pick) = sess.next_pick() {
        if smoke.is_some() && played >= 3 {
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
        let leg = leg_text(pick.leg.as_ref());
        if !leg.is_empty() {
            line(&format!("  {leg}"));
        }
        if let Some(f) = pick
            .friend
            .and_then(|(w, s)| crate::friends::friend_text(&friends, w, s))
        {
            line(&format!("  {f}"));
        }
        if let Some(r) = &pick.reason {
            line(&format!(
                "  {}{}",
                reason_text(p.store, r),
                keep_text(&pick)
            ));
        }

        let after = match smoke {
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
            KeyCode::Char('s') => {
                p.log(Some(t), EventKind::Surprise, Some(pos));
                sess.set_mode(Mode::Surprise);
                line("  ✦ surprise coming up next");
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
