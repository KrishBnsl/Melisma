//! `dhunn engine`: the recommendation engine for another player (such as the Elsewhere
//! terminal player). The player does the playing; the engine chooses what comes next and
//! learns from what happens. They talk in JSON lines: one request per line on stdin, one
//! reply per line on stdout.
//!
//! Requests (`pos` is seconds into the current song; `path` defaults to the current song):
//!
//! ```text
//! {"op":"hello"}                     who is listening, how big the library is
//! {"op":"next"}                      the engine's next pick
//! {"op":"upcoming","n":5}            what it would play next if you listen through
//! {"op":"played","path":"…"}         a song started (picked by the engine or by you)
//! {"op":"finished"}                  it played to the end
//! {"op":"skip","pos":12.5}           skipped
//! {"op":"choose","pos":12.5}         you chose something else (not counted as a skip)
//! {"op":"like","path":"…"}           liked
//! {"op":"dislike"}                   never again
//! {"op":"further"} {"op":"back"}     wander further out / come back toward home
//! {"op":"surprise"}                  a surprise next
//! {"op":"pause","pos":…} {"op":"resume","pos":…} {"op":"seek","pos":…}
//! {"op":"from","path":"…"}           start choosing from this song (the one playing)
//! {"op":"mode","mode":"usual"}       drift, usual, new or surprise; or
//!   {"op":"mode","to":"…"} / {"op":"mode","vibe":"…"} / {"op":"mode","friend":"…"}
//! {"op":"library"}                   the paths of every song it knows
//! {"op":"likes","paths":["…"]}       songs you like elsewhere (adds the ones it hasn't heard of)
//! {"op":"unlike","path":"…"}         a like taken back
//! {"op":"timecapsule"}              past favorites from the listener's imported history
//! {"op":"profile"}                   what it has learned about you, as lines of text
//! {"op":"quit","pos":…}              end the session (also on end of input)
//! ```
//!
//! Every reply has `"ok"`; failures have `"error"`. A song the engine hasn't analyzed is
//! accepted (`"known":false`) and otherwise ignored.

use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use dhunn_core::TrackId;
use dhunn_index::SoundSpace;
use dhunn_index::places::Places;
use dhunn_profile::Friend;
use dhunn_steer::{Feedback, Mode, Session};
use dhunn_store::{EventKind, SessionId, Store, User};
use serde_json::{Value, json};

use crate::listen::{self, SteerArgs};

struct Engine<'a> {
    store: Store,
    space: &'a SoundSpace,
    sess: Session<'a>,
    known: Option<&'a listen::Known>,
    listener: Option<User>,
    friends: Vec<Friend>,
    session: SessionId,
    user: Option<dhunn_core::UserId>,
    who: String,
    /// The engine's last pick, so a `played` for it isn't taken as your own choice.
    offered: Option<TrackId>,
    /// The song playing now, with its length.
    current: Option<(TrackId, Duration)>,
}

pub fn run(db: &Path, args: &SteerArgs) -> Result<()> {
    let (store, space) = listen::load(db)?;
    let places = Places::build(&space);
    let user = listen::resolve_user(&store, args.user.as_deref())?;
    let seeds = listen::resolve_seeds(&store, &space, &args.seeds, user.as_ref())?;
    let known = listen::know(&store, &space, user.as_ref())?;
    let friends = listen::session_friends(&store, &space, user.as_ref(), known.as_ref())?;
    let cfg = listen::session_config(
        &store,
        &space,
        user.as_ref(),
        known.as_ref(),
        &friends,
        args,
        listen::now_ms() as u64,
    )?;
    let retainer = listen::retainer(&space, known.as_ref());
    let sess = listen::start(&space, &places, &seeds, cfg, retainer.as_ref())?;
    let params = json!({
        "seeds": seeds.iter().map(|t| t.0).collect::<Vec<_>>(),
        "exploration": sess.exploration(),
        "client": "engine",
        "private": args.private,
    });
    let mut store = store;
    let session = store.start_session(
        user.as_ref().map(|u| u.id),
        sess.mode().name(),
        &params.to_string(),
    )?;
    let mut e = Engine {
        store,
        space: &space,
        sess,
        known: known.as_ref(),
        listener: user.clone(),
        friends,
        session,
        user: user.as_ref().map(|u| u.id),
        who: user
            .as_ref()
            .map_or_else(|| "guest".into(), |u| u.name.clone()),
        offered: None,
        current: None,
    };
    let tracks = space.len();

    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let (reply, quit) = match serde_json::from_str::<Value>(&line) {
            Ok(req) => e.handle(&req, tracks),
            Err(err) => (
                json!({"ok": false, "error": format!("bad request: {err}")}),
                false,
            ),
        };
        writeln!(out, "{reply}")?;
        out.flush()?;
        if quit {
            break;
        }
    }
    e.store.end_session(e.session)?;
    Ok(())
}

impl Engine<'_> {
    fn log(&mut self, t: Option<TrackId>, kind: EventKind, pos: Option<Duration>) {
        let pos = pos.map(|d| d.as_millis() as i64);
        if let Err(err) = self
            .store
            .log_event(self.session, self.user, t, kind, pos, None)
        {
            eprintln!("engine: could not log {}: {err}", kind.as_str());
        }
    }

    /// The track for a path: exactly as analyzed, or after resolving links.
    fn track_at(&self, path: &str) -> Option<TrackId> {
        if let Ok(Some(t)) = self.store.track_by_path(path) {
            return Some(t);
        }
        let real = std::fs::canonicalize(path).ok()?;
        self.store
            .track_by_path(&real.to_string_lossy())
            .ok()
            .flatten()
    }

    fn handle(&mut self, req: &Value, tracks: usize) -> (Value, bool) {
        let op = req["op"].as_str().unwrap_or("");
        let pos = req["pos"]
            .as_f64()
            .map(|s| Duration::from_secs_f64(s.max(0.0)));
        // The song a request is about: the given path, or the current song.
        let target = match req["path"].as_str() {
            Some(p) => self.track_at(p),
            None => self.current.map(|c| c.0),
        };
        let ok = json!({"ok": true});
        let reply = match op {
            "hello" => json!({
                "ok": true,
                "user": self.who,
                "tracks": tracks,
                "mode": self.mode(),
                "exploration": self.sess.exploration(),
            }),
            "next" => self.next(),
            "from" => {
                let Some(t) = target else {
                    return (json!({"ok": true, "known": false}), false);
                };
                self.sess.start_from(t);
                json!({"ok": true, "known": true})
            }
            "mode" => match self.set_mode(req) {
                Ok(()) => json!({"ok": true, "mode": self.mode(),
                                 "exploration": self.sess.exploration()}),
                Err(err) => json!({"ok": false, "error": err.to_string()}),
            },
            "timecapsule" => match self.timecapsule() {
                Ok(songs) => json!({"ok": true, "songs": songs}),
                Err(err) => json!({"ok": false, "error": err.to_string()}),
            },
            "library" => match self.store.track_paths() {
                Ok(rows) => {
                    let paths: Vec<String> = rows
                        .into_iter()
                        .filter(|(t, _)| self.space.contains(*t))
                        .map(|(_, p)| p)
                        .collect();
                    json!({"ok": true, "paths": paths})
                }
                Err(err) => json!({"ok": false, "error": err.to_string()}),
            },
            "likes" => {
                let liked = match self.user {
                    Some(u) => self.store.liked_tracks(u).unwrap_or_default(),
                    None => Default::default(),
                };
                let mut added = 0;
                for p in req["paths"].as_array().into_iter().flatten() {
                    if let Some(t) = p.as_str().and_then(|p| self.track_at(p))
                        && !liked.contains(&t)
                    {
                        self.log(Some(t), EventKind::Like, None);
                        added += 1;
                    }
                }
                json!({"ok": true, "added": added})
            }
            "unlike" => {
                if let Some(t) = target {
                    self.log(Some(t), EventKind::Unlike, None);
                }
                ok
            }
            "profile" => match &self.listener {
                Some(u) => match crate::profile::profile_lines(&self.store, self.space, u) {
                    Ok(lines) => json!({"ok": true, "lines": lines}),
                    Err(err) => json!({"ok": false, "error": err.to_string()}),
                },
                None => json!({"ok": false,
                               "error": "no listener: set DHUNN_USER (see `dhunn user list`)"}),
            },
            "upcoming" => self.upcoming(req["n"].as_u64().unwrap_or(5) as usize),
            "played" => {
                let Some(t) = target else {
                    self.current = None;
                    return (json!({"ok": true, "known": false}), false);
                };
                if self.offered != Some(t) {
                    // You chose it yourself: the session follows you there.
                    self.sess.observe(t);
                }
                self.offered = None;
                let length = self
                    .store
                    .track(t)
                    .ok()
                    .flatten()
                    .map_or(Duration::ZERO, |r| {
                        Duration::from_secs_f64(r.duration_s.max(0.0))
                    });
                self.current = Some((t, length));
                self.log(Some(t), EventKind::Play, Some(Duration::ZERO));
                json!({"ok": true, "known": true})
            }
            "finished" => {
                if let Some((t, length)) = self.current.take() {
                    self.log(Some(t), EventKind::Complete, Some(length));
                    self.sess.feedback(t, Feedback::Finished);
                }
                ok
            }
            "skip" => {
                if let Some((t, length)) = self.current.take() {
                    let at = match (pos, length.is_zero()) {
                        (Some(p), false) => (p.as_secs_f32() / length.as_secs_f32()).min(1.0),
                        _ => 0.0,
                    };
                    self.log(Some(t), EventKind::Skip, pos);
                    self.sess.feedback(t, Feedback::Skipped { at });
                }
                ok
            }
            "choose" => {
                if let Some((t, _)) = self.current.take() {
                    self.log(Some(t), EventKind::Choose, pos);
                }
                ok
            }
            "like" | "dislike" | "further" | "back" | "surprise" => {
                let Some(t) = target else {
                    let error = if req["path"].is_string() {
                        "melisma hasn't analyzed this song yet"
                    } else {
                        "nothing is playing"
                    };
                    return (json!({"ok": false, "error": error}), false);
                };
                let message = match op {
                    "like" => {
                        self.log(Some(t), EventKind::Like, pos);
                        self.sess.feedback(t, Feedback::Like);
                        "loved · more like this".to_owned()
                    }
                    "dislike" => {
                        self.log(Some(t), EventKind::Dislike, pos);
                        self.sess.feedback(t, Feedback::Dislike);
                        if self.current.is_some_and(|c| c.0 == t) {
                            self.current = None;
                        }
                        "never again".to_owned()
                    }
                    "further" => {
                        self.log(Some(t), EventKind::Further, pos);
                        self.sess.feedback(t, Feedback::Further);
                        format!("further out · exploration {:.2}", self.sess.exploration())
                    }
                    "back" => {
                        self.log(Some(t), EventKind::Back, pos);
                        self.sess.feedback(t, Feedback::Back);
                        format!(
                            "back toward home · exploration {:.2}",
                            self.sess.exploration()
                        )
                    }
                    _ => {
                        self.log(Some(t), EventKind::Surprise, pos);
                        self.sess.set_mode(Mode::Surprise);
                        "a surprise is coming up next".to_owned()
                    }
                };
                json!({"ok": true, "message": message, "mode": self.mode(),
                       "exploration": self.sess.exploration()})
            }
            "pause" | "resume" | "seek" => {
                let kind = match op {
                    "pause" => EventKind::Pause,
                    "resume" => EventKind::Resume,
                    _ => EventKind::Seek,
                };
                if let Some((t, _)) = self.current {
                    self.log(Some(t), kind, pos);
                }
                ok
            }
            "quit" => {
                if let Some((t, _)) = self.current.take() {
                    self.log(Some(t), EventKind::Quit, pos);
                }
                return (ok, true);
            }
            other => json!({"ok": false, "error": format!("unknown op {other:?}")}),
        };
        (reply, false)
    }

    /// A new direction: a mode by name, or toward a song/artist, a described sound, or
    /// the part of a friend's taste you don't share yet.
    fn set_mode(&mut self, req: &Value) -> Result<()> {
        let mode = if let Some(words) = req["to"].as_str() {
            listen::destination(&self.store, self.space, words)?
        } else if let Some(text) = req["vibe"].as_str() {
            listen::vibe_target(self.space, text)?.0
        } else if let Some(who) = req["friend"].as_str() {
            let f = crate::friends::find_friend(&self.store, &self.friends, who)?;
            let me = self.known.map(|k| k.profile.clone()).unwrap_or_default();
            Mode::Toward {
                target: dhunn_profile::beyond_me(&me, &f.profile)
                    .ok_or_else(|| anyhow::anyhow!("{} has no taste profile yet", f.name))?,
                label: format!("{}'s taste", f.name),
            }
        } else {
            match req["mode"].as_str().unwrap_or("") {
                "drift" => Mode::Drift,
                "usual" => Mode::Usual,
                "new" => Mode::New,
                "surprise" => Mode::Surprise,
                other => anyhow::bail!("unknown mode {other:?} (drift, usual, new, surprise)"),
            }
        };
        self.sess.set_mode(mode);
        Ok(())
    }

    fn timecapsule(&self) -> Result<Vec<Value>> {
        let Some(user) = self.user else {
            return Ok(Vec::new());
        };
        let cutoff = listen::now_ms() - 30 * 24 * 60 * 60 * 1000;
        let recent = self.store.recently_played(user, cutoff)?;
        let mut tracks = self.store.track_play_stats(user)?;
        tracks.retain(|(track, plays, _, ts)| {
            *plays > 0 && *ts <= cutoff / 1000 && !recent.contains(track)
        });
        tracks
            .sort_unstable_by_key(|&(track, plays, _, ts)| (std::cmp::Reverse(plays), ts, track.0));
        let mut songs = Vec::new();
        for (track, plays, _, ts) in tracks {
            if let Some(path) = self.store.track_path(track)? {
                songs.push(json!({"path": path, "play_count": plays, "last_played": ts}));
                if songs.len() == 20 {
                    break;
                }
            }
        }
        Ok(songs)
    }

    fn mode(&self) -> String {
        match self.sess.mode() {
            Mode::Toward { label, .. } => format!("toward {label}"),
            m => m.name().to_owned(),
        }
    }

    fn next(&mut self) -> Value {
        // A few tries, in case a pick's file has gone missing.
        for _ in 0..25 {
            let Some(pick) = self.sess.next_pick() else {
                return json!({"ok": false, "error": "that's the whole library", "done": true});
            };
            let t = pick.track;
            let Some(mut song) = self.describe(&pick) else {
                self.log(Some(t), EventKind::Error, None);
                continue;
            };
            self.offered = Some(t);
            song["ok"] = json!(true);
            song["mode"] = json!(self.mode());
            song["exploration"] = json!(self.sess.exploration());
            return song;
        }
        json!({"ok": false, "error": "couldn't find a playable song"})
    }

    /// What melisma would play next if you listen straight through (a forecast: skipping
    /// or steering changes it).
    fn upcoming(&self, n: usize) -> Value {
        let current = self.current.map(|c| c.0);
        let songs: Vec<Value> = self
            .sess
            .forecast(current, n.clamp(1, 20))
            .iter()
            .filter_map(|p| self.describe(p))
            .collect();
        json!({"ok": true, "songs": songs, "mode": self.mode()})
    }

    /// A pick as JSON: its file, details and reasons. None if the file is gone.
    fn describe(&self, pick: &dhunn_steer::Pick) -> Option<Value> {
        let t = pick.track;
        let path = self.store.track_path(t).ok().flatten()?;
        if !Path::new(&path).exists() {
            return None;
        }
        let row = self.store.track(t).ok().flatten();
        let mut why = Vec::new();
        if let Some(r) = &pick.reason {
            why.push(listen::reason_text(&self.store, r));
        }
        let leg = listen::leg_text(pick.leg.as_ref());
        if !leg.is_empty() {
            why.push(leg);
        }
        if let Some(f) = pick
            .friend
            .and_then(|(w, s)| crate::friends::friend_text(&self.friends, w, s))
        {
            why.push(f);
        }
        if let Some(k) = pick.keep {
            why.push(format!("{:.0}% likely you keep listening", k * 100.0));
        }
        Some(json!({
            "path": path,
            "track": t.0,
            "title": row.as_ref().map(|r| r.title.clone()),
            "artist": row.as_ref().map(|r| r.artists.join(", ")),
            "album": row.as_ref().and_then(|r| r.album.clone()),
            "why": why,
            "leg": pick.leg.as_ref().map(|leg| json!({
                "step": leg.step, "of": leg.of, "hop": leg.hop
            })),
            "facts": listen::facts(&self.store, t),
        }))
    }
}
