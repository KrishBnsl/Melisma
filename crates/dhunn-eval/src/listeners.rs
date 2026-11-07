//! Simulated listeners reacting to live sessions.
//!
//! Each listener has a hidden taste (see [`crate::taste`]). A session starts from one of
//! their favourite tracks; every pick is played to them, and they skip, finish, like or
//! dislike it according to how much they like it. The recommender only sees those
//! reactions, as it would with a real person. All pickers get the same listeners and the
//! same starting tracks.
//!
//! Before its sessions, each listener gets 60 days of simulated history from the same hidden
//! taste; steering builds the listener's profile and skip model from it, as it would from an
//! imported Spotify history. "no profile" rows show steering without that knowledge.

use std::collections::HashSet;

use dhunn_core::TrackId;
use dhunn_import::{spotify::ExtendedRow, time::parse_timestamp};
use dhunn_index::SoundSpace;
use dhunn_index::places::Places;
use dhunn_profile::{Profile, ProfileConfig, Retainer, SkipModel, history_sessions};
use dhunn_steer::{Feedback, Mode, Session, SessionConfig};
use dhunn_store::{MatchedPlay, Store};
use serde::{Deserialize, Serialize};

use crate::rng::Rng;
use crate::simulate::{SimConfig, SimError, simulate_spotify_history};
use crate::taste::{Taste, TasteSpace, skip_probability};

#[derive(Debug, Clone)]
pub struct ListenerConfig {
    pub listeners: usize,
    pub sessions: usize,
    pub length: usize,
    pub seed: u64,
    /// Steering is run once per adventurousness level.
    pub adventure_levels: Vec<f32>,
    /// Base steering configuration (artist/song keys, parameters).
    pub steer: SessionConfig,
    /// Which features the listeners' hidden tastes are defined in.
    pub judge: TasteSpace,
}

impl Default for ListenerConfig {
    fn default() -> Self {
        Self {
            listeners: 16,
            sessions: 6,
            length: 25,
            seed: 7,
            adventure_levels: vec![0.2, 0.5, 0.8],
            steer: SessionConfig::default(),
            judge: TasteSpace::Dsp,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionRow {
    pub name: String,
    /// Mean taste percentile of the picks (100 = the listener's favourite, 50 = average).
    pub taste: f64,
    pub skip: f64,
    pub like: f64,
    /// Mean sound similarity between consecutive tracks (the seed included).
    pub smooth: f64,
    /// Mean over sessions of the least similar consecutive pair: the session's worst jump.
    pub worst_hop: f64,
    /// Mean `1 − similarity(last pick, seed)`: how far sessions travel from where they began.
    pub travel: f64,
    /// Per listener, distinct tracks over all their picks: low = the same tracks every
    /// session.
    pub variety: f64,
    /// Share of consecutive picks by the same primary artist.
    pub same_artist: f64,
}

/// How each steering mode does (adventurousness 0.5, with the listener's profile).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModeRow {
    pub name: String,
    pub taste: f64,
    pub skip: f64,
    /// Share of picks the listener had never played before.
    pub new_to_you: f64,
    pub smooth: f64,
    pub worst_hop: f64,
    pub travel: f64,
    /// Destinations only: share of sessions that got there (a pick among the
    /// destination's 10 nearest tracks), and how many picks that took on average.
    pub reached: Option<f64>,
    pub picks_to_reach: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ListenerReport {
    /// Which features the listeners' tastes were defined in.
    #[serde(default)]
    pub judge: TasteSpace,
    pub listeners: usize,
    pub sessions: usize,
    pub length: usize,
    pub rows: Vec<SessionRow>,
    #[serde(default)]
    pub modes: Vec<ModeRow>,
}

/// Where a session could travel: the library map, and a destination for "toward" runs.
struct Trip<'a> {
    places: &'a Places,
    target: TrackId,
    /// The destination's 10 nearest tracks: reaching any of them counts as arriving.
    near_target: HashSet<TrackId>,
}

/// What steering knows about a listener: their profile (in `cfg.taste`) and skip model.
struct Known<'a> {
    cfg: SessionConfig,
    retainer: Option<Retainer<'a>>,
}

/// Library plays of a simulated history (the simulator tags them with the track id).
fn library_plays(rows: &[ExtendedRow]) -> Vec<MatchedPlay> {
    rows.iter()
        .filter_map(|r| {
            let id = r
                .spotify_track_uri
                .as_deref()?
                .strip_prefix("spotify:track:dhunnsim")?
                .parse()
                .ok()?;
            Some(MatchedPlay {
                ts: parse_timestamp(&r.ts)?,
                track: TrackId(id),
                ms_played: r.ms_played,
                listened: !r.skipped.unwrap_or(false),
            })
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum Picker {
    Random,
    Nearest,
    /// Adventurousness, and whether the listener's profile and skip model are used.
    Steer(f32, bool),
    /// A steering mode at adventurousness 0.5 with the profile.
    Mode(ModeKind),
}

#[derive(Clone, Copy, PartialEq)]
enum ModeKind {
    Drift,
    Usual,
    Surprise,
    New,
    /// Toward a destination; `false` = head straight there instead of planning a route.
    Toward(bool),
}

impl Picker {
    fn name(self) -> String {
        match self {
            Picker::Random => "random".into(),
            Picker::Nearest => "nearest chain".into(),
            Picker::Steer(a, true) => format!("steer a={a:.1}"),
            Picker::Steer(a, false) => format!("steer a={a:.1}, no profile"),
            Picker::Mode(m) => match m {
                ModeKind::Drift => "drift".into(),
                ModeKind::Usual => "usual".into(),
                ModeKind::Surprise => "surprise".into(),
                ModeKind::New => "something new".into(),
                ModeKind::Toward(true) => "toward (route)".into(),
                ModeKind::Toward(false) => "toward (straight)".into(),
            },
        }
    }
}

#[derive(Default)]
struct Acc {
    picks: usize,
    taste: f64,
    skips: usize,
    likes: usize,
    hops: f64,
    hop_n: usize,
    worst: f64,
    travel: f64,
    sessions: usize,
    same_artist: usize,
    pairs: usize,
    /// Distinct tracks for the current listener.
    distinct: HashSet<TrackId>,
    /// Sum over listeners of distinct tracks.
    distinct_total: usize,
    new_to_you: usize,
    journeys: usize,
    reached: usize,
    reach_picks: usize,
}

/// Picks of one session under `picker`, with the listener's reactions fed back.
#[allow(clippy::too_many_arguments)]
fn run_session(
    space: &SoundSpace,
    taste: &Taste,
    picker: Picker,
    seed_track: TrackId,
    length: usize,
    base: &SessionConfig,
    known: &Known<'_>,
    trip: &Trip<'_>,
    rng_seed: u64,
    acc: &mut Acc,
) {
    let mut react = Rng::new(rng_seed);
    let mut choose = Rng::new(rng_seed ^ 0xA5A5);
    let key = |t: TrackId| base.song_of.get(&t).copied().unwrap_or(t.0 | (1 << 63));
    let mut heard: HashSet<u64> = [key(seed_track)].into();
    let mut session = match picker {
        Picker::Steer(a, with_profile) => {
            let cfg = SessionConfig {
                adventurousness: a,
                seed: rng_seed | 1,
                ..if with_profile {
                    known.cfg.clone()
                } else {
                    base.clone()
                }
            };
            let s = Session::new(space, &[seed_track], cfg);
            match (&known.retainer, with_profile) {
                (Some(r), true) => s.map(|s| s.with_retention(r)),
                _ => s,
            }
        }
        Picker::Mode(m) => {
            let mode = match m {
                ModeKind::Drift => Mode::Drift,
                ModeKind::Usual => Mode::Usual,
                ModeKind::Surprise => Mode::Surprise,
                ModeKind::New => Mode::New,
                ModeKind::Toward(_) => Mode::Toward {
                    target: space
                        .vector(trip.target)
                        .expect("target from space")
                        .to_vec(),
                    label: "destination".into(),
                },
            };
            let cfg = SessionConfig {
                adventurousness: 0.5,
                seed: rng_seed | 1,
                mode,
                plan_route: m != ModeKind::Toward(false),
                ..known.cfg.clone()
            };
            let s = Session::new(space, &[seed_track], cfg).map(|s| s.with_places(trip.places));
            match &known.retainer {
                Some(r) => s.map(|s| s.with_retention(r)),
                None => s,
            }
        }
        _ => None,
    };
    let toward = matches!(picker, Picker::Mode(ModeKind::Toward(_)));
    let mut reached_at: Option<usize> = None;
    let first_pick = acc.picks;
    let mut prev = seed_track;
    let mut worst = f64::INFINITY;
    let mut last = None;
    for _ in 0..length {
        let pick = match picker {
            Picker::Random => {
                let avail: Vec<TrackId> = space
                    .ids()
                    .iter()
                    .copied()
                    .filter(|&t| !heard.contains(&key(t)))
                    .collect();
                avail
                    .get(choose.int(0, avail.len() as i64 - 1).max(0) as usize)
                    .copied()
            }
            Picker::Nearest => space.vector(prev).map(<[f32]>::to_vec).and_then(|v| {
                space
                    .nearest(&v, 1, |t| !heard.contains(&key(t)))
                    .first()
                    .map(|x| x.0)
            }),
            Picker::Steer(..) | Picker::Mode(_) => session
                .as_mut()
                .and_then(|s| s.next_pick())
                .map(|p| p.track),
        };
        let Some(t) = pick else { break };
        heard.insert(key(t));
        let Some(i) = taste.index_of(t) else { break };
        let (aff, pct) = (taste.affinity[i], taste.percentile[i]);

        // The listener reacts.
        let fb = if react.chance(skip_probability(aff)) {
            acc.skips += 1;
            if pct < 0.15 && react.chance(0.3) {
                Feedback::Dislike
            } else {
                Feedback::Skipped {
                    at: react.range(0.03, 0.3) as f32,
                }
            }
        } else if pct >= 0.95 && react.chance(0.5) {
            acc.likes += 1;
            Feedback::Like
        } else {
            Feedback::Finished
        };
        if let Some(s) = session.as_mut() {
            if fb == Feedback::Like {
                s.feedback(t, Feedback::Finished);
            }
            s.feedback(t, fb);
        }

        let hop = space.similarity(prev, t) as f64;
        acc.hops += hop;
        acc.hop_n += 1;
        worst = worst.min(hop);
        if let (Some(a), Some(b)) = (base.artist_of.get(&prev), base.artist_of.get(&t)) {
            acc.same_artist += usize::from(a == b);
        }
        acc.pairs += 1;
        acc.picks += 1;
        acc.taste += pct;
        acc.distinct.insert(t);
        acc.new_to_you += usize::from(!known.cfg.known.contains(&t));
        if toward && reached_at.is_none() && trip.near_target.contains(&t) {
            reached_at = Some(acc.picks);
        }
        prev = t;
        last = Some(t);
    }
    if toward {
        acc.journeys += 1;
        if let Some(at) = reached_at {
            acc.reached += 1;
            acc.reach_picks += at - first_pick + 1;
        }
    }
    if let Some(l) = last {
        acc.sessions += 1;
        acc.worst += worst;
        acc.travel += 1.0 - space.similarity(l, seed_track) as f64;
    }
}

pub fn simulated_listeners(
    store: &Store,
    space: &SoundSpace,
    cfg: &ListenerConfig,
) -> Result<ListenerReport, SimError> {
    let mut pickers = vec![Picker::Random, Picker::Nearest];
    pickers.extend(cfg.adventure_levels.iter().map(|&a| Picker::Steer(a, true)));
    pickers.push(Picker::Steer(0.5, false));
    let main_rows = pickers.len();
    pickers.extend(
        [
            ModeKind::Drift,
            ModeKind::Usual,
            ModeKind::Surprise,
            ModeKind::New,
            ModeKind::Toward(true),
            ModeKind::Toward(false),
        ]
        .map(Picker::Mode),
    );
    let places = Places::build(space);
    let mut accs: Vec<Acc> = pickers.iter().map(|_| Acc::default()).collect();

    for l in 0..cfg.listeners {
        let lseed = cfg.seed.wrapping_mul(1_000_003).wrapping_add(l as u64);
        let mut rng = Rng::new(lseed);
        let taste = Taste::hidden_in(store, cfg.judge, &mut rng, 2)?;
        // The listener's past: the same hidden taste, simulated as 60 days of history.
        let (rows, _, _) = simulate_spotify_history(
            store,
            &SimConfig {
                seed: lseed,
                days: 60,
                taste_regions: 2,
                unowned_share: 0.0,
                respell_share: 0.0,
                taste_space: cfg.judge,
                ..Default::default()
            },
        )?;
        let plays = library_plays(&rows);
        let past = history_sessions(&plays, 30 * 60);
        let now = plays.last().map_or(0, |p| p.ts);
        let profile = Profile::build(
            space,
            &past,
            &cfg.steer.artist_of,
            now,
            &ProfileConfig::default(),
        );
        let model = SkipModel::train(space, &profile, &past);
        let known = Known {
            cfg: SessionConfig {
                taste: profile.centroids(),
                known: profile.tracks.keys().copied().collect(),
                ..cfg.steer.clone()
            },
            retainer: model.as_ref().map(|m| Retainer::new(m, space, &profile)),
        };
        let favs: Vec<TrackId> = taste
            .favourites(0.05)
            .into_iter()
            .filter(|&t| space.contains(t))
            .collect();
        if favs.is_empty() {
            continue;
        }
        for s in 0..cfg.sessions {
            let seed_track = *rng.pick(&favs);
            let run_seed = lseed.wrapping_mul(31).wrapping_add(s as u64 + 1);
            // A destination: somewhere far from the start that the listener feels middling
            // about (not a favourite, not hated).
            let far: Vec<TrackId> = taste
                .ids
                .iter()
                .zip(&taste.percentile)
                .filter(|(t, p)| {
                    (0.3..=0.7).contains(*p) && space.similarity(**t, seed_track) < 0.0
                })
                .map(|(t, _)| *t)
                .collect();
            let target = if far.is_empty() {
                *rng.pick(space.ids())
            } else {
                *rng.pick(&far)
            };
            let tv = space.vector(target).expect("target from space").to_vec();
            let trip = Trip {
                places: &places,
                target,
                near_target: space
                    .nearest(&tv, 10, |_| true)
                    .into_iter()
                    .map(|x| x.0)
                    .collect(),
            };
            for (p, acc) in pickers.iter().zip(&mut accs) {
                run_session(
                    space, &taste, *p, seed_track, cfg.length, &cfg.steer, &known, &trip, run_seed,
                    acc,
                );
            }
        }
        for acc in &mut accs {
            acc.distinct_total += acc.distinct.len();
            acc.distinct.clear();
        }
    }

    let modes = pickers[main_rows..]
        .iter()
        .zip(&accs[main_rows..])
        .map(|(p, a)| {
            let picks = a.picks.max(1) as f64;
            let sessions = a.sessions.max(1) as f64;
            let toward = matches!(p, Picker::Mode(ModeKind::Toward(_)));
            ModeRow {
                name: p.name(),
                taste: 100.0 * a.taste / picks,
                skip: a.skips as f64 / picks,
                new_to_you: a.new_to_you as f64 / picks,
                smooth: a.hops / a.hop_n.max(1) as f64,
                worst_hop: a.worst / sessions,
                travel: a.travel / sessions,
                reached: toward.then(|| a.reached as f64 / a.journeys.max(1) as f64),
                picks_to_reach: (toward && a.reached > 0)
                    .then(|| a.reach_picks as f64 / a.reached as f64),
            }
        })
        .collect();
    let rows = pickers[..main_rows]
        .iter()
        .zip(&accs)
        .map(|(p, a)| {
            let picks = a.picks.max(1) as f64;
            let sessions = a.sessions.max(1) as f64;
            SessionRow {
                name: p.name(),
                taste: 100.0 * a.taste / picks,
                skip: a.skips as f64 / picks,
                like: a.likes as f64 / picks,
                smooth: a.hops / a.hop_n.max(1) as f64,
                worst_hop: a.worst / sessions,
                travel: a.travel / sessions,
                variety: a.distinct_total as f64 / picks,
                same_artist: a.same_artist as f64 / a.pairs.max(1) as f64,
            }
        })
        .collect();
    Ok(ListenerReport {
        judge: cfg.judge,
        listeners: cfg.listeners,
        sessions: cfg.sessions,
        length: cfg.length,
        rows,
        modes,
    })
}
