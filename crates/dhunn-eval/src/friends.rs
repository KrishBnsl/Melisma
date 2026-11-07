//! Simulated friend groups: does knowing what friends love help, and do the friend modes do
//! what they say?
//!
//! Each group has three listeners with hidden tastes of two home tracks each: **A** (whose
//! sessions we run), **B** (shares one home with A — a friend with overlapping taste) and
//! **C** (unrelated homes). Everyone gets 60 days of simulated history, from which steering
//! builds profiles, A's skip model and the friend signal — as it would from imported
//! histories. Sessions start from one of A's favourites; A reacts to every pick (in a
//! bridge session, a track is skipped if either A or B would skip it).

use std::collections::HashSet;

use dhunn_core::TrackId;
use dhunn_index::SoundSpace;
use dhunn_index::places::Places;
use dhunn_profile::{
    Friend, Obs, Profile, ProfileConfig, Retainer, SkipModel, beyond_me, friend_signal,
    history_sessions, meeting_point, overlap,
};
use dhunn_steer::{Feedback, Mode, Session, SessionConfig};
use dhunn_store::Store;
use serde::{Deserialize, Serialize};

use crate::listeners::library_plays;
use crate::rng::Rng;
use crate::simulate::{SimConfig, SimError, simulate_with};
use crate::taste::{Taste, TasteSpace, skip_probability};

#[derive(Debug, Clone)]
pub struct FriendConfig {
    pub groups: usize,
    pub sessions: usize,
    pub length: usize,
    pub seed: u64,
    pub judge: TasteSpace,
    /// Base steering configuration (artist/song keys, parameters).
    pub steer: SessionConfig,
}

impl Default for FriendConfig {
    fn default() -> Self {
        Self {
            groups: 8,
            sessions: 4,
            length: 25,
            seed: 11,
            judge: TasteSpace::Dsp,
            steer: SessionConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FriendRow {
    pub name: String,
    /// Mean taste percentile of the picks for A.
    pub taste: f64,
    pub skip: f64,
    /// Share of picks A had never played.
    pub new_to_you: f64,
    /// Share of picks among B's or C's favourites (their top 10%).
    pub friends_favourites: f64,
    /// Mean taste percentile of the picks for B.
    pub taste_b: f64,
    /// Mean over picks of min(A's, B's percentile): how well picks suit both.
    pub both: f64,
    /// B's mean percentile over each session's last 5 picks: where the session ended up.
    pub b_at_end: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FriendReport {
    pub groups: usize,
    pub sessions: usize,
    pub length: usize,
    /// Mean taste overlap A–B and A–C, as the profiles measured it.
    pub overlap_b: f64,
    pub overlap_c: f64,
    pub rows: Vec<FriendRow>,
}

#[derive(Clone, Copy, PartialEq)]
enum Run {
    Solo,
    WithFriends,
    TowardB,
    /// A's solo steering, but B listens too (the baseline for the bridge).
    SoloTogether,
    Bridge,
}

impl Run {
    const ALL: [Run; 5] = [
        Run::Solo,
        Run::WithFriends,
        Run::TowardB,
        Run::SoloTogether,
        Run::Bridge,
    ];

    fn name(self) -> &'static str {
        match self {
            Run::Solo => "solo",
            Run::WithFriends => "with friend signal",
            Run::TowardB => "toward friend B",
            Run::SoloTogether => "solo, B listening too",
            Run::Bridge => "bridge A + B",
        }
    }

    fn together(self) -> bool {
        matches!(self, Run::SoloTogether | Run::Bridge)
    }
}

#[derive(Default)]
struct Acc {
    picks: usize,
    taste: f64,
    skips: usize,
    new: usize,
    fav: usize,
    taste_b: f64,
    both: f64,
    end_b: f64,
    end_n: usize,
}

struct Person {
    taste: Taste,
    profile: Profile,
    /// Their simulated past, as sessions (what the skip model trains on).
    past: Vec<Vec<Obs>>,
}

/// What a listener with this taste does with a track.
fn react(rng: &mut Rng, taste: &Taste, t: TrackId) -> Feedback {
    let Some(i) = taste.index_of(t) else {
        return Feedback::Finished;
    };
    if rng.chance(skip_probability(taste.affinity[i])) {
        Feedback::Skipped {
            at: rng.range(0.03, 0.3) as f32,
        }
    } else {
        Feedback::Finished
    }
}

fn person(
    store: &Store,
    space: &SoundSpace,
    taste: Taste,
    seed: u64,
    artist_of: &std::collections::HashMap<TrackId, i64>,
) -> Result<Person, SimError> {
    let cfg = SimConfig {
        seed,
        days: 60,
        unowned_share: 0.0,
        respell_share: 0.0,
        ..Default::default()
    };
    let mut rng = Rng::new(seed);
    let (rows, _, _) = simulate_with(store, &cfg, &taste, &mut rng)?;
    let plays = library_plays(&rows);
    let now = plays.last().map_or(0, |p| p.ts);
    let past = history_sessions(&plays, 30 * 60);
    let profile = Profile::build(space, &past, artist_of, now, &ProfileConfig::default());
    Ok(Person {
        taste,
        profile,
        past,
    })
}

pub fn friend_groups(
    store: &Store,
    space: &SoundSpace,
    cfg: &FriendConfig,
) -> Result<FriendReport, SimError> {
    let places = Places::build(space);
    let mut accs: Vec<Acc> = Run::ALL.iter().map(|_| Acc::default()).collect();
    let (mut ob, mut oc, mut groups) = (0.0, 0.0, 0usize);
    let artist_of = &cfg.steer.artist_of;

    for g in 0..cfg.groups {
        let gseed = cfg.seed.wrapping_mul(7_919).wrapping_add(g as u64);
        let mut rng = Rng::new(gseed);
        let base = Taste::hidden_in(store, cfg.judge, &mut rng, 1)?;
        let n = base.ids.len();
        let mut homes: Vec<usize> = Vec::new();
        while homes.len() < 5.min(n) {
            let i = rng.int(0, n as i64 - 1) as usize;
            if !homes.contains(&i) {
                homes.push(i);
            }
        }
        let [h1, h2, h3, h4, h5] = [homes[0], homes[1], homes[2], homes[3], homes[4]];
        let a = person(
            store,
            space,
            base.with_homes(vec![h1, h2]),
            gseed ^ 1,
            artist_of,
        )?;
        let b = person(
            store,
            space,
            base.with_homes(vec![h2, h3]),
            gseed ^ 2,
            artist_of,
        )?;
        let c = person(
            store,
            space,
            base.with_homes(vec![h4, h5]),
            gseed ^ 3,
            artist_of,
        )?;
        if a.profile.regions.is_empty() || b.profile.regions.is_empty() {
            continue;
        }
        groups += 1;
        let o_b = overlap(&a.profile, &b.profile);
        let o_c = overlap(&a.profile, &c.profile);
        ob += o_b as f64;
        oc += o_c as f64;
        let friends = vec![
            Friend {
                name: "B".into(),
                profile: b.profile.clone(),
                overlap: o_b,
            },
            Friend {
                name: "C".into(),
                profile: c.profile.clone(),
                overlap: o_c,
            },
        ];
        let signal = friend_signal(&friends);
        let fav = |p: &Person, t: TrackId| p.taste.percentile_of(t).is_some_and(|x| x >= 0.9);
        let model = SkipModel::train(space, &a.profile, &a.past);
        let retainer = model.as_ref().map(|m| Retainer::new(m, space, &a.profile));
        let known: HashSet<TrackId> = a.profile.tracks.keys().copied().collect();
        let a_cfg = SessionConfig {
            adventurousness: 0.5,
            taste: a.profile.centroids(),
            known: known.clone(),
            exclude: a.profile.disliked.clone(),
            ..cfg.steer.clone()
        };
        let favs: Vec<TrackId> = a
            .taste
            .favourites(0.05)
            .into_iter()
            .filter(|&t| space.contains(t))
            .collect();
        if favs.is_empty() {
            continue;
        }

        for s in 0..cfg.sessions {
            let seed_track = *rng.pick(&favs);
            let run_seed = gseed.wrapping_mul(31).wrapping_add(s as u64 + 1);
            for (run, acc) in Run::ALL.iter().zip(&mut accs) {
                let mut sc = SessionConfig {
                    seed: run_seed | 1,
                    ..a_cfg.clone()
                };
                match run {
                    Run::Solo | Run::SoloTogether => {}
                    Run::WithFriends => sc.friend_signal = signal.clone(),
                    Run::TowardB => {
                        sc.friend_signal = signal.clone();
                        if let Some(t) = beyond_me(&a.profile, &b.profile) {
                            sc.mode = Mode::Toward {
                                target: t,
                                label: "B".into(),
                            };
                        }
                    }
                    Run::Bridge => {
                        sc.co_listeners = vec![b.profile.centroids()];
                        if let Some(t) = meeting_point(&[&a.profile, &b.profile]) {
                            sc.mode = Mode::Toward {
                                target: t,
                                label: "A + B".into(),
                            };
                        }
                    }
                }
                let Some(mut sess) = Session::new(space, &[seed_track], sc) else {
                    continue;
                };
                sess = sess.with_places(&places);
                if let Some(r) = &retainer {
                    sess = sess.with_retention(r);
                }
                let mut react_a = Rng::new(run_seed);
                let mut react_b = Rng::new(run_seed ^ 0xB);
                let mut b_scores: Vec<f64> = Vec::new();
                for _ in 0..cfg.length {
                    let Some(p) = sess.next_pick() else { break };
                    let t = p.track;
                    let pa = a.taste.percentile_of(t).unwrap_or(0.5);
                    let pb = b.taste.percentile_of(t).unwrap_or(0.5);
                    let mut fb = react(&mut react_a, &a.taste, t);
                    if run.together() {
                        let fb_b = react(&mut react_b, &b.taste, t);
                        if matches!(fb_b, Feedback::Skipped { .. }) {
                            fb = fb_b;
                        }
                    }
                    acc.skips += usize::from(matches!(fb, Feedback::Skipped { .. }));
                    sess.feedback(t, fb);
                    acc.picks += 1;
                    acc.taste += pa;
                    acc.taste_b += pb;
                    acc.both += pa.min(pb);
                    acc.new += usize::from(!known.contains(&t));
                    acc.fav += usize::from(fav(&b, t) || fav(&c, t));
                    b_scores.push(pb);
                }
                for x in b_scores.iter().rev().take(5) {
                    acc.end_b += x;
                    acc.end_n += 1;
                }
            }
        }
    }

    let rows = Run::ALL
        .iter()
        .zip(&accs)
        .map(|(r, a)| {
            let n = a.picks.max(1) as f64;
            FriendRow {
                name: r.name().into(),
                taste: 100.0 * a.taste / n,
                skip: a.skips as f64 / n,
                new_to_you: a.new as f64 / n,
                friends_favourites: a.fav as f64 / n,
                taste_b: 100.0 * a.taste_b / n,
                both: 100.0 * a.both / n,
                b_at_end: 100.0 * a.end_b / a.end_n.max(1) as f64,
            }
        })
        .collect();
    let g = groups.max(1) as f64;
    Ok(FriendReport {
        groups,
        sessions: cfg.sessions,
        length: cfg.length,
        overlap_b: ob / g,
        overlap_c: oc / g,
        rows,
    })
}
