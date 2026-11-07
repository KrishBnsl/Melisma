//! The learned skip model: P(the listener keeps listening to this track, now).
//!
//! Logistic regression on a handful of session features. It trains in milliseconds on a
//! few hundred plays, which is all a single listener has at first; a sequence model
//! (GRU) can replace it once there are thousands of logged sessions and eval shows it wins.
//!
//! Training uses the profile's own counts, so a play's track/artist listen rates are
//! computed *leaving that play out* — otherwise the label leaks into the feature.

use dhunn_core::TrackId;
use dhunn_index::SoundSpace;
use dhunn_steer::Retention;
use serde::{Deserialize, Serialize};

use crate::{Counts, Obs, Profile, auc};

pub const FEATURES: [&str; 9] = [
    "previous track",
    "your taste",
    "your overall taste",
    "track listen rate",
    "track plays",
    "artist listen rate",
    "recent skips",
    "near a recent skip",
    "position",
];
const N: usize = FEATURES.len();
const TASTE: usize = 1;
const OVERALL: usize = 2;
const TRACK_RATE: usize = 3;
const TRACK_PLAYS: usize = 4;
const ARTIST_RATE: usize = 5;

/// Pseudo-plays pulling a sparse listen rate toward the listener's overall rate.
const PRIOR: f32 = 2.0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkipModel {
    mean: [f32; N],
    std: [f32; N],
    /// Per standardized feature; positive = makes a listen more likely.
    pub weights: [f32; N],
    pub bias: f32,
    pub samples: usize,
    /// AUC on the training samples (optimistic; eval measures it on held-out plays).
    pub train_auc: f64,
}

fn log_odds(c: Counts, global: f32) -> f32 {
    let r = ((c.listens + PRIOR * global) / (c.plays + PRIOR)).clamp(0.01, 0.99);
    (r / (1.0 - r)).ln()
}

/// Features of `cand` given the session so far. `leave_out`: the candidate's own play
/// (its label) to subtract from the counts when training.
pub fn features(
    space: &SoundSpace,
    profile: &Profile,
    history: &[(TrackId, bool)],
    cand: TrackId,
    leave_out: Option<bool>,
) -> [f32; N] {
    let minus = |mut c: Counts| {
        if let Some(l) = leave_out {
            c.listens = (c.listens - f32::from(u8::from(l))).max(0.0);
            c.plays = (c.plays - 1.0).max(0.0);
        }
        c
    };
    let v = space.vector(cand);
    let prev = history
        .last()
        .map_or(0.0, |(t, _)| space.similarity(*t, cand));
    let taste = v.and_then(|v| profile.taste_of(v)).unwrap_or(0.0);
    let overall = v.and_then(|v| profile.overall_of(v)).unwrap_or(0.0);
    let track = minus(profile.tracks.get(&cand).copied().unwrap_or_default());
    let artist = profile
        .artist_of
        .get(&cand)
        .and_then(|a| profile.artists.get(a))
        .copied()
        .map(minus)
        .unwrap_or_default();
    let recent: Vec<&(TrackId, bool)> = history.iter().rev().take(3).collect();
    let recent_skips = if recent.is_empty() {
        0.0
    } else {
        recent.iter().filter(|h| !h.1).count() as f32 / recent.len() as f32
    };
    let near_skip = history
        .iter()
        .rev()
        .take(5)
        .filter(|h| !h.1)
        .map(|h| space.similarity(h.0, cand))
        .fold(0.0, f32::max);
    [
        prev,
        taste,
        overall,
        log_odds(track, profile.listen_rate),
        (1.0 + track.plays).ln(),
        log_odds(artist, profile.listen_rate),
        recent_skips,
        near_skip,
        (1.0 + history.len() as f32).ln(),
    ]
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl SkipModel {
    /// Trains on every observation in `sessions` (features from the session before it).
    /// `None` if there are too few plays or no skips/listens to learn from.
    pub fn train(space: &SoundSpace, profile: &Profile, sessions: &[Vec<Obs>]) -> Option<Self> {
        let mut xs: Vec<[f32; N]> = Vec::new();
        let mut ys: Vec<bool> = Vec::new();
        for s in sessions {
            let mut history: Vec<(TrackId, bool)> = Vec::new();
            for o in s.iter().filter(|o| space.contains(o.track)) {
                xs.push(features(
                    space,
                    profile,
                    &history,
                    o.track,
                    Some(o.listened),
                ));
                ys.push(o.listened);
                history.push((o.track, o.listened));
            }
        }
        let pos = ys.iter().filter(|y| **y).count();
        if xs.len() < 30 || pos < 5 || xs.len() - pos < 5 {
            return None;
        }
        let n = xs.len() as f32;
        let mut mean = [0f32; N];
        let mut std = [0f32; N];
        for x in &xs {
            for d in 0..N {
                mean[d] += x[d] / n;
            }
        }
        for x in &xs {
            for d in 0..N {
                std[d] += (x[d] - mean[d]).powi(2) / n;
            }
        }
        std.iter_mut().for_each(|s| *s = s.sqrt().max(1e-4));
        let z: Vec<[f32; N]> = xs
            .iter()
            .map(|x| std::array::from_fn(|d| (x[d] - mean[d]) / std[d]))
            .collect();

        // Full-batch gradient descent with a little L2; deterministic.
        let (mut w, mut b) = ([0f32; N], (pos as f32 / (n - pos as f32)).ln());
        let (lr, l2) = (0.5f32, 1e-3f32);
        for _ in 0..400 {
            let mut gw = [0f32; N];
            let mut gb = 0f32;
            for (x, &y) in z.iter().zip(&ys) {
                let p = sigmoid(b + (0..N).map(|d| w[d] * x[d]).sum::<f32>());
                let e = p - f32::from(u8::from(y));
                for d in 0..N {
                    gw[d] += e * x[d] / n;
                }
                gb += e / n;
            }
            for d in 0..N {
                w[d] -= lr * (gw[d] + l2 * w[d]);
            }
            b -= lr * gb;
        }
        let mut m = SkipModel {
            mean,
            std,
            weights: w,
            bias: b,
            samples: xs.len(),
            train_auc: 0.5,
        };
        let scored: Vec<(f32, bool)> = xs.iter().map(|x| m.predict(x)).zip(ys).collect();
        m.train_auc = auc(&scored).unwrap_or(0.5);
        Some(m)
    }

    pub fn predict(&self, x: &[f32; N]) -> f32 {
        sigmoid(
            self.bias
                + (0..N)
                    .map(|d| self.weights[d] * (x[d] - self.mean[d]) / self.std[d])
                    .sum::<f32>(),
        )
    }
}

/// A trained model plus what it needs to compute features: plugs into a steering session.
pub struct Retainer<'a> {
    pub model: &'a SkipModel,
    pub space: &'a SoundSpace,
    pub profile: &'a Profile,
    /// True predicts best (eval uses it). Steering sets it false: the model then judges
    /// the session (what just played, recent skips) and still steers away from tracks and
    /// artists you tend to skip, but never favours a track for being familiar — that would
    /// replay the same favourites every session — and leaves taste to steering's own term.
    pub familiarity_bonus: bool,
}

impl<'a> Retainer<'a> {
    /// The steering version (no familiarity bonus).
    pub fn new(model: &'a SkipModel, space: &'a SoundSpace, profile: &'a Profile) -> Self {
        Self {
            model,
            space,
            profile,
            familiarity_bonus: false,
        }
    }
}

impl Retention for Retainer<'_> {
    fn listen_probability(&self, history: &[(TrackId, bool)], candidate: TrackId) -> f32 {
        let mut x = features(self.space, self.profile, history, candidate, None);
        if !self.familiarity_bonus {
            let avg = log_odds(Counts::default(), self.profile.listen_rate);
            for d in [TRACK_RATE, ARTIST_RATE] {
                x[d] = x[d].min(avg);
            }
            x[TRACK_PLAYS] = 0.0;
            // Taste is steering's own term (one that fades as exploration rises); counting
            // it here too would pull every session home regardless of adventurousness.
            for d in [TASTE, OVERALL] {
                x[d] = self.model.mean[d];
            }
        }
        self.model.predict(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{obs, ring};
    use std::collections::HashMap;

    #[test]
    fn learns_that_the_listener_skips_one_side() {
        let s = ring();
        // Tracks 1–30 are listened to, 31–60 skipped, in mixed sessions.
        let sessions: Vec<Vec<Obs>> = (0..12)
            .map(|k| {
                (0..10)
                    .map(|i| {
                        let t = (k * 7 + i * 13) % 60 + 1;
                        obs(t, t <= 30)
                    })
                    .collect()
            })
            .collect();
        let p = Profile::build(&s, &sessions, &HashMap::new(), 0, &Default::default());
        let m = SkipModel::train(&s, &p, &sessions).expect("enough data");
        assert!(m.train_auc > 0.85, "{m:?}");
        let r = Retainer {
            familiarity_bonus: true,
            ..Retainer::new(&m, &s, &p)
        };
        let hist = [(TrackId(10), true)];
        assert!(
            r.listen_probability(&hist, TrackId(12)) > r.listen_probability(&hist, TrackId(45)),
            "{m:?}"
        );
    }

    #[test]
    fn steering_version_gives_no_familiarity_bonus() {
        let s = ring();
        // Track 10 is always finished, 40 always skipped; everything else is mixed.
        let mut sessions = Vec::new();
        for k in 0..20u64 {
            sessions.push(vec![
                obs(10, true),
                obs(40, false),
                obs(20 + k % 10, k % 2 == 0),
                obs(50 + k % 5, k % 3 != 0),
            ]);
        }
        let p = Profile::build(&s, &sessions, &HashMap::new(), 0, &Default::default());
        let m = SkipModel::train(&s, &p, &sessions).unwrap();
        let full = Retainer {
            familiarity_bonus: true,
            ..Retainer::new(&m, &s, &p)
        };
        let steer = Retainer::new(&m, &s, &p);
        let t = |r: &Retainer, id| r.listen_probability(&[], TrackId(id));
        assert!(t(&steer, 10) < t(&full, 10), "favourite loses its bonus");
        assert!(
            t(&steer, 40) < t(&steer, 10) && t(&steer, 40) < t(&steer, 22),
            "a track you always skip stays below the rest"
        );
    }

    #[test]
    fn needs_both_outcomes() {
        let s = ring();
        let sessions = vec![(1..=40).map(|t| obs(t, true)).collect::<Vec<_>>()];
        let p = Profile::build(&s, &sessions, &HashMap::new(), 0, &Default::default());
        assert!(SkipModel::train(&s, &p, &sessions).is_none());
    }

    #[test]
    fn leaving_a_play_out_hides_its_label() {
        let s = ring();
        let sessions = vec![vec![obs(5, true), obs(40, false)]];
        let p = Profile::build(&s, &sessions, &HashMap::new(), 0, &Default::default());
        let with = features(&s, &p, &[], TrackId(5), None);
        let without = features(&s, &p, &[], TrackId(5), Some(true));
        assert!(with[3] > without[3]);
        assert_eq!(without[4], 0.0, "no plays left once its own is removed");
    }
}
