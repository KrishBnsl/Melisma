//! Listening sessions: which track plays next, and how the session reacts to the listener.
//!
//! This is the first, simple steering model (Milestone 1 step 4). A session keeps:
//! * an **anchor** — a point in the sound space the session is "about" right now. It starts
//!   at the seeds and follows what the listener finishes and likes, so sessions drift
//!   gradually instead of jumping;
//! * an **exploration level** (0..1, starts from the user's adventurousness) — how much the
//!   next pick favours sound the session hasn't been near yet over sound close to the
//!   anchor. Together with continuity (similarity to the track just played) this moves the
//!   session forward along a path instead of circling or hopping;
//! * **avoid points** — recently skipped/disliked tracks; candidates near them are penalized.
//!
//! "Further" raises exploration and moves the anchor to where the session is now; "back"
//! lowers it and restores the previous anchor. The full planner (destinations, friend
//! modes, vibe text) replaces the pick rule in step 6; the feedback model carries over.

use std::collections::{HashMap, HashSet};

use dhunn_core::TrackId;
use dhunn_index::{Aspect, SoundSpace, dot, normalize};

#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// The user's adventurousness, 0..1.
    pub adventurousness: f32,
    pub seed: u64,
    /// Tracks not to play in this session (e.g. heard in the last day or two).
    pub exclude: HashSet<TrackId>,
    /// Primary artist per track, to avoid playing the same artist back to back.
    pub artist_of: HashMap<TrackId, i64>,
    /// Longest run of tracks by one artist.
    pub max_artist_run: usize,
    /// Song identity per track: tracks with the same key are versions of one song (a FLAC
    /// and an MP3, a remaster) and are never both played in a session.
    pub song_of: HashMap<TrackId, u64>,
    pub params: SteerParams,
    /// The listener's long-term taste: unit-vector centroids in the sound space (empty =
    /// no profile). Picks close to any of them score higher, less so as exploration rises.
    pub taste: Vec<Vec<f32>>,
}

/// Predicts whether the listener will keep listening to a candidate (the learned skip
/// model lives in `dhunn-profile`; steering only needs this view of it).
pub trait Retention: Sync {
    /// `history`: this session's tracks so far, oldest first, with whether each was
    /// listened to (seeds count as listened). Returns P(listen) in 0..1.
    fn listen_probability(&self, history: &[(TrackId, bool)], candidate: TrackId) -> f32;
}

/// The steering model's tunable constants (what `dhunn eval --param` changes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SteerParams {
    /// Weight of similarity to the previous track, relative to similarity to the anchor.
    pub smoothness: f32,
    /// How far past the current track "further" moves the anchor, as a multiple of the
    /// step from what was heard before to the current track.
    pub onward: f32,
    /// How many recent tracks define "already heard".
    pub heard_memory: usize,
    /// How strongly candidates near recent skips are penalized.
    pub avoid_penalty: f32,
    /// How far a finished track pulls the anchor toward itself.
    pub finish_pull: f32,
    /// How far a liked track pulls the anchor toward itself.
    pub like_pull: f32,
    /// Weight of closeness to the listener's long-term taste (scaled by (1 − exploration)²).
    pub taste_weight: f32,
    /// Weight of the skip model's P(listen) − 0.5 (scaled by 1 − exploration).
    pub retention_weight: f32,
}

impl Default for SteerParams {
    fn default() -> Self {
        Self {
            smoothness: 1.0,
            onward: 1.0,
            heard_memory: 15,
            avoid_penalty: 0.5,
            finish_pull: 0.2,
            like_pull: 0.4,
            taste_weight: 0.5,
            retention_weight: 1.0,
        }
    }
}

impl SteerParams {
    pub const NAMES: [&'static str; 8] = [
        "smoothness",
        "onward",
        "heard_memory",
        "avoid_penalty",
        "finish_pull",
        "like_pull",
        "taste_weight",
        "retention_weight",
    ];

    /// Sets one parameter by name; `Err` lists the valid names.
    pub fn set(&mut self, name: &str, value: f32) -> Result<(), String> {
        match name {
            "smoothness" => self.smoothness = value,
            "onward" => self.onward = value,
            "heard_memory" => self.heard_memory = value.max(1.0).round() as usize,
            "avoid_penalty" => self.avoid_penalty = value,
            "finish_pull" => self.finish_pull = value.clamp(0.0, 1.0),
            "like_pull" => self.like_pull = value.clamp(0.0, 1.0),
            "taste_weight" => self.taste_weight = value,
            "retention_weight" => self.retention_weight = value,
            _ => {
                return Err(format!(
                    "unknown steering parameter `{name}` (one of: {})",
                    Self::NAMES.join(", ")
                ));
            }
        }
        Ok(())
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            adventurousness: 0.5,
            seed: 1,
            exclude: HashSet::new(),
            artist_of: HashMap::new(),
            max_artist_run: 2,
            song_of: HashMap::new(),
            params: SteerParams::default(),
            taste: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Feedback {
    /// Played to (near) the end.
    Finished,
    /// Skipped after this fraction of the track (0..1).
    Skipped {
        at: f32,
    },
    Like,
    Dislike,
    /// "Take me somewhere newer."
    Further,
    /// "That's too far, go back."
    Back,
}

/// Why a track was picked: the earlier track (or seed) it is closest to, and the aspects
/// they share most.
#[derive(Debug, Clone, PartialEq)]
pub struct Reason {
    pub like: TrackId,
    pub shared: Vec<(Aspect, f32)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pick {
    pub track: TrackId,
    /// Similarity to the anchor at pick time.
    pub similarity: f32,
    /// Rank among candidates (0 = closest).
    pub rank: usize,
    pub exploration: f32,
    pub reason: Option<Reason>,
    /// Closeness to the listener's long-term taste (max over taste centroids), if known.
    pub taste: Option<f32>,
    /// The skip model's P(listen), if a model is attached.
    pub keep: Option<f32>,
}

/// Deterministic xorshift for pick sampling.
struct Rng(u64);

impl Rng {
    fn f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

const AVOID_MEMORY: usize = 6;

pub struct Session<'a> {
    space: &'a SoundSpace,
    cfg: SessionConfig,
    anchor: Vec<f32>,
    anchor_stack: Vec<Vec<f32>>,
    base_exploration: f32,
    exploration: f32,
    /// Seeds and tracks the listener finished or liked: what explanations point back to.
    positives: Vec<TrackId>,
    avoid: Vec<(Vec<f32>, f32)>,
    banned: HashSet<TrackId>,
    played: Vec<TrackId>,
    /// Seeds and played tracks with whether each was listened to (for the skip model).
    history: Vec<(TrackId, bool)>,
    retention: Option<&'a dyn Retention>,
    /// Songs already heard in this session (seeds included), by `song_of` key or track id.
    heard_songs: HashSet<u64>,
    rng: Rng,
}

fn song_key(cfg: &SessionConfig, t: TrackId) -> u64 {
    // Tracks without a song key get one of their own (high bit keeps them apart).
    cfg.song_of.get(&t).copied().unwrap_or(t.0 | (1 << 63))
}

/// Maps adventurousness (0..1) to a starting exploration level.
pub fn exploration_for(adventurousness: f32) -> f32 {
    0.1 + 0.5 * adventurousness.clamp(0.0, 1.0)
}

impl<'a> Session<'a> {
    /// Starts at the centre of `seeds` (tracks not in the space are ignored). Returns `None`
    /// if no seed is in the space.
    pub fn new(space: &'a SoundSpace, seeds: &[TrackId], cfg: SessionConfig) -> Option<Self> {
        let weighted: Vec<(TrackId, f32)> = seeds.iter().map(|&t| (t, 1.0)).collect();
        let anchor = space.centroid(&weighted)?;
        let exploration = exploration_for(cfg.adventurousness);
        let seed = cfg.seed.max(1);
        let heard_songs = seeds.iter().map(|&t| song_key(&cfg, t)).collect();
        Some(Self {
            space,
            anchor,
            anchor_stack: Vec::new(),
            base_exploration: exploration,
            exploration,
            positives: seeds
                .iter()
                .copied()
                .filter(|&t| space.contains(t))
                .collect(),
            avoid: Vec::new(),
            banned: HashSet::new(),
            played: Vec::new(),
            history: seeds
                .iter()
                .copied()
                .filter(|&t| space.contains(t))
                .map(|t| (t, true))
                .collect(),
            retention: None,
            heard_songs,
            rng: Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1),
            cfg,
        })
    }

    /// Attaches a skip model: candidates the listener is likely to skip score lower.
    pub fn with_retention(mut self, model: &'a dyn Retention) -> Self {
        self.retention = Some(model);
        self
    }

    fn taste_of(&self, v: &[f32]) -> Option<f32> {
        self.cfg.taste.iter().map(|c| dot(c, v)).reduce(f32::max)
    }

    pub fn exploration(&self) -> f32 {
        self.exploration
    }

    pub fn played(&self) -> &[TrackId] {
        &self.played
    }

    fn artist_run_blocks(&self, t: TrackId) -> bool {
        let Some(a) = self.cfg.artist_of.get(&t) else {
            return false;
        };
        let run = self
            .played
            .iter()
            .rev()
            .take_while(|p| self.cfg.artist_of.get(p) == Some(a))
            .count();
        run >= self.cfg.max_artist_run
    }

    /// Candidates for the next track, best first: `(rank by anchor similarity, track,
    /// anchor similarity, score)`. Artist-run limits are not applied here.
    fn scored(&self, pool: usize) -> Vec<(usize, TrackId, f32, f32)> {
        let fresh = |t: TrackId| {
            !self.heard_songs.contains(&song_key(&self.cfg, t)) && !self.banned.contains(&t)
        };
        let available = |t: TrackId| fresh(t) && !self.cfg.exclude.contains(&t);
        let e = self.exploration.clamp(0.0, 1.0);
        let mut cands = self.space.nearest(&self.anchor, pool, available);
        if cands.is_empty() {
            // Fall back to anything not played in this session (exclusions relaxed).
            cands = self.space.nearest(&self.anchor, pool, fresh);
        }
        // Score = closeness to the anchor (less as exploration rises)
        //       + continuity with the track just played (keeps the session moving one way)
        //       − closeness to what was already heard (more as exploration rises: this is
        //         what makes the session drift toward new sound instead of circling)
        //       − closeness to recent skips.
        let p = &self.cfg.params;
        let last = self
            .played
            .last()
            .and_then(|&t| self.space.vector(t))
            .map(<[f32]>::to_vec);
        let heard: Vec<(TrackId, f32)> = self
            .positives
            .iter()
            .chain(&self.played)
            .rev()
            .take(p.heard_memory)
            .map(|&t| (t, 1.0))
            .collect();
        let heard = self.space.centroid(&heard);
        let mut scored: Vec<(usize, TrackId, f32, f32)> = cands
            .iter()
            .enumerate()
            .map(|(rank, &(t, sim))| {
                let v = self
                    .space
                    .vector(t)
                    .expect("candidates come from the space");
                let smooth = last.as_deref().map_or(sim, |l| dot(l, v));
                let familiar = heard.as_deref().map_or(0.0, |h| dot(h, v));
                let penalty: f32 = self
                    .avoid
                    .iter()
                    .map(|(a, w)| w * dot(a, v).max(0.0))
                    .sum::<f32>()
                    * p.avoid_penalty;
                let taste = self.taste_of(v).unwrap_or(0.0);
                let keep = self
                    .retention
                    .map_or(0.5, |r| r.listen_probability(&self.history, t));
                // The pull of long-term taste fades fast as exploration rises: an adventurous
                // listener (or one who pressed "further") isn't dragged back home.
                let home = 1.0 - e;
                let score = (1.0 - e) * sim + p.smoothness * smooth - e * familiar - penalty
                    + p.taste_weight * home * home * taste
                    + p.retention_weight * home * (keep - 0.5);
                (rank, t, sim, score)
            })
            .collect();
        scored.sort_by(|a, b| b.3.total_cmp(&a.3).then(a.1.cmp(&b.1)));
        scored
    }

    fn pool(&self) -> usize {
        60 + (self.exploration.clamp(0.0, 1.0) * 140.0) as usize
    }

    /// The best `k` next tracks by score, without sampling (for evaluation): what
    /// `next_pick` chooses among.
    pub fn ranked(&self, k: usize) -> Vec<(TrackId, f32)> {
        self.scored(self.pool().max(k))
            .into_iter()
            .take(k)
            .map(|c| (c.1, c.3))
            .collect()
    }

    /// Records that `track` played without this session choosing it (replaying history,
    /// or the listener picking a track themselves). Follow with `feedback`.
    pub fn observe(&mut self, track: TrackId) {
        self.played.push(track);
        self.history.push((track, true));
        self.heard_songs.insert(song_key(&self.cfg, track));
    }

    /// Chooses the next track, or `None` when everything has been played or excluded.
    pub fn next_pick(&mut self) -> Option<Pick> {
        let e = self.exploration.clamp(0.0, 1.0);
        let scored = self.scored(self.pool());
        if scored.is_empty() {
            return None;
        }
        let ok: Vec<&(usize, TrackId, f32, f32)> = scored
            .iter()
            .filter(|c| !self.artist_run_blocks(c.1))
            .collect();
        let ok = if ok.is_empty() {
            scored.iter().collect()
        } else {
            ok
        };
        // Sample among the best few; a little more variety as exploration rises.
        let width = (3 + (e * 5.0) as usize).min(ok.len());
        let weights: Vec<f32> = (0..width).map(|i| 1.0 / (1.0 + i as f32 * 0.5)).collect();
        let total: f32 = weights.iter().sum();
        let mut x = self.rng.f32() * total;
        let mut idx = width - 1;
        for (i, w) in weights.iter().enumerate() {
            x -= w;
            if x < 0.0 {
                idx = i;
                break;
            }
        }
        let &(rank, track, similarity, _) = ok[idx];
        self.played.push(track);
        self.history.push((track, true));
        let v = self.space.vector(track).expect("picked from the space");
        self.heard_songs.insert(song_key(&self.cfg, track));
        Some(Pick {
            track,
            similarity,
            rank,
            exploration: e,
            reason: self.reason(track),
            taste: self.taste_of(v),
            keep: self
                .retention
                .map(|r| r.listen_probability(&self.history[..self.history.len() - 1], track)),
        })
    }

    fn reason(&self, track: TrackId) -> Option<Reason> {
        let like = self
            .positives
            .iter()
            .rev()
            .take(20)
            .filter(|&&p| p != track)
            .max_by(|&&a, &&b| {
                self.space
                    .similarity(track, a)
                    .total_cmp(&self.space.similarity(track, b))
            })
            .copied()?;
        let shared = self
            .space
            .aspect_similarities(track, like)
            .into_iter()
            .take(2)
            .collect();
        Some(Reason { like, shared })
    }

    fn pull(&mut self, track: TrackId, amount: f32) {
        if let Some(v) = self.space.vector(track) {
            self.anchor
                .iter_mut()
                .zip(v)
                .for_each(|(a, b)| *a = (1.0 - amount) * *a + amount * b);
            normalize(&mut self.anchor);
        }
    }

    /// Moves the anchor past `track`, continuing the direction the session has been
    /// travelling (from what was heard before toward `track`).
    fn push_onward(&mut self, track: TrackId) {
        let Some(v) = self.space.vector(track).map(<[f32]>::to_vec) else {
            return;
        };
        let before: Vec<(TrackId, f32)> = self
            .positives
            .iter()
            .chain(&self.played)
            .filter(|&&t| t != track)
            .rev()
            .take(self.cfg.params.heard_memory)
            .map(|&t| (t, 1.0))
            .collect();
        let onward = self.cfg.params.onward;
        match self.space.centroid(&before) {
            Some(h) => {
                self.anchor = v
                    .iter()
                    .zip(&h)
                    .map(|(x, y)| x + onward * (x - y))
                    .collect();
                normalize(&mut self.anchor);
            }
            None => self.pull(track, 0.5),
        }
    }

    fn remember_avoid(&mut self, track: TrackId, weight: f32) {
        if let Some(v) = self.space.vector(track) {
            self.avoid.push((v.to_vec(), weight));
            if self.avoid.len() > AVOID_MEMORY {
                self.avoid.remove(0);
            }
        }
    }

    fn mark_skipped(&mut self, track: TrackId) {
        if let Some(h) = self.history.iter_mut().rev().find(|h| h.0 == track) {
            h.1 = false;
        }
    }

    /// Updates the session after the listener reacts to `track`.
    pub fn feedback(&mut self, track: TrackId, fb: Feedback) {
        match fb {
            Feedback::Finished => {
                self.pull(track, self.cfg.params.finish_pull);
                self.positives.push(track);
                // Settle slowly back toward the user's usual exploration.
                self.exploration += (self.base_exploration - self.exploration) * 0.1;
            }
            Feedback::Like => {
                self.anchor_stack.push(self.anchor.clone());
                self.pull(track, self.cfg.params.like_pull);
                if !self.positives.contains(&track) {
                    self.positives.push(track);
                }
            }
            Feedback::Skipped { at } => {
                // Past halfway, a skip is "enough of this", not "not this".
                if at < 0.5 {
                    self.mark_skipped(track);
                }
                // An early skip is a clear "not this"; a late skip barely says anything.
                let weight = (1.0 - at.clamp(0.0, 1.0)).powi(2);
                if weight > 0.2 {
                    self.remember_avoid(track, weight);
                    self.exploration = (self.exploration - 0.04 * weight).max(0.0);
                }
            }
            Feedback::Dislike => {
                self.mark_skipped(track);
                self.remember_avoid(track, 2.0);
                self.banned.insert(track);
                self.positives.retain(|&p| p != track);
            }
            Feedback::Further => {
                self.anchor_stack.push(self.anchor.clone());
                self.exploration = (self.exploration + 0.2).min(1.0);
                self.push_onward(track);
            }
            Feedback::Back => {
                self.exploration = (self.exploration - 0.25).max(0.0);
                if let Some(prev) = self.anchor_stack.pop() {
                    self.anchor = prev;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dhunn_index::AspectWeights;

    /// 60 tracks on a circle in a 2-D "timbre/rhythm" space: neighbours by id are similar.
    fn ring() -> SoundSpace {
        let names: Vec<String> = ["mfcc00_mean", "log2_tempo"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let rows = (0..60)
            .map(|i| {
                let a = i as f32 / 60.0 * std::f32::consts::TAU;
                (TrackId(i + 1), vec![a.cos(), a.sin()])
            })
            .collect();
        SoundSpace::build(&names, rows, &AspectWeights::default())
    }

    fn dist(a: TrackId, b: TrackId) -> u64 {
        let d = a.0.abs_diff(b.0);
        d.min(60 - d)
    }

    #[test]
    fn stays_close_and_never_repeats() {
        let s = ring();
        let cfg = SessionConfig {
            adventurousness: 0.0,
            ..Default::default()
        };
        let mut sess = Session::new(&s, &[TrackId(1)], cfg).unwrap();
        let first = sess.next_pick().unwrap();
        assert!(dist(first.track, TrackId(1)) <= 3, "{first:?}");
        // 60 tracks minus the seed (already heard) minus the first pick.
        let mut seen = HashSet::new();
        for _ in 0..58 {
            let p = sess.next_pick().unwrap();
            assert!(seen.insert(p.track));
            sess.feedback(p.track, Feedback::Finished);
        }
        assert!(sess.next_pick().is_none(), "library exhausted");
    }

    #[test]
    fn drifts_gradually() {
        let s = ring();
        let mut sess = Session::new(
            &s,
            &[TrackId(1)],
            SessionConfig {
                adventurousness: 0.2,
                ..Default::default()
            },
        )
        .unwrap();
        let mut prev = TrackId(1);
        let mut far = 0;
        for _ in 0..15 {
            let p = sess.next_pick().unwrap();
            assert!(
                dist(p.track, prev) <= 8,
                "jumped from {prev:?} to {:?}",
                p.track
            );
            sess.feedback(p.track, Feedback::Finished);
            far = far.max(dist(p.track, TrackId(1)));
            prev = p.track;
        }
        assert!(
            far >= 6,
            "session should have moved away from the seed, max distance {far}"
        );
    }

    /// Farthest ring distance from the seed reached in `n` picks after applying `fb` to the
    /// first pick, and the exploration level right after that feedback.
    fn reach(fb: &[Feedback], n: usize) -> (u64, f32) {
        let s = ring();
        let cfg = SessionConfig {
            adventurousness: 0.0,
            ..Default::default()
        };
        let mut sess = Session::new(&s, &[TrackId(1)], cfg).unwrap();
        let first = sess.next_pick().unwrap();
        for &f in fb {
            sess.feedback(first.track, f);
        }
        let explored = sess.exploration();
        let mut far = 0;
        let mut path = vec![first.track.0]; // handy when debugging: eprintln!("{path:?}")
        for _ in 0..n {
            let p = sess.next_pick().unwrap();
            sess.feedback(p.track, Feedback::Finished);
            far = far.max(dist(p.track, TrackId(1)));
            path.push(p.track.0);
        }
        let _ = path;
        (far, explored)
    }

    #[test]
    fn further_goes_further_and_back_returns() {
        let (plain, e0) = reach(&[], 6);
        let (further, e1) = reach(&[Feedback::Further, Feedback::Further], 6);
        assert!(e1 > e0 + 0.3, "{e0} → {e1}");
        assert!(further > plain, "further reached {further}, plain {plain}");
        let (_, e2) = reach(
            &[
                Feedback::Further,
                Feedback::Further,
                Feedback::Back,
                Feedback::Back,
            ],
            6,
        );
        assert!(e2 < e0 + 0.05, "{e0} vs {e2}");
    }

    #[test]
    fn skips_push_away_and_dislikes_ban() {
        let s = ring();
        let mut sess = Session::new(
            &s,
            &[TrackId(30)],
            SessionConfig {
                adventurousness: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        // Skip everything on one side of the seed early: picks should move to the other side.
        for t in [31, 32, 33] {
            sess.feedback(TrackId(t), Feedback::Skipped { at: 0.05 });
        }
        let mut sides = (0, 0);
        for _ in 0..4 {
            let p = sess.next_pick().unwrap();
            if p.track.0 > 30 {
                sides.0 += 1
            } else {
                sides.1 += 1
            }
        }
        assert!(sides.1 > sides.0, "{sides:?}");
        sess.feedback(TrackId(29), Feedback::Dislike);
        for _ in 0..50 {
            match sess.next_pick() {
                Some(p) => assert_ne!(p.track, TrackId(29)),
                None => break,
            }
        }
    }

    #[test]
    fn respects_exclusions_and_artist_runs() {
        let s = ring();
        let artist_of: HashMap<TrackId, i64> = (1..=60)
            .map(|i| (TrackId(i), if i <= 30 { 1 } else { i as i64 }))
            .collect();
        let exclude: HashSet<TrackId> = [TrackId(2), TrackId(60)].into();
        let cfg = SessionConfig {
            adventurousness: 0.0,
            exclude,
            artist_of: artist_of.clone(),
            ..Default::default()
        };
        let mut sess = Session::new(&s, &[TrackId(1)], cfg).unwrap();
        let picks: Vec<TrackId> = (0..10).map(|_| sess.next_pick().unwrap().track).collect();
        assert!(!picks.contains(&TrackId(2)) && !picks.contains(&TrackId(60)));
        let longest_run = picks
            .windows(3)
            .filter(|w| w.iter().all(|t| artist_of[t] == 1))
            .count();
        assert_eq!(longest_run, 0, "{picks:?}");
    }

    #[test]
    fn never_replays_seeds_or_other_versions_of_a_song() {
        let s = ring();
        // Tracks 2 and 3 are two versions of one song; 1 is the seed.
        let song_of: HashMap<TrackId, u64> = [(TrackId(2), 7), (TrackId(3), 7)].into();
        let cfg = SessionConfig {
            adventurousness: 0.0,
            song_of,
            ..Default::default()
        };
        let mut sess = Session::new(&s, &[TrackId(1)], cfg).unwrap();
        let picks: Vec<TrackId> = (0..10).map(|_| sess.next_pick().unwrap().track).collect();
        assert!(!picks.contains(&TrackId(1)), "{picks:?}");
        let versions = picks
            .iter()
            .filter(|t| **t == TrackId(2) || **t == TrackId(3))
            .count();
        assert_eq!(versions, 1, "{picks:?}");
    }

    #[test]
    fn explains_picks() {
        let s = ring();
        let mut sess = Session::new(&s, &[TrackId(10)], SessionConfig::default()).unwrap();
        let p = sess.next_pick().unwrap();
        let r = p.reason.unwrap();
        assert_eq!(r.like, TrackId(10));
        assert!(!r.shared.is_empty());
    }

    #[test]
    fn unknown_seed_gives_no_session() {
        assert!(Session::new(&ring(), &[TrackId(999)], SessionConfig::default()).is_none());
    }
}
