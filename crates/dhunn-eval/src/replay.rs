//! Next-track prediction on real (or sample) listening history.
//!
//! The user's matched plays are split by time: everything before the last `holdout_days` is
//! "train" (what the recommender may know), the rest is "test". Test plays are cut into
//! listening sessions; at every step of a session each model ranks the whole library given
//! what has played so far in that session, and we record where the track the user actually
//! listened to next landed. Skipped next tracks are not targets (the user didn't want them),
//! but they still play into the session, as they did in reality.
//!
//! Caveat: on simulated history the "previous track" model has an edge by construction (the
//! simulator picks by similarity to the previous track). On real history it's a fair race.

use std::collections::{HashMap, HashSet};

use dhunn_core::{TrackId, UserId};
use dhunn_index::{SoundSpace, dot};
use dhunn_steer::{Feedback, Session, SessionConfig};
use dhunn_store::{MatchedPlay, Store, StoreError};
use serde::{Deserialize, Serialize};

/// Ranks beyond this count as misses (for MRR and the steering model's candidate list).
pub const MAX_RANK: usize = 100;

#[derive(Debug, Clone)]
pub struct ReplayConfig {
    pub holdout_days: u32,
    /// A pause longer than this (beyond the track's own length) starts a new session.
    pub session_gap_s: i64,
    /// Base steering configuration (artist/song keys, parameters, adventurousness).
    pub steer: SessionConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankRow {
    pub name: String,
    /// Share of steps where the next track was in the model's top 10.
    pub hit10: f64,
    pub hit50: f64,
    /// Mean reciprocal rank (0 past rank 100).
    pub mrr: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayReport {
    pub holdout_days: u32,
    pub train_listens: usize,
    pub test_plays: usize,
    pub test_sessions: usize,
    /// Predictions made (test plays that were listened to and new in their session).
    pub steps: usize,
    pub rows: Vec<RankRow>,
    /// How well a model's score separates listened from skipped test plays (0.5 = chance).
    pub skip_auc: Vec<(String, f64)>,
}

#[derive(Default)]
struct Acc {
    hit10: f64,
    hit50: f64,
    rr: f64,
    n: usize,
}

impl Acc {
    /// `rank` is 1-based and may be fractional (expected rank under ties); `None` = miss.
    fn add(&mut self, rank: Option<f64>) {
        self.n += 1;
        if let Some(r) = rank {
            self.hit10 += f64::from(r <= 10.0);
            self.hit50 += f64::from(r <= 50.0);
            if r <= MAX_RANK as f64 {
                self.rr += 1.0 / r;
            }
        }
    }

    fn row(&self, name: &str) -> RankRow {
        let n = self.n.max(1) as f64;
        RankRow {
            name: name.to_owned(),
            hit10: self.hit10 / n,
            hit50: self.hit50 / n,
            mrr: self.rr / n,
        }
    }
}

/// Expected 1-based rank of `target` when tracks are ordered by `score` (ties broken at
/// random), among `ids` not in `skip`.
fn rank_by(
    ids: &[TrackId],
    skip: &HashSet<TrackId>,
    target: TrackId,
    score: impl Fn(TrackId) -> f32,
) -> f64 {
    let s = score(target);
    let (mut above, mut ties) = (0usize, 0usize);
    for &t in ids {
        if t == target || skip.contains(&t) {
            continue;
        }
        let x = score(t);
        if x > s {
            above += 1;
        } else if x == s {
            ties += 1;
        }
    }
    1.0 + above as f64 + ties as f64 / 2.0
}

/// Area under the ROC curve: probability a random positive outscores a random negative.
pub fn auc(scored: &[(f32, bool)]) -> Option<f64> {
    let mut v = scored.to_vec();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let pos = v.iter().filter(|x| x.1).count();
    let neg = v.len() - pos;
    if pos == 0 || neg == 0 {
        return None;
    }
    // Rank sum of positives, with average ranks for ties.
    let mut rank_sum = 0.0;
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j < v.len() && v[j].0 == v[i].0 {
            j += 1;
        }
        let avg = (i + j + 1) as f64 / 2.0; // ranks i+1..=j
        rank_sum += avg * v[i..j].iter().filter(|x| x.1).count() as f64;
        i = j;
    }
    Some((rank_sum - (pos * (pos + 1)) as f64 / 2.0) / (pos * neg) as f64)
}

fn sessions(plays: &[MatchedPlay], gap_s: i64) -> Vec<Vec<MatchedPlay>> {
    let mut out: Vec<Vec<MatchedPlay>> = Vec::new();
    for p in plays {
        let new = match out.last().and_then(|s| s.last()) {
            // `ts` is when a play ended: a session continues if this play started soon after.
            Some(prev) => p.ts - p.ms_played / 1000 - prev.ts > gap_s,
            None => true,
        };
        if new {
            out.push(Vec::new());
        }
        out.last_mut().expect("pushed").push(*p);
    }
    out
}

pub fn replay(
    store: &Store,
    space: &SoundSpace,
    user: UserId,
    cfg: &ReplayConfig,
) -> Result<Option<ReplayReport>, StoreError> {
    let plays: Vec<MatchedPlay> = store
        .matched_plays(user)?
        .into_iter()
        .filter(|p| space.contains(p.track))
        .collect();
    let Some(last) = plays.last() else {
        return Ok(None);
    };
    let cutoff = last.ts - cfg.holdout_days as i64 * 86_400;
    let (train, test): (Vec<MatchedPlay>, Vec<MatchedPlay>) =
        plays.iter().partition(|p| p.ts < cutoff);

    let mut listens: HashMap<TrackId, f32> = HashMap::new();
    for p in train.iter().filter(|p| p.listened) {
        *listens.entry(p.track).or_default() += 1.0;
    }
    let train_listens = listens.values().sum::<f32>() as usize;
    let taste_pts: Vec<(TrackId, f32)> = listens.iter().map(|(&t, &w)| (t, w)).collect();
    let taste = space.centroid(&taste_pts);

    let mut durations: HashMap<TrackId, f64> = HashMap::new();
    for p in &test {
        if let std::collections::hash_map::Entry::Vacant(e) = durations.entry(p.track) {
            let d = store.track(p.track)?.map_or(0.0, |t| t.duration_s);
            e.insert(d);
        }
    }

    let ids = space.ids();
    let sim = |a: TrackId, b: TrackId| space.similarity(a, b);
    let taste_score = |t: TrackId| match (&taste, space.vector(t)) {
        (Some(c), Some(v)) => dot(c, v),
        _ => 0.0,
    };

    let mut acc: [Acc; 5] = Default::default();
    let names = ["random", "popular", "your taste", "previous track", "steer"];
    let sess = sessions(&test, cfg.session_gap_s);
    let mut skip_taste = Vec::new();
    let mut skip_prev = Vec::new();
    for s in &sess {
        let mut played: HashSet<TrackId> = HashSet::new();
        let mut steer: Option<Session> = None;
        for (i, p) in s.iter().enumerate() {
            skip_taste.push((taste_score(p.track), p.listened));
            if i > 0 {
                skip_prev.push((sim(s[i - 1].track, p.track), p.listened));
            }
            if i > 0 && p.listened && !played.contains(&p.track) {
                let avail = ids.len() - played.len();
                // Random ranking: hit@k = k / available, E[1/rank] = H(min(N, 100)) / N.
                let h: f64 = (1..=avail.min(MAX_RANK)).map(|r| 1.0 / r as f64).sum();
                acc[0].n += 1;
                acc[0].hit10 += (10.0 / avail as f64).min(1.0);
                acc[0].hit50 += (50.0 / avail as f64).min(1.0);
                acc[0].rr += h / avail as f64;

                let pop = |t: TrackId| listens.get(&t).copied().unwrap_or(0.0);
                acc[1].add(Some(rank_by(ids, &played, p.track, pop)));
                acc[2].add(Some(rank_by(ids, &played, p.track, taste_score)));
                let prev = s[i - 1].track;
                acc[3].add(Some(rank_by(ids, &played, p.track, |t| sim(prev, t))));
                let rank = steer.as_ref().and_then(|st| {
                    st.ranked(MAX_RANK)
                        .iter()
                        .position(|(t, _)| *t == p.track)
                        .map(|r| r as f64 + 1.0)
                });
                acc[4].add(rank);
            }
            played.insert(p.track);
            match &mut steer {
                None => {
                    let c = SessionConfig {
                        seed: 1,
                        ..cfg.steer.clone()
                    };
                    steer = Session::new(space, &[p.track], c);
                }
                Some(st) => {
                    st.observe(p.track);
                    let fb = if p.listened {
                        Feedback::Finished
                    } else {
                        let dur = durations.get(&p.track).copied().unwrap_or(0.0);
                        let at = if dur > 0.0 {
                            (p.ms_played as f64 / 1000.0 / dur) as f32
                        } else {
                            0.1
                        };
                        Feedback::Skipped { at }
                    };
                    st.feedback(p.track, fb);
                }
            }
        }
    }

    let steps = acc[0].n;
    if steps == 0 {
        return Ok(None);
    }
    let mut skip_auc = Vec::new();
    if let Some(a) = auc(&skip_taste) {
        skip_auc.push(("your taste".to_owned(), a));
    }
    if let Some(a) = auc(&skip_prev) {
        skip_auc.push(("previous track".to_owned(), a));
    }
    Ok(Some(ReplayReport {
        holdout_days: cfg.holdout_days,
        train_listens,
        test_plays: test.len(),
        test_sessions: sess.len(),
        steps,
        rows: names.iter().zip(&acc).map(|(n, a)| a.row(n)).collect(),
        skip_auc,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_basics() {
        assert_eq!(auc(&[(1.0, true), (0.0, false)]), Some(1.0));
        assert_eq!(auc(&[(0.0, true), (1.0, false)]), Some(0.0));
        assert_eq!(auc(&[(0.5, true), (0.5, false)]), Some(0.5));
        assert_eq!(auc(&[(0.5, true)]), None);
        let a = auc(&[(0.1, false), (0.4, true), (0.35, false), (0.8, true)]).unwrap();
        assert!((a - 1.0).abs() < 1e-12, "{a}");
    }

    #[test]
    fn ranks_with_ties_and_exclusions() {
        let ids: Vec<TrackId> = (1..=5).map(TrackId).collect();
        let skip: HashSet<TrackId> = [TrackId(5)].into();
        // Scores 1..5; target 3 has 4 above it, but 5 is excluded.
        assert_eq!(rank_by(&ids, &skip, TrackId(3), |t| t.0 as f32), 2.0);
        // All tied: expected middle of 4 candidates.
        assert_eq!(rank_by(&ids, &skip, TrackId(3), |_| 0.0), 2.5);
    }

    #[test]
    fn sessions_split_on_gaps() {
        let p = |ts, ms| MatchedPlay {
            ts,
            track: TrackId(1),
            ms_played: ms,
            listened: true,
        };
        // Second play starts 20 s after the first ends; third starts 2 h later.
        let s = sessions(
            &[p(1000, 200_000), p(1220, 200_000), p(9000, 200_000)],
            1800,
        );
        assert_eq!(s.iter().map(Vec::len).collect::<Vec<_>>(), vec![2, 1]);
    }
}
