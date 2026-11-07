//! Per-user taste profiles and the learned skip model (Milestone 1 step 6).
//!
//! A [`Profile`] is rebuilt from a user's listening data whenever it's needed: imported
//! history plus `dhunn play` sessions. At this scale (thousands of plays) that takes
//! milliseconds, so nothing is stored and it's always current.
//!
//! * **Taste:** every track gets a time-decayed weight (listens up, skips down, likes and
//!   dislikes strongly). Positively weighted tracks are clustered (spherical k-means) into a
//!   few centroids in the sound space: the listener's "regions".
//! * **Counts** per track and artist (listens / plays) feed the skip model.
//! * **Dislikes** with no later like are remembered, so sessions never play them again.

pub mod data;
pub mod group;
pub mod skip;

use std::collections::{HashMap, HashSet};

use dhunn_core::TrackId;
use dhunn_index::{SoundSpace, dot, normalize};

pub use data::{Obs, history_sessions, logged_sessions, shared_sessions};
pub use group::{Friend, beyond_me, friend_signal, meeting_point, overlap};
pub use skip::{FEATURES, Retainer, SkipModel};

#[derive(Debug, Clone)]
pub struct ProfileConfig {
    /// A play this many days old counts half as much as one today.
    pub half_life_days: f64,
    pub max_regions: usize,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            half_life_days: 90.0,
            max_regions: 6,
        }
    }
}

/// One region of a listener's taste.
#[derive(Debug, Clone)]
pub struct Region {
    /// Unit vector in the sound space.
    pub centroid: Vec<f32>,
    /// Share of the listener's positive weight (regions sum to 1).
    pub share: f32,
    /// The region's most played tracks, heaviest first (up to 5).
    pub top: Vec<TrackId>,
}

/// Listens and plays (undecayed) of one track or artist.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Counts {
    pub listens: f32,
    pub plays: f32,
}

#[derive(Debug, Clone, Default)]
pub struct Profile {
    pub regions: Vec<Region>,
    /// Share-weighted centre of the regions (unit vector).
    pub overall: Option<Vec<f32>>,
    /// Time-decayed weight per track (positive = liked overall).
    pub weight: HashMap<TrackId, f32>,
    pub disliked: HashSet<TrackId>,
    pub tracks: HashMap<TrackId, Counts>,
    pub artists: HashMap<i64, Counts>,
    pub artist_of: HashMap<TrackId, i64>,
    /// Listens / plays over everything.
    pub listen_rate: f32,
    pub observations: usize,
}

impl Profile {
    /// Builds a profile from sessions of observations (`now` in Unix seconds).
    pub fn build(
        space: &SoundSpace,
        sessions: &[Vec<Obs>],
        artist_of: &HashMap<TrackId, i64>,
        now: i64,
        cfg: &ProfileConfig,
    ) -> Self {
        let mut p = Profile {
            artist_of: artist_of.clone(),
            ..Default::default()
        };
        let mut last_dislike: HashMap<TrackId, i64> = HashMap::new();
        let mut last_like: HashMap<TrackId, i64> = HashMap::new();
        let (mut listens, mut plays) = (0f32, 0f32);
        for o in sessions
            .iter()
            .flatten()
            .filter(|o| space.contains(o.track))
        {
            let age_days = ((now - o.ts).max(0) as f64) / 86_400.0;
            let decay = 0.5f64.powf(age_days / cfg.half_life_days) as f32;
            let signal = if o.listened { 1.0 } else { -0.3 } + if o.liked { 2.0 } else { 0.0 }
                - if o.disliked { 3.0 } else { 0.0 };
            *p.weight.entry(o.track).or_default() += decay * signal;
            let l = f32::from(u8::from(o.listened));
            let c = p.tracks.entry(o.track).or_default();
            c.listens += l;
            c.plays += 1.0;
            if let Some(a) = artist_of.get(&o.track) {
                let c = p.artists.entry(*a).or_default();
                c.listens += l;
                c.plays += 1.0;
            }
            listens += l;
            plays += 1.0;
            if o.disliked {
                last_dislike.insert(o.track, o.ts);
            }
            if o.liked {
                last_like.insert(o.track, o.ts);
            }
        }
        p.listen_rate = if plays > 0.0 { listens / plays } else { 0.5 };
        p.observations = plays as usize;
        p.disliked = last_dislike
            .into_iter()
            .filter(|(t, ts)| last_like.get(t).is_none_or(|l| l < ts))
            .map(|(t, _)| t)
            .collect();
        p.regions = regions(space, &p.weight, cfg.max_regions);
        if let Some(first) = p.regions.first() {
            let mut c = vec![0f32; first.centroid.len()];
            for r in &p.regions {
                c.iter_mut()
                    .zip(&r.centroid)
                    .for_each(|(a, b)| *a += r.share * b);
            }
            normalize(&mut c);
            p.overall = Some(c);
        }
        p
    }

    /// Closeness of a unit vector to the nearest taste region (`None` without regions).
    pub fn taste_of(&self, v: &[f32]) -> Option<f32> {
        self.regions
            .iter()
            .map(|r| dot(&r.centroid, v))
            .reduce(f32::max)
    }

    /// Closeness to the listener's taste as a whole (the share-weighted centre of all
    /// regions). Worse than [`Profile::taste_of`] at finding what they'll play next, better
    /// at telling what they'll listen to from what they'll skip.
    pub fn overall_of(&self, v: &[f32]) -> Option<f32> {
        self.overall.as_deref().map(|c| dot(c, v))
    }

    pub fn centroids(&self) -> Vec<Vec<f32>> {
        self.regions.iter().map(|r| r.centroid.clone()).collect()
    }
}

/// Weighted spherical k-means over the positively weighted tracks.
fn regions(space: &SoundSpace, weight: &HashMap<TrackId, f32>, max: usize) -> Vec<Region> {
    let mut pts: Vec<(TrackId, f32, &[f32])> = weight
        .iter()
        .filter(|(_, w)| **w > 0.0)
        .filter_map(|(&t, &w)| space.vector(t).map(|v| (t, w, v)))
        .collect();
    if pts.is_empty() {
        return Vec::new();
    }
    pts.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let k = ((pts.len() as f64 / 4.0).sqrt().round() as usize).clamp(1, max.max(1));

    // Deterministic k-means++-style start: the heaviest track, then repeatedly the track
    // with the most weight far from every chosen centre.
    let mut centres: Vec<Vec<f32>> = vec![pts[0].2.to_vec()];
    while centres.len() < k {
        let next = pts
            .iter()
            .map(|(_, w, v)| {
                let near = centres.iter().map(|c| dot(c, v)).fold(f32::MIN, f32::max);
                w * (1.0 - near)
            })
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
            .expect("points exist");
        centres.push(pts[next].2.to_vec());
    }
    let dims = space.dims();
    let mut assign = vec![0usize; pts.len()];
    for _ in 0..25 {
        let mut changed = false;
        for (i, (_, _, v)) in pts.iter().enumerate() {
            let best = (0..centres.len())
                .max_by(|&a, &b| dot(&centres[a], v).total_cmp(&dot(&centres[b], v)))
                .expect("centres exist");
            changed |= assign[i] != best;
            assign[i] = best;
        }
        for (c, centre) in centres.iter_mut().enumerate() {
            let mut sum = vec![0f32; dims];
            for (i, (_, w, v)) in pts.iter().enumerate() {
                if assign[i] == c {
                    sum.iter_mut().zip(*v).for_each(|(s, x)| *s += w * x);
                }
            }
            if sum.iter().any(|x| *x != 0.0) {
                normalize(&mut sum);
                *centre = sum;
            }
        }
        if !changed {
            break;
        }
    }
    let total: f32 = pts.iter().map(|p| p.1).sum();
    let mut out: Vec<Region> = centres
        .into_iter()
        .enumerate()
        .filter_map(|(c, centroid)| {
            let members: Vec<&(TrackId, f32, &[f32])> = pts
                .iter()
                .zip(&assign)
                .filter(|(_, a)| **a == c)
                .map(|(p, _)| p)
                .collect();
            let w: f32 = members.iter().map(|p| p.1).sum();
            (w > 0.0).then(|| Region {
                centroid,
                share: w / total,
                // `pts` is sorted heaviest first.
                top: members.iter().take(5).map(|p| p.0).collect(),
            })
        })
        .collect();
    out.sort_by(|a, b| b.share.total_cmp(&a.share));
    out
}

/// Area under the ROC curve: probability a random positive outscores a random negative
/// (ties count half). `None` without both classes.
pub fn auc(scored: &[(f32, bool)]) -> Option<f64> {
    let mut v = scored.to_vec();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let pos = v.iter().filter(|x| x.1).count();
    let neg = v.len() - pos;
    if pos == 0 || neg == 0 {
        return None;
    }
    let mut rank_sum = 0.0;
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j < v.len() && v[j].0 == v[i].0 {
            j += 1;
        }
        let avg = (i + j + 1) as f64 / 2.0;
        rank_sum += avg * v[i..j].iter().filter(|x| x.1).count() as f64;
        i = j;
    }
    Some((rank_sum - (pos * (pos + 1)) as f64 / 2.0) / (pos * neg) as f64)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use dhunn_index::AspectWeights;

    /// 60 tracks on a circle: neighbours by id sound alike.
    pub fn ring() -> SoundSpace {
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

    pub fn obs(t: u64, listened: bool) -> Obs {
        Obs {
            track: TrackId(t),
            ts: 0,
            listened,
            liked: false,
            disliked: false,
        }
    }

    #[test]
    fn finds_two_taste_regions_and_remembers_dislikes() {
        let s = ring();
        // Listens around tracks 5 and 35 (opposite sides), skips elsewhere.
        let mut sess = Vec::new();
        for _ in 0..3 {
            for t in [3, 4, 5, 6, 7, 33, 34, 35, 36, 37] {
                sess.push(obs(t, true));
            }
            sess.push(obs(20, false));
        }
        sess.push(Obs {
            disliked: true,
            ..obs(50, false)
        });
        let p = Profile::build(&s, &[sess], &HashMap::new(), 0, &Default::default());
        assert_eq!(p.regions.len(), 2, "{:?}", p.regions);
        let v = |t| s.vector(TrackId(t)).unwrap();
        assert!(p.taste_of(v(5)).unwrap() > 0.99);
        assert!(p.taste_of(v(35)).unwrap() > 0.99);
        assert!(p.taste_of(v(20)).unwrap() < 0.1);
        assert!(p.disliked.contains(&TrackId(50)));
        assert!(p.weight[&TrackId(20)] < 0.0);
        assert!((p.regions[0].share - 0.5).abs() < 0.05);
        assert_eq!(p.tracks[&TrackId(5)].listens, 3.0);
    }

    #[test]
    fn old_plays_count_less() {
        let s = ring();
        let old = Obs {
            ts: 0,
            ..obs(5, true)
        };
        let new = Obs {
            ts: 180 * 86_400,
            ..obs(35, true)
        };
        let p = Profile::build(
            &s,
            &[vec![old, new]],
            &HashMap::new(),
            180 * 86_400,
            &Default::default(),
        );
        assert!((p.weight[&TrackId(5)] - 0.25).abs() < 1e-3);
        assert!((p.weight[&TrackId(35)] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn a_later_like_forgives_a_dislike() {
        let s = ring();
        let d = Obs {
            ts: 10,
            disliked: true,
            ..obs(5, false)
        };
        let l = Obs {
            ts: 20,
            liked: true,
            ..obs(5, true)
        };
        let p = Profile::build(&s, &[vec![d, l]], &HashMap::new(), 30, &Default::default());
        assert!(p.disliked.is_empty());
    }

    #[test]
    fn auc_basics() {
        assert_eq!(auc(&[(1.0, true), (0.0, false)]), Some(1.0));
        assert_eq!(auc(&[(0.5, true), (0.5, false)]), Some(0.5));
        assert_eq!(auc(&[(0.5, true)]), None);
    }
}
