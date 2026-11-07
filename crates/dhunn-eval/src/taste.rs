//! A simulated listener's hidden taste: a few "home" tracks in z-scored DSP space, and each
//! library track's affinity = cosine similarity to the nearest home.
//!
//! The taste lives in plain z-scored DSP features, *not* in the recommender's weighted sound
//! space, so a recommender is not graded by its own yardstick (the two are still related:
//! both come from the same features).

use std::collections::HashMap;

use dhunn_core::TrackId;
use dhunn_dsp::DSP_VERSION;
use dhunn_store::Store;

use crate::rng::Rng;
use crate::simulate::SimError;

pub struct Taste {
    pub ids: Vec<TrackId>,
    /// Indices into `ids` of the home tracks.
    pub homes: Vec<usize>,
    /// Affinity (−1..1) per track, aligned with `ids`.
    pub affinity: Vec<f64>,
    /// Share of the library this listener likes less than each track (0..1), aligned with
    /// `ids`: 1 ≈ a favourite, 0.5 = middling, 0 = the track they'd like least.
    pub percentile: Vec<f64>,
    vecs: Vec<Vec<f32>>,
    index: HashMap<TrackId, usize>,
}

fn zscore(rows: &mut [Vec<f32>]) {
    let Some(dims) = rows.first().map(Vec::len) else {
        return;
    };
    for d in 0..dims {
        let n = rows.len() as f64;
        let mean = rows.iter().map(|r| r[d] as f64).sum::<f64>() / n;
        let sd = (rows
            .iter()
            .map(|r| (r[d] as f64 - mean).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        let sd = if sd > 1e-9 { sd } else { 1.0 };
        for r in rows.iter_mut() {
            r[d] = ((r[d] as f64 - mean) / sd) as f32;
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        ab += x as f64 * y as f64;
        aa += x as f64 * x as f64;
        bb += y as f64 * y as f64;
    }
    if aa == 0.0 || bb == 0.0 {
        0.0
    } else {
        ab / (aa.sqrt() * bb.sqrt())
    }
}

/// How likely a listener is to skip a track they have this affinity for.
pub fn skip_probability(affinity: f64) -> f64 {
    (1.0 / (1.0 + ((affinity - 0.15) * 8.0).exp())).clamp(0.03, 0.9)
}

impl Taste {
    /// Draws `regions` home tracks at random from the library's analyzed tracks.
    pub fn hidden(store: &Store, rng: &mut Rng, regions: usize) -> Result<Self, SimError> {
        let (_, rows) = store.feature_matrix("dsp", DSP_VERSION)?;
        if rows.len() < 10 {
            return Err(SimError::TooFewTracks(rows.len()));
        }
        let ids: Vec<TrackId> = rows.iter().map(|r| r.0).collect();
        let mut vecs: Vec<Vec<f32>> = rows.into_iter().map(|r| r.1).collect();
        zscore(&mut vecs);
        let n = ids.len();
        let mut homes: Vec<usize> = Vec::new();
        while homes.len() < regions.clamp(1, n) {
            let i = rng.int(0, n as i64 - 1) as usize;
            if !homes.contains(&i) {
                homes.push(i);
            }
        }
        let affinity: Vec<f64> = (0..n)
            .map(|i| {
                homes
                    .iter()
                    .map(|&h| cosine(&vecs[i], &vecs[h]))
                    .fold(f64::MIN, f64::max)
            })
            .collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| affinity[a].total_cmp(&affinity[b]));
        let mut percentile = vec![0.0; n];
        for (rank, &i) in order.iter().enumerate() {
            percentile[i] = rank as f64 / (n - 1).max(1) as f64;
        }
        let index = ids.iter().enumerate().map(|(i, &t)| (t, i)).collect();
        Ok(Self {
            ids,
            homes,
            affinity,
            percentile,
            vecs,
            index,
        })
    }

    pub fn cosine(&self, a: usize, b: usize) -> f64 {
        cosine(&self.vecs[a], &self.vecs[b])
    }

    pub fn index_of(&self, t: TrackId) -> Option<usize> {
        self.index.get(&t).copied()
    }

    /// The listener's favourite tracks: the top `share` of the library by affinity.
    pub fn favourites(&self, share: f64) -> Vec<TrackId> {
        let mut v: Vec<(TrackId, f64)> = self
            .ids
            .iter()
            .zip(&self.percentile)
            .filter(|(_, p)| **p >= 1.0 - share)
            .map(|(&t, &p)| (t, p))
            .collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        v.into_iter().map(|(t, _)| t).collect()
    }
}
