//! The sound space: every analyzed track as a point, so "sounds like" becomes "is near".
//!
//! Built from the DSP feature vectors:
//! 1. each dimension is z-scored across the library (and clipped to ±4 so a few extreme
//!    tracks can't stretch the space);
//! 2. dimensions are grouped into perceptual **aspects** (timbre, brightness, harmony,
//!    rhythm, punch, dynamics, pitch); each aspect is scaled so that it contributes in
//!    proportion to its weight, however many dimensions it has (timbre has 80, rhythm 4);
//! 3. each track's vector is L2-normalized, so a dot product is a cosine similarity.
//!
//! Search is exact (brute force, parallel): at ≤ 500k tracks × ~134 dims that's a few
//! milliseconds, and exact results are easier to reason about while tuning.

pub mod places;

use std::collections::HashMap;

use dhunn_core::TrackId;
use dhunn_store::{Store, StoreError};
use rayon::prelude::*;

/// Perceptual groups of DSP dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Aspect {
    Timbre,
    Brightness,
    Harmony,
    Rhythm,
    Punch,
    Dynamics,
    Pitch,
}

impl Aspect {
    pub const ALL: [Aspect; 7] = [
        Aspect::Timbre,
        Aspect::Brightness,
        Aspect::Harmony,
        Aspect::Rhythm,
        Aspect::Punch,
        Aspect::Dynamics,
        Aspect::Pitch,
    ];

    /// Words for explanations ("similar timbre and rhythm").
    pub fn label(self) -> &'static str {
        match self {
            Aspect::Timbre => "timbre",
            Aspect::Brightness => "brightness",
            Aspect::Harmony => "harmony",
            Aspect::Rhythm => "rhythm",
            Aspect::Punch => "low-end punch",
            Aspect::Dynamics => "dynamics",
            Aspect::Pitch => "melodic range",
        }
    }

    /// Which aspect a DSP feature belongs to, by its name (see `DspSummary::named_values`).
    pub fn of_feature(name: &str) -> Aspect {
        let starts = |p: &str| name.starts_with(p);
        if starts("mfcc") {
            Aspect::Timbre
        } else if starts("chroma") || starts("mode_") || starts("key_") {
            Aspect::Harmony
        } else if starts("log2_tempo") || starts("pulse") || starts("onset") {
            Aspect::Rhythm
        } else if starts("thump") || starts("harmonic_ratio") {
            Aspect::Punch
        } else if starts("rms") || starts("dynamic") {
            Aspect::Dynamics
        } else if starts("f0") || starts("voiced") || starts("periodicity") {
            Aspect::Pitch
        } else {
            Aspect::Brightness // centroid, bandwidth, rolloff, flatness, zcr, contrast
        }
    }
}

/// Relative importance of each aspect in "sounds like".
#[derive(Debug, Clone, PartialEq)]
pub struct AspectWeights(pub HashMap<Aspect, f32>);

impl Default for AspectWeights {
    fn default() -> Self {
        Self(HashMap::from([
            (Aspect::Timbre, 1.0),
            (Aspect::Brightness, 1.0),
            (Aspect::Harmony, 0.7),
            (Aspect::Rhythm, 1.0),
            (Aspect::Punch, 0.8),
            (Aspect::Dynamics, 0.5),
            (Aspect::Pitch, 0.5),
        ]))
    }
}

impl AspectWeights {
    pub fn get(&self, a: Aspect) -> f32 {
        self.0.get(&a).copied().unwrap_or(1.0)
    }
}

const Z_CLIP: f32 = 4.0;

pub struct SoundSpace {
    ids: Vec<TrackId>,
    pos: HashMap<TrackId, usize>,
    dims: usize,
    /// Weighted, L2-normalized rows (`ids.len() × dims`).
    unit: Vec<f32>,
    /// Z-scored rows before weighting, for per-aspect comparisons.
    z: Vec<f32>,
    /// Dimension indices of each aspect present.
    aspects: Vec<(Aspect, Vec<usize>)>,
    /// Typical (mean) z-space distance between two tracks on each aspect, for judging
    /// whether a pair is *unusually* close on it.
    typical: Vec<f32>,
}

fn aspect_distance(ra: &[f32], rb: &[f32], dims: &[usize]) -> f32 {
    dims.iter()
        .map(|&d| (ra[d] - rb[d]).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Normalizes in place; leaves an all-zero vector as is.
pub fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl SoundSpace {
    /// Builds the space from feature rows sharing one layout (`names`).
    pub fn build(
        names: &[String],
        rows: Vec<(TrackId, Vec<f32>)>,
        weights: &AspectWeights,
    ) -> Self {
        let dims = names.len();
        let n = rows.len();
        let mut aspects: Vec<(Aspect, Vec<usize>)> =
            Aspect::ALL.iter().map(|&a| (a, Vec::new())).collect();
        for (d, name) in names.iter().enumerate() {
            let a = Aspect::of_feature(name);
            aspects
                .iter_mut()
                .find(|(x, _)| *x == a)
                .expect("all aspects listed")
                .1
                .push(d);
        }
        aspects.retain(|(_, d)| !d.is_empty());

        let ids: Vec<TrackId> = rows.iter().map(|r| r.0).collect();
        let mut z: Vec<f32> = Vec::with_capacity(n * dims);
        for (_, v) in &rows {
            assert_eq!(
                v.len(),
                dims,
                "feature vector length differs from its schema"
            );
            z.extend_from_slice(v);
        }
        drop(rows);
        for d in 0..dims {
            let mean = (0..n).map(|i| z[i * dims + d] as f64).sum::<f64>() / n.max(1) as f64;
            let var = (0..n)
                .map(|i| (z[i * dims + d] as f64 - mean).powi(2))
                .sum::<f64>()
                / n.max(1) as f64;
            let sd = if var.sqrt() > 1e-9 { var.sqrt() } else { 1.0 };
            for i in 0..n {
                let v = ((z[i * dims + d] as f64 - mean) / sd) as f32;
                z[i * dims + d] = v.clamp(-Z_CLIP, Z_CLIP);
            }
        }

        let mut scale = vec![0f32; dims];
        for (a, ds) in &aspects {
            let s = weights.get(*a) / (ds.len() as f32).sqrt();
            ds.iter().for_each(|&d| scale[d] = s);
        }
        let mut unit = z.clone();
        unit.par_chunks_mut(dims.max(1)).for_each(|row| {
            row.iter_mut().zip(&scale).for_each(|(x, s)| *x *= s);
            normalize(row);
        });

        // Typical per-aspect distance from a deterministic sample of pairs.
        let mut typical = vec![1f32; aspects.len()];
        if n >= 2 {
            let mut state = 0x2545_F491_4F6C_DD1Du64;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % n as u64) as usize
            };
            let pairs: Vec<(usize, usize)> = (0..4000)
                .map(|_| (next(), next()))
                .filter(|(a, b)| a != b)
                .collect();
            for (k, (_, ds)) in aspects.iter().enumerate() {
                let total: f32 = pairs
                    .iter()
                    .map(|&(a, b)| {
                        aspect_distance(
                            &z[a * dims..(a + 1) * dims],
                            &z[b * dims..(b + 1) * dims],
                            ds,
                        )
                    })
                    .sum();
                let mean = total / pairs.len().max(1) as f32;
                typical[k] = if mean > 1e-6 { mean } else { 1.0 };
            }
        }

        let pos = ids.iter().enumerate().map(|(i, &t)| (t, i)).collect();
        Self {
            ids,
            pos,
            dims,
            unit,
            z,
            aspects,
            typical,
        }
    }

    /// Builds the space from the store's current DSP features.
    pub fn from_store(store: &Store, weights: &AspectWeights) -> Result<Self, StoreError> {
        let (names, rows) = store.feature_matrix("dsp", dhunn_dsp::DSP_VERSION)?;
        Ok(Self::build(&names, rows, weights))
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn ids(&self) -> &[TrackId] {
        &self.ids
    }

    pub fn contains(&self, t: TrackId) -> bool {
        self.pos.contains_key(&t)
    }

    /// The track's unit vector.
    pub fn vector(&self, t: TrackId) -> Option<&[f32]> {
        self.pos
            .get(&t)
            .map(|&i| &self.unit[i * self.dims..(i + 1) * self.dims])
    }

    /// Cosine similarity of two tracks (−1..1), 0 if either is unknown.
    pub fn similarity(&self, a: TrackId, b: TrackId) -> f32 {
        match (self.vector(a), self.vector(b)) {
            (Some(x), Some(y)) => dot(x, y),
            _ => 0.0,
        }
    }

    /// Normalized weighted mean of track vectors (e.g. the centre of a few seeds).
    pub fn centroid(&self, tracks: &[(TrackId, f32)]) -> Option<Vec<f32>> {
        let mut c = vec![0f32; self.dims];
        let mut any = false;
        for &(t, w) in tracks {
            if let Some(v) = self.vector(t) {
                c.iter_mut().zip(v).for_each(|(a, b)| *a += w * b);
                any = true;
            }
        }
        any.then(|| {
            normalize(&mut c);
            c
        })
    }

    /// The `k` tracks most similar to `query` (a unit vector) that pass `keep`, best first.
    pub fn nearest(
        &self,
        query: &[f32],
        k: usize,
        keep: impl Fn(TrackId) -> bool + Sync,
    ) -> Vec<(TrackId, f32)> {
        let mut scored: Vec<(TrackId, f32)> = self
            .unit
            .par_chunks(self.dims.max(1))
            .zip(self.ids.par_iter())
            .filter(|(_, t)| keep(**t))
            .map(|(row, &t)| (t, dot(row, query)))
            .collect();
        let k = k.min(scored.len());
        if k == 0 {
            return Vec::new();
        }
        if k < scored.len() {
            scored.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
            scored.truncate(k);
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored
    }

    /// Per-aspect *closeness* of two tracks, most close first: `1 − distance / typical
    /// distance` on that aspect, so 1 = identical, 0 = as different as two random tracks in
    /// the library, negative = more different than usual. This is what explanations cite.
    pub fn aspect_similarities(&self, a: TrackId, b: TrackId) -> Vec<(Aspect, f32)> {
        let (Some(&i), Some(&j)) = (self.pos.get(&a), self.pos.get(&b)) else {
            return Vec::new();
        };
        let (ra, rb) = (
            &self.z[i * self.dims..(i + 1) * self.dims],
            &self.z[j * self.dims..(j + 1) * self.dims],
        );
        let mut out: Vec<(Aspect, f32)> = self
            .aspects
            .iter()
            .zip(&self.typical)
            .map(|((asp, ds), typ)| (*asp, 1.0 - aspect_distance(ra, rb, ds) / typ))
            .collect();
        out.sort_by(|p, q| q.1.total_cmp(&p.1));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> {
        // 4 timbre dims and 1 rhythm dim: aspects must still weigh equally.
        [
            "mfcc00_mean",
            "mfcc01_mean",
            "mfcc02_mean",
            "mfcc03_mean",
            "log2_tempo",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    fn space(rows: Vec<Vec<f32>>) -> SoundSpace {
        let rows = rows
            .into_iter()
            .enumerate()
            .map(|(i, v)| (TrackId(i as u64 + 1), v))
            .collect();
        let mut w = AspectWeights::default();
        w.0.insert(Aspect::Timbre, 1.0);
        w.0.insert(Aspect::Rhythm, 1.0);
        SoundSpace::build(&names(), rows, &w)
    }

    #[test]
    fn aspects_from_feature_names() {
        assert_eq!(Aspect::of_feature("mfcc_delta03_std"), Aspect::Timbre);
        assert_eq!(Aspect::of_feature("contrast2_db_mean"), Aspect::Brightness);
        assert_eq!(Aspect::of_feature("chroma_rel05"), Aspect::Harmony);
        assert_eq!(Aspect::of_feature("thump_flux_p90"), Aspect::Punch);
        assert_eq!(Aspect::of_feature("f0_midi_median"), Aspect::Pitch);
        assert_eq!(Aspect::of_feature("log2_tempo"), Aspect::Rhythm);
    }

    #[test]
    fn nearest_finds_the_neighbourhood() {
        let s = space(vec![
            vec![1.0, 1.0, 1.0, 1.0, 7.0],
            vec![1.1, 0.9, 1.0, 1.0, 7.0],
            vec![-1.0, -1.0, -1.0, -1.0, 6.0],
            vec![-1.1, -0.9, -1.0, -1.0, 6.0],
        ]);
        let q = s.vector(TrackId(1)).unwrap().to_vec();
        let near = s.nearest(&q, 2, |_| true);
        assert_eq!(
            near.iter().map(|x| x.0).collect::<Vec<_>>(),
            vec![TrackId(1), TrackId(2)]
        );
        assert!((near[0].1 - 1.0).abs() < 1e-5);
        let excl = s.nearest(&q, 1, |t| t != TrackId(1));
        assert_eq!(excl[0].0, TrackId(2));
        assert!(s.similarity(TrackId(1), TrackId(3)) < 0.0);
    }

    #[test]
    fn aspects_weigh_equally_regardless_of_dimension_count() {
        // Track 2 differs from 1 only in timbre (4 dims), track 3 only in rhythm (1 dim).
        let s = space(vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            vec![-1.0, -1.0, -1.0, -1.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, -1.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
        ]);
        let timbre_off = s.similarity(TrackId(1), TrackId(2));
        let rhythm_off = s.similarity(TrackId(1), TrackId(3));
        assert!(
            (timbre_off - rhythm_off).abs() < 0.05,
            "{timbre_off} vs {rhythm_off}"
        );

        let a = s.aspect_similarities(TrackId(1), TrackId(3));
        assert_eq!(a[0].0, Aspect::Timbre);
        assert!(a[0].1 > 0.99);
    }

    #[test]
    fn centroid_and_empty() {
        let s = space(vec![
            vec![1.0, 0.0, 0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0, 0.0, 2.0],
            vec![0.0; 5],
        ]);
        let c = s.centroid(&[(TrackId(1), 1.0), (TrackId(2), 1.0)]).unwrap();
        assert!((dot(&c, &c) - 1.0).abs() < 1e-5);
        assert!(s.centroid(&[(TrackId(99), 1.0)]).is_none());
        assert!(s.nearest(&c, 0, |_| true).is_empty());
    }
}
