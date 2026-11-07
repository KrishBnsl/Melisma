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
/// The name CLAP embeddings are stored under (see `dhunn-embed`).
pub const CLAP_MODEL: &str = "clap";
/// Default share of "sounds like" that comes from CLAP when embeddings exist. Chosen with
/// `dhunn eval --clap-share` under listeners whose taste is defined in DSP features and in
/// CLAP embeddings: 0.25 has the best worst case of the two (see ARCHITECTURE.md §11).
pub const DEFAULT_CLAP_SHARE: f32 = 0.25;

pub struct SoundSpace {
    ids: Vec<TrackId>,
    pos: HashMap<TrackId, usize>,
    dims: usize,
    /// Weighted, L2-normalized rows (`ids.len() × dims`).
    unit: Vec<f32>,
    /// Z-scored DSP rows before weighting, for per-aspect comparisons (`zdims` wide).
    z: Vec<f32>,
    zdims: usize,
    /// Raw (unit) CLAP embeddings, for matching text — they share CLAP's text space.
    clap: HashMap<TrackId, Vec<f32>>,
    /// Share of similarity that comes from CLAP (0 = DSP only).
    clap_share: f32,
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
            zdims: dims,
            clap: HashMap::new(),
            clap_share: 0.0,
            aspects,
            typical,
        }
    }

    /// Mixes learned embeddings (CLAP) into the space: each track's vector becomes
    /// `[√(1−share)·dsp, √share·clap]`, so similarity = (1−share)·DSP + share·CLAP. The
    /// embeddings are centred on the library mean first (CLAP vectors share a large common
    /// component, which would make every pair look alike). Tracks without an embedding keep
    /// only their DSP part. Skipped (DSP only) if fewer than half the tracks have one.
    pub fn with_embeddings(mut self, emb: HashMap<TrackId, Vec<f32>>, share: f32) -> Self {
        let share = share.clamp(0.0, 1.0);
        let covered = self.ids.iter().filter(|t| emb.contains_key(t)).count();
        let Some(edims) = emb.values().next().map(Vec::len) else {
            return self;
        };
        if covered * 2 < self.ids.len() || edims == 0 {
            return self;
        }
        let mut mean = vec![0f64; edims];
        for t in &self.ids {
            if let Some(e) = emb.get(t) {
                mean.iter_mut().zip(e).for_each(|(m, x)| *m += *x as f64);
            }
        }
        mean.iter_mut().for_each(|m| *m /= covered as f64);
        let (a, b) = ((1.0 - share).sqrt(), share.sqrt());
        let old = self.dims;
        let dims = old + edims;
        let mut unit = vec![0f32; self.ids.len() * dims];
        for (i, t) in self.ids.iter().enumerate() {
            let row = &mut unit[i * dims..(i + 1) * dims];
            row[..old]
                .iter_mut()
                .zip(&self.unit[i * old..(i + 1) * old])
                .for_each(|(r, x)| *r = a * x);
            if let Some(e) = emb.get(t) {
                let mut c: Vec<f32> = e.iter().zip(&mean).map(|(x, m)| x - *m as f32).collect();
                normalize(&mut c);
                row[old..].iter_mut().zip(&c).for_each(|(r, x)| *r = b * x);
            }
            normalize(row);
        }
        self.unit = unit;
        self.dims = dims;
        self.clap_share = share;
        self.clap = emb
            .into_iter()
            .filter(|(t, _)| self.pos.contains_key(t))
            .map(|(t, mut e)| {
                normalize(&mut e);
                (t, e)
            })
            .collect();
        self
    }

    /// Share of similarity that comes from learned embeddings (0 = DSP only).
    pub fn clap_share(&self) -> f32 {
        self.clap_share
    }

    pub fn has_clap(&self) -> bool {
        !self.clap.is_empty()
    }

    /// The tracks that best match a CLAP text (or audio) embedding, best first, by raw
    /// CLAP similarity — independent of how much CLAP counts in "sounds like".
    pub fn clap_matches(
        &self,
        query: &[f32],
        k: usize,
        keep: impl Fn(TrackId) -> bool,
    ) -> Vec<(TrackId, f32)> {
        let mut v: Vec<(TrackId, f32)> = self
            .clap
            .iter()
            .filter(|(t, _)| keep(**t))
            .map(|(&t, e)| (t, dot(e, query)))
            .collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(k);
        v
    }

    /// Builds the space from the store's current DSP features, without learned embeddings.
    pub fn from_store(store: &Store, weights: &AspectWeights) -> Result<Self, StoreError> {
        let (names, rows) = store.feature_matrix("dsp", dhunn_dsp::DSP_VERSION)?;
        Ok(Self::build(&names, rows, weights))
    }

    /// DSP features plus the store's CLAP embeddings (if any) at `clap_share`.
    pub fn from_store_blended(
        store: &Store,
        weights: &AspectWeights,
        clap_share: f32,
    ) -> Result<Self, StoreError> {
        let space = Self::from_store(store, weights)?;
        let emb = store.latest_embeddings(CLAP_MODEL)?;
        Ok(space.with_embeddings(emb, clap_share))
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
            &self.z[i * self.zdims..(i + 1) * self.zdims],
            &self.z[j * self.zdims..(j + 1) * self.zdims],
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

    /// 40 tracks on a circle in a 2-D DSP space.
    fn circle() -> SoundSpace {
        let names: Vec<String> = ["mfcc00_mean", "log2_tempo"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let rows = (0..40)
            .map(|i| {
                let a = i as f32 / 40.0 * std::f32::consts::TAU;
                (TrackId(i + 1), vec![a.cos(), a.sin()])
            })
            .collect();
        SoundSpace::build(&names, rows, &AspectWeights::default())
    }

    #[test]
    fn embeddings_blend_into_similarity() {
        // CLAP says tracks 1 and 21 (opposite on the DSP circle) sound alike; others differ.
        let emb = |t: u64| -> Vec<f32> {
            let mut v = vec![0.1f32; 8];
            let slot = if t == 1 || t == 21 {
                0
            } else {
                (t % 7 + 1) as usize
            };
            v[slot] = 1.0;
            v
        };
        let all: HashMap<TrackId, Vec<f32>> = (1..=40).map(|t| (TrackId(t), emb(t))).collect();
        let dsp = circle();
        let before = dsp.similarity(TrackId(1), TrackId(21));
        assert!(before < -0.9);

        let half = circle().with_embeddings(all.clone(), 0.5);
        let mid = half.similarity(TrackId(1), TrackId(21));
        assert!(mid > before + 0.5, "{before} → {mid}");
        assert!((half.clap_share() - 0.5).abs() < 1e-6);
        // Every vector is still a unit vector.
        let v = half.vector(TrackId(5)).unwrap();
        assert!((dot(v, v) - 1.0).abs() < 1e-5);

        let only = circle().with_embeddings(all.clone(), 1.0);
        assert!(only.similarity(TrackId(1), TrackId(21)) > 0.99);
        assert!(
            circle()
                .with_embeddings(all.clone(), 0.0)
                .similarity(TrackId(1), TrackId(21))
                < -0.9
        );

        // Text matching uses the raw embeddings.
        let q = emb(1);
        let m = half.clap_matches(&q, 2, |_| true);
        let ids: Vec<u64> = m.iter().map(|x| x.0.0).collect();
        assert!(ids.contains(&1) && ids.contains(&21), "{ids:?}");

        // Too few tracks with embeddings: DSP only.
        let few: HashMap<TrackId, Vec<f32>> = (1..=10).map(|t| (TrackId(t), emb(t))).collect();
        let s = circle().with_embeddings(few, 0.5);
        assert!(!s.has_clap());
        assert_eq!(s.similarity(TrackId(1), TrackId(21)), before);
        // Aspect explanations still work on the blended space.
        assert!(!half.aspect_similarities(TrackId(1), TrackId(2)).is_empty());
    }
}
