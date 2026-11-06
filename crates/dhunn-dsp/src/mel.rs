//! Mel filterbank, dB conversion, MFCC and delta features (librosa-compatible).

use rayon::prelude::*;

use crate::stft::Matrix;

const F_SP: f64 = 200.0 / 3.0;
const MIN_LOG_HZ: f64 = 1000.0;
const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;

fn logstep() -> f64 {
    6.4f64.ln() / 27.0
}

/// Slaney-style mel scale (librosa default, `htk=False`).
pub fn hz_to_mel(f: f64) -> f64 {
    if f >= MIN_LOG_HZ {
        MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() / logstep()
    } else {
        f / F_SP
    }
}

pub fn mel_to_hz(m: f64) -> f64 {
    if m >= MIN_LOG_MEL {
        MIN_LOG_HZ * (logstep() * (m - MIN_LOG_MEL)).exp()
    } else {
        F_SP * m
    }
}

/// A filterbank stored sparsely: each filter is a contiguous run of non-zero weights.
#[derive(Debug, Clone)]
pub struct SparseFilterbank {
    filters: Vec<(usize, Vec<f32>)>,
}

impl SparseFilterbank {
    /// Builds from dense `filters × bins` weights, keeping only the non-zero span of each filter.
    pub fn from_dense(dense: &[Vec<f64>]) -> Self {
        let filters = dense
            .iter()
            .map(|w| {
                let first = w.iter().position(|&v| v != 0.0);
                let last = w.iter().rposition(|&v| v != 0.0);
                match (first, last) {
                    (Some(a), Some(b)) => (a, w[a..=b].iter().map(|&v| v as f32).collect()),
                    _ => (0, Vec::new()),
                }
            })
            .collect();
        Self { filters }
    }

    pub fn n_filters(&self) -> usize {
        self.filters.len()
    }

    /// `out[t][m] = Σ_f W[m][f] · spec[t][f]`.
    pub fn apply(&self, spec: &Matrix) -> Matrix {
        let n = self.filters.len();
        let mut out = Matrix::zeros(spec.rows(), n);
        out.data_mut()
            .par_chunks_mut(n.max(1))
            .enumerate()
            .for_each(|(t, row)| {
                let s = spec.row(t);
                for (o, (start, w)) in row.iter_mut().zip(&self.filters) {
                    *o = w
                        .iter()
                        .zip(&s[*start..])
                        .map(|(&a, &b)| a as f64 * b as f64)
                        .sum::<f64>() as f32;
                }
            });
        out
    }
}

/// `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)` with Slaney normalization.
pub fn mel_filterbank(
    sr: f64,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
) -> SparseFilterbank {
    let n_bins = n_fft / 2 + 1;
    let fftfreqs: Vec<f64> = (0..n_bins).map(|k| k as f64 * sr / n_fft as f64).collect();
    let (mmin, mmax) = (hz_to_mel(fmin), hz_to_mel(fmax));
    let mel_f: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mmin + (mmax - mmin) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let dense: Vec<Vec<f64>> = (0..n_mels)
        .map(|i| {
            let (lo, mid, hi) = (mel_f[i], mel_f[i + 1], mel_f[i + 2]);
            let enorm = 2.0 / (hi - lo);
            fftfreqs
                .iter()
                .map(|&f| {
                    let lower = (f - lo) / (mid - lo);
                    let upper = (hi - f) / (hi - mid);
                    lower.min(upper).max(0.0) * enorm
                })
                .collect()
        })
        .collect();
    SparseFilterbank::from_dense(&dense)
}

/// `librosa.power_to_db(x, ref=1.0, amin=1e-10, top_db)` applied over the whole slice.
pub fn power_to_db_inplace(x: &mut [f32], top_db: Option<f32>) {
    const AMIN: f32 = 1e-10;
    let mut max = f32::NEG_INFINITY;
    for v in x.iter_mut() {
        *v = 10.0 * v.max(AMIN).log10();
        max = max.max(*v);
    }
    if let Some(top) = top_db {
        let floor = max - top;
        for v in x.iter_mut() {
            if *v < floor {
                *v = floor;
            }
        }
    }
}

/// Orthonormal DCT-II matrix, `n_out × n_in` (`scipy.fft.dct(type=2, norm="ortho")`).
pub fn dct2_ortho(n_out: usize, n_in: usize) -> Vec<Vec<f32>> {
    let n = n_in as f64;
    (0..n_out)
        .map(|k| {
            let scale = if k == 0 {
                (1.0 / n).sqrt()
            } else {
                (2.0 / n).sqrt()
            };
            (0..n_in)
                .map(|i| {
                    (scale
                        * (std::f64::consts::PI * k as f64 * (2 * i + 1) as f64 / (2.0 * n)).cos())
                        as f32
                })
                .collect()
        })
        .collect()
}

/// MFCCs from a log-mel spectrogram (`frames × mels`) → `frames × n_mfcc`.
pub fn mfcc(log_mel: &Matrix, n_mfcc: usize) -> Matrix {
    let dct = dct2_ortho(n_mfcc, log_mel.cols());
    let mut out = Matrix::zeros(log_mel.rows(), n_mfcc);
    out.data_mut()
        .par_chunks_mut(n_mfcc)
        .enumerate()
        .for_each(|(t, row)| {
            let l = log_mel.row(t);
            for (o, basis) in row.iter_mut().zip(&dct) {
                *o = basis
                    .iter()
                    .zip(l)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum::<f64>() as f32;
            }
        });
    out
}

/// First-order delta with a 9-frame window, matching `librosa.feature.delta(x)`
/// (`savgol_filter(polyorder=1, deriv=1, mode="interp")`). Returns zeros if shorter than 9.
pub fn delta(x: &[f32]) -> Vec<f32> {
    const HALF: usize = 4;
    const WIDTH: usize = 2 * HALF + 1;
    const NORM: f64 = 60.0; // Σ k² for k in -4..=4
    let n = x.len();
    if n < WIDTH {
        return vec![0.0; n];
    }
    let slope = |start: usize| -> f32 {
        (0..WIDTH)
            .map(|k| (k as f64 - HALF as f64) * x[start + k] as f64)
            .sum::<f64>() as f32
            / NORM as f32
    };
    let mut out = vec![0f32; n];
    for (t, o) in out.iter_mut().enumerate().take(n - HALF).skip(HALF) {
        *o = slope(t - HALF);
    }
    let first = slope(0);
    let last = slope(n - WIDTH);
    out[..HALF].fill(first);
    out[n - HALF..].fill(last);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_scale_roundtrip() {
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-9);
        for f in [0.0, 200.0, 999.0, 1000.0, 4000.0, 11025.0] {
            assert!((mel_to_hz(hz_to_mel(f)) - f).abs() < 1e-6);
        }
    }

    #[test]
    fn filterbank_shape() {
        let fb = mel_filterbank(22050.0, 2048, 128, 0.0, 11025.0);
        assert_eq!(fb.n_filters(), 128);
        assert!(fb.filters.iter().all(|(_, w)| !w.is_empty()));
    }

    #[test]
    fn delta_of_ramp_is_slope() {
        let x: Vec<f32> = (0..30).map(|i| 2.0 * i as f32 + 1.0).collect();
        assert!(delta(&x).iter().all(|&d| (d - 2.0).abs() < 1e-5));
    }

    #[test]
    fn dct_is_orthonormal() {
        let d = dct2_ortho(8, 8);
        for i in 0..8 {
            for j in 0..8 {
                let dot: f32 = d[i].iter().zip(&d[j]).map(|(a, b)| a * b).sum();
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expect).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn power_to_db_clamps() {
        let mut x = [1.0, 1e-12, 0.1];
        power_to_db_inplace(&mut x, Some(80.0));
        assert_eq!(x, [0.0, -80.0, -10.0]);
    }
}
