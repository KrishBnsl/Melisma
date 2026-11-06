//! Chroma features (librosa `chroma_stft`) and key estimation (Krumhansl–Kessler profiles).

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{stats::pearson, stft::Matrix};

pub const KEY_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// `librosa.filters.chroma(sr, n_fft, tuning=0)` → dense `12 × bins`, row 0 = C.
pub fn chroma_filterbank(sr: f64, n_fft: usize) -> Vec<Vec<f32>> {
    const N_CHROMA: f64 = 12.0;
    const CTROCT: f64 = 5.0;
    const OCTWIDTH: f64 = 2.0;
    // Fractional chroma bin of each FFT bin (A0 = 27.5 Hz is bin 0); FFT bin 0 is extrapolated.
    let mut frqbins = vec![0f64; n_fft];
    for (i, b) in frqbins.iter_mut().enumerate().skip(1) {
        let f = i as f64 * sr / n_fft as f64;
        *b = N_CHROMA * (f / 27.5).log2();
    }
    frqbins[0] = frqbins[1] - 1.5 * N_CHROMA;
    let mut binwidth: Vec<f64> = frqbins.windows(2).map(|w| (w[1] - w[0]).max(1.0)).collect();
    binwidth.push(1.0);

    let mut wts = vec![vec![0f64; n_fft]; 12];
    for (c, row) in wts.iter_mut().enumerate() {
        for f in 0..n_fft {
            let d = (frqbins[f] - c as f64 + 6.0 + 10.0 * N_CHROMA).rem_euclid(N_CHROMA) - 6.0;
            row[f] = (-0.5 * (2.0 * d / binwidth[f]).powi(2)).exp();
        }
    }
    for f in 0..n_fft {
        let norm = (0..12).map(|c| wts[c][f] * wts[c][f]).sum::<f64>().sqrt();
        let gauss = (-0.5 * ((frqbins[f] / N_CHROMA - CTROCT) / OCTWIDTH).powi(2)).exp();
        for row in wts.iter_mut() {
            if norm > f64::MIN_POSITIVE {
                row[f] /= norm;
            }
            row[f] *= gauss;
        }
    }
    // Rotate so that row 0 is C instead of A, and keep only the non-negative frequencies.
    (0..12)
        .map(|c| {
            wts[(c + 3) % 12][..n_fft / 2 + 1]
                .iter()
                .map(|&v| v as f32)
                .collect()
        })
        .collect()
}

/// Chroma per frame from a power spectrogram, each frame scaled so its max is 1.
pub fn chroma_frames(power: &Matrix, fb: &[Vec<f32>]) -> Matrix {
    let mut out = Matrix::zeros(power.rows(), 12);
    out.data_mut()
        .par_chunks_mut(12)
        .enumerate()
        .for_each(|(t, row)| {
            let p = power.row(t);
            for (o, w) in row.iter_mut().zip(fb) {
                *o = w
                    .iter()
                    .zip(p)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum::<f64>() as f32;
            }
            let max = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
            if max >= f32::MIN_POSITIVE {
                row.iter_mut().for_each(|v| *v /= max);
            }
        });
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Major,
    Minor,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyEstimate {
    /// Pitch class of the tonic, 0 = C.
    pub tonic: usize,
    pub mode: Mode,
    /// Correlation of the chroma with the best key profile (−1..1).
    pub clarity: f32,
}

impl KeyEstimate {
    pub fn name(&self) -> String {
        let m = match self.mode {
            Mode::Major => "major",
            Mode::Minor => "minor",
        };
        format!("{} {m}", KEY_NAMES[self.tonic])
    }
}

const MAJOR_PROFILE: [f32; 12] = [
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
];
const MINOR_PROFILE: [f32; 12] = [
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
];

/// Krumhansl–Kessler key finding on a mean chroma vector.
pub fn estimate_key(chroma_mean: &[f32]) -> KeyEstimate {
    let mut best = KeyEstimate {
        tonic: 0,
        mode: Mode::Major,
        clarity: f32::NEG_INFINITY,
    };
    for (mode, profile) in [(Mode::Major, &MAJOR_PROFILE), (Mode::Minor, &MINOR_PROFILE)] {
        for tonic in 0..12 {
            let rotated: Vec<f32> = (0..12).map(|i| profile[(i + 12 - tonic) % 12]).collect();
            let r = pearson(chroma_mean, &rotated);
            if r > best.clarity {
                best = KeyEstimate {
                    tonic,
                    mode,
                    clarity: r,
                };
            }
        }
    }
    best
}

/// Rotates a chroma vector so index 0 is the tonic (key-invariant harmony).
pub fn relative_to_tonic(chroma: &[f32], tonic: usize) -> [f32; 12] {
    std::array::from_fn(|i| chroma[(i + tonic) % 12])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn power_frame_with_tones(freqs: &[f64]) -> Matrix {
        let mut m = Matrix::zeros(1, 1025);
        for &f in freqs {
            let k = (f * 2048.0 / 22050.0).round() as usize;
            m.row_mut(0)[k] += 1.0;
        }
        m
    }

    #[test]
    fn a440_maps_to_a() {
        let fb = chroma_filterbank(22050.0, 2048);
        let c = chroma_frames(&power_frame_with_tones(&[440.0]), &fb);
        let argmax = (0..12)
            .max_by(|&a, &b| c.get(0, a).total_cmp(&c.get(0, b)))
            .unwrap();
        assert_eq!(KEY_NAMES[argmax], "A");
    }

    #[test]
    fn c_major_profile_is_c_major() {
        // Chroma shaped like the C-major profile itself.
        let k = estimate_key(&MAJOR_PROFILE);
        assert_eq!((k.tonic, k.mode), (0, Mode::Major));
        // Rotated to A minor.
        let a_minor: Vec<f32> = (0..12).map(|i| MINOR_PROFILE[(i + 12 - 9) % 12]).collect();
        let k = estimate_key(&a_minor);
        assert_eq!((k.tonic, k.mode), (9, Mode::Minor));
        assert_eq!(k.name(), "A minor");
    }

    #[test]
    fn relative_rotation() {
        let mut c = [0f32; 12];
        c[7] = 1.0; // G
        assert_eq!(relative_to_tonic(&c, 7)[0], 1.0);
    }
}
