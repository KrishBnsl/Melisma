//! Spectral shape (centroid, bandwidth, rolloff, flatness, contrast) and time-domain
//! frame features (RMS, zero-crossing rate), matching librosa's definitions.

use rayon::prelude::*;

use crate::{mel::power_to_db_inplace, stft::Matrix};

/// Bin centre frequencies, `k · sr / n_fft`.
pub fn fft_frequencies(sr: f64, n_fft: usize) -> Vec<f64> {
    (0..=n_fft / 2)
        .map(|k| k as f64 * sr / n_fft as f64)
        .collect()
}

/// Per-frame spectral shape descriptors computed from a magnitude spectrum.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FrameShape {
    pub centroid: f32,
    pub bandwidth: f32,
    pub rolloff: f32,
    pub flatness: f32,
}

/// Centroid, bandwidth (p=2), rolloff (85%) and flatness (power=2) for one magnitude frame.
pub fn frame_shape(mag: &[f32], freqs: &[f64]) -> FrameShape {
    let total: f64 = mag.iter().map(|&v| v as f64).sum();

    // Flatness uses the power spectrum with an amplitude floor.
    const AMIN: f64 = 1e-10;
    let (mut log_sum, mut lin_sum) = (0f64, 0f64);
    for &v in mag {
        let p = (v as f64 * v as f64).max(AMIN);
        log_sum += p.ln();
        lin_sum += p;
    }
    let n = mag.len() as f64;
    let flatness = ((log_sum / n).exp() / (lin_sum / n)) as f32;

    if total < f32::MIN_POSITIVE as f64 {
        return FrameShape {
            flatness,
            ..Default::default()
        };
    }
    let centroid: f64 = mag
        .iter()
        .zip(freqs)
        .map(|(&s, &f)| s as f64 * f)
        .sum::<f64>()
        / total;
    let bandwidth = (mag
        .iter()
        .zip(freqs)
        .map(|(&s, &f)| (s as f64 / total) * (f - centroid).powi(2))
        .sum::<f64>())
    .sqrt();
    let threshold = 0.85 * total;
    let mut cum = 0f64;
    let mut rolloff = 0f64;
    for (&s, &f) in mag.iter().zip(freqs) {
        cum += s as f64;
        if cum >= threshold {
            rolloff = f;
            break;
        }
    }
    FrameShape {
        centroid: centroid as f32,
        bandwidth: bandwidth as f32,
        rolloff: rolloff as f32,
        flatness,
    }
}

/// Bins used by each contrast band, and how many of them form the peak/valley quantile.
struct ContrastBand {
    bins: Vec<usize>,
    quantile_n: usize,
}

fn contrast_bands(freqs: &[f64], fmin: f64, n_bands: usize, quantile: f64) -> Vec<ContrastBand> {
    // Octave edges: [0, fmin, 2·fmin, …, 2^n_bands·fmin].
    let mut octa = vec![0.0];
    octa.extend((0..=n_bands).map(|k| fmin * 2f64.powi(k as i32)));
    let n = freqs.len();
    (0..=n_bands)
        .map(|k| {
            let (lo, hi) = (octa[k], octa[k + 1]);
            let mut in_band: Vec<bool> = freqs.iter().map(|&f| f >= lo && f <= hi).collect();
            let idx: Vec<usize> = (0..n).filter(|&i| in_band[i]).collect();
            if let (Some(&first), Some(&last)) = (idx.first(), idx.last()) {
                if k > 0 && first > 0 {
                    in_band[first - 1] = true;
                }
                if k == n_bands {
                    in_band[last + 1..].iter_mut().for_each(|b| *b = true);
                }
            }
            let count = in_band.iter().filter(|&&b| b).count();
            let mut bins: Vec<usize> = (0..n).filter(|&i| in_band[i]).collect();
            if k < n_bands {
                bins.pop();
            }
            let quantile_n = ((quantile * count as f64).round_ties_even() as usize).max(1);
            ContrastBand { bins, quantile_n }
        })
        .collect()
}

/// `librosa.feature.spectral_contrast` (fmin=200, n_bands=6, quantile=0.02), `frames × (n_bands+1)` in dB.
pub fn spectral_contrast(
    mag: &Matrix,
    freqs: &[f64],
    fmin: f64,
    n_bands: usize,
    quantile: f64,
) -> Matrix {
    let bands = contrast_bands(freqs, fmin, n_bands, quantile);
    let nb = bands.len();
    let mut peak = Matrix::zeros(mag.rows(), nb);
    let mut valley = Matrix::zeros(mag.rows(), nb);
    peak.data_mut()
        .par_chunks_mut(nb)
        .zip(valley.data_mut().par_chunks_mut(nb))
        .enumerate()
        .for_each(|(t, (prow, vrow))| {
            let row = mag.row(t);
            let mut buf = Vec::new();
            for (b, band) in bands.iter().enumerate() {
                buf.clear();
                buf.extend(band.bins.iter().map(|&i| row[i]));
                if buf.is_empty() {
                    continue;
                }
                buf.sort_by(f32::total_cmp);
                let q = band.quantile_n.min(buf.len());
                let mean = |s: &[f32]| s.iter().map(|&v| v as f64).sum::<f64>() / s.len() as f64;
                vrow[b] = mean(&buf[..q]) as f32;
                prow[b] = mean(&buf[buf.len() - q..]) as f32;
            }
        });
    power_to_db_inplace(peak.data_mut(), Some(80.0));
    power_to_db_inplace(valley.data_mut(), Some(80.0));
    for (p, v) in peak.data_mut().iter_mut().zip(valley.data()) {
        *p -= v;
    }
    peak
}

/// Frame RMS with centered, zero-padded frames (`librosa.feature.rms`).
pub fn rms(y: &[f32], frame: usize, hop: usize) -> Vec<f32> {
    let pad = frame / 2;
    let n_frames = 1 + y.len() / hop;
    let mut prefix = vec![0f64; y.len() + 2 * pad + 1];
    for i in 0..y.len() + 2 * pad {
        let v = if i >= pad && i < pad + y.len() {
            y[i - pad] as f64
        } else {
            0.0
        };
        prefix[i + 1] = prefix[i] + v * v;
    }
    (0..n_frames)
        .map(|t| {
            let s = t * hop;
            (((prefix[s + frame] - prefix[s]) / frame as f64).max(0.0)).sqrt() as f32
        })
        .collect()
}

/// Zero-crossing rate with centered, edge-padded frames (`librosa.feature.zero_crossing_rate`).
pub fn zero_crossing_rate(y: &[f32], frame: usize, hop: usize) -> Vec<f32> {
    const THRESHOLD: f32 = 1e-10;
    let pad = frame / 2;
    let n_frames = 1 + y.len() / hop;
    let first = *y.first().unwrap_or(&0.0);
    let last = *y.last().unwrap_or(&0.0);
    let total = y.len() + 2 * pad;
    let sample = |i: usize| -> f32 {
        let v = if i < pad {
            first
        } else if i >= pad + y.len() {
            last
        } else {
            y[i - pad]
        };
        if v.abs() <= THRESHOLD { 0.0 } else { v }
    };
    // crossing[i] = sign changes between i-1 and i; prefix sums for O(1) frame counts.
    let mut prefix = vec![0u32; total + 1];
    let mut prev = sample(0).is_sign_negative();
    for i in 1..total {
        let cur = sample(i).is_sign_negative();
        prefix[i + 1] = prefix[i] + u32::from(cur != prev);
        prev = cur;
    }
    prefix[1] = 0;
    (0..n_frames)
        .map(|t| {
            let s = t * hop;
            // Crossings strictly inside the frame: indices s+1 ..= s+frame-1.
            let c = prefix[s + frame] - prefix[s + 1];
            c as f32 / frame as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_shape() {
        let freqs = fft_frequencies(22050.0, 2048);
        let mut mag = vec![0f32; 1025];
        let k = 93; // ≈ 1001 Hz
        mag[k] = 1.0;
        let s = frame_shape(&mag, &freqs);
        assert!((s.centroid as f64 - freqs[k]).abs() < 1e-3);
        assert!(s.bandwidth < 1e-3);
        assert!((s.rolloff as f64 - freqs[k]).abs() < 1e-3);
        assert!(s.flatness < 1e-6);
    }

    #[test]
    fn white_spectrum_is_flat() {
        let freqs = fft_frequencies(22050.0, 2048);
        let s = frame_shape(&vec![1.0; 1025], &freqs);
        assert!((s.flatness - 1.0).abs() < 1e-6);
    }

    #[test]
    fn contrast_band_layout() {
        let freqs = fft_frequencies(22050.0, 2048);
        let bands = contrast_bands(&freqs, 200.0, 6, 0.02);
        assert_eq!(bands.len(), 7);
        assert_eq!(bands[0].bins, (0..18).collect::<Vec<_>>()); // 0..=18 minus the last
        assert_eq!(*bands[6].bins.last().unwrap(), 1024);
    }

    #[test]
    fn rms_of_constant() {
        let y = vec![0.5f32; 22050];
        let r = rms(&y, 2048, 512);
        assert_eq!(r.len(), 44);
        assert!((r[20] - 0.5).abs() < 1e-6);
        // The first frame is half zero padding.
        assert!((r[0] - 0.5 * 0.5f32.sqrt()).abs() < 1e-3);
    }

    #[test]
    fn zcr_of_alternating_signal() {
        let y: Vec<f32> = (0..22050)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let z = zero_crossing_rate(&y, 2048, 512);
        assert!((z[20] - 2047.0 / 2048.0).abs() < 1e-6);
    }
}
