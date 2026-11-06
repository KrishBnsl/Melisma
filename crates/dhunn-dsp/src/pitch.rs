//! YIN fundamental-frequency estimation, matching `librosa.yin` (librosa 1.0).

use rayon::prelude::*;
use realfft::{RealFftPlanner, num_complex::Complex32};

pub struct Yin {
    /// Estimated f0 per frame in Hz (meaningless on unvoiced/silent frames — mask before use).
    pub f0: Vec<f32>,
    /// Minimum of the cumulative-mean-normalized difference per frame: low ⇒ periodic.
    pub aperiodicity: Vec<f32>,
}

/// YIN with centered, zero-padded frames.
///
/// Per frame `x` of length `N`: `d(k) = 2·(acf(0) − acf(k)) − Σ_{m<k} x[m]²` for
/// `k ∈ 1..=max_period`, normalized by its cumulative mean; the period is the first trough
/// below `trough_threshold` (else the global minimum), refined by parabolic interpolation.
pub fn yin(
    y: &[f32],
    sr: f64,
    frame_length: usize,
    hop: usize,
    fmin: f64,
    fmax: f64,
    trough_threshold: f64,
) -> Yin {
    let min_period = ((sr / fmax).floor() as usize).max(1);
    let max_period = ((sr / fmin).ceil() as usize).min(frame_length - 1);
    let pad = frame_length / 2;
    let mut padded = vec![0f32; y.len() + 2 * pad];
    padded[pad..pad + y.len()].copy_from_slice(y);
    let n_frames = 1 + y.len() / hop;

    // Linear autocorrelation via a zero-padded FFT of size ≥ 2N − 1.
    let nfft = (2 * frame_length - 1).next_power_of_two();
    let mut planner = RealFftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(nfft);
    let inv = planner.plan_fft_inverse(nfft);

    struct Bufs {
        input: Vec<f32>,
        spec: Vec<Complex32>,
        acf: Vec<f32>,
        d: Vec<f64>,
        v: Vec<f64>,
    }
    let results: Vec<(f32, f32)> = (0..n_frames)
        .into_par_iter()
        .map_init(
            || Bufs {
                input: fwd.make_input_vec(),
                spec: fwd.make_output_vec(),
                acf: inv.make_output_vec(),
                d: vec![0.0; max_period + 1],
                v: Vec::with_capacity(max_period + 1),
            },
            |b, t| {
                let frame = &padded[t * hop..t * hop + frame_length];
                b.input.fill(0.0);
                b.input[..frame_length].copy_from_slice(frame);
                fwd.process(&mut b.input, &mut b.spec)
                    .expect("planner sizes");
                for c in b.spec.iter_mut() {
                    *c = Complex32::new(c.norm_sqr(), 0.0);
                }
                inv.process(&mut b.spec, &mut b.acf)
                    .expect("power spectrum is real");
                let scale = 1.0 / nfft as f64;
                let acf0 = b.acf[0] as f64 * scale;

                // d(k) with the running head energy Σ_{m<k} x[m]².
                b.d[0] = 0.0;
                let mut head = 0f64;
                for k in 1..=max_period {
                    head += frame[k - 1] as f64 * frame[k - 1] as f64;
                    b.d[k] = 2.0 * (acf0 - b.acf[k] as f64 * scale) - head;
                }

                // Cumulative mean normalized difference over [min_period, max_period].
                b.v.clear();
                let mut cum = 0f64;
                for k in 1..=max_period {
                    cum += b.d[k];
                    if k >= min_period {
                        b.v.push(b.d[k] / (cum / k as f64 + f32::MIN_POSITIVE as f64));
                    }
                }
                let v = &b.v;
                let len = v.len();
                if len < 2 {
                    return (0.0, 1.0);
                }
                // util.localmin: strictly below the left neighbour, not above the right one.
                let is_trough = |i: usize| -> bool {
                    if i == 0 {
                        v[0] < v[1]
                    } else if i == len - 1 {
                        v[i] < v[i - 1]
                    } else {
                        v[i] < v[i - 1] && v[i] <= v[i + 1]
                    }
                };
                let global_min = (0..len).min_by(|&a, &c| v[a].total_cmp(&v[c])).unwrap_or(0);
                let idx = (0..len)
                    .find(|&i| is_trough(i) && v[i] < trough_threshold)
                    .unwrap_or(global_min);
                let shift = if idx == 0 || idx == len - 1 {
                    0.0
                } else {
                    let a = v[idx + 1] + v[idx - 1] - 2.0 * v[idx];
                    let bb = (v[idx + 1] - v[idx - 1]) / 2.0;
                    if bb.abs() >= a.abs() { 0.0 } else { -bb / a }
                };
                let period = (min_period + idx) as f64 + shift;
                ((sr / period) as f32, v[global_min] as f32)
            },
        )
        .collect();

    Yin {
        f0: results.iter().map(|r| r.0).collect(),
        aperiodicity: results.iter().map(|r| r.1).collect(),
    }
}

/// Hz → MIDI note number (A4 = 69).
pub fn hz_to_midi(f: f32) -> f32 {
    69.0 + 12.0 * (f / 440.0).log2()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_f0() {
        let sr = 22050.0;
        for f in [110.0f32, 220.0, 440.0, 880.0] {
            let y: Vec<f32> = (0..22050)
                .map(|i| (std::f32::consts::TAU * f * i as f32 / sr as f32).sin())
                .collect();
            let r = yin(&y, sr, 2048, 512, 65.0, 2093.0, 0.1);
            let mid = r.f0[r.f0.len() / 2];
            assert!((mid - f).abs() / f < 0.01, "expected {f}, got {mid}");
            assert!(r.aperiodicity[r.f0.len() / 2] < 0.1);
        }
    }

    #[test]
    fn midi_conversion() {
        assert!((hz_to_midi(440.0) - 69.0).abs() < 1e-6);
        assert!((hz_to_midi(261.6256) - 60.0).abs() < 1e-3);
    }
}
