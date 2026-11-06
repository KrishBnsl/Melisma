//! Onset strength and global tempo estimation (librosa `onset_strength` + `feature.tempo`).

use rayon::prelude::*;
use realfft::{RealFftPlanner, num_complex::Complex32};

use crate::stft::{Matrix, hann_periodic};

/// Spectral-flux onset envelope from a log-mel spectrogram (`frames × mels`), with
/// librosa's alignment: lag 1, then left-padded by `1 + n_fft / (2·hop)` = 3 frames.
pub fn onset_envelope(log_mel: &Matrix) -> Vec<f32> {
    const PAD: usize = 3;
    let n = log_mel.rows();
    let m = log_mel.cols().max(1);
    let mut env = vec![0f32; n];
    for (i, e) in env.iter_mut().enumerate().skip(PAD) {
        let (cur, prev) = (log_mel.row(i - PAD + 1), log_mel.row(i - PAD));
        *e = (cur
            .iter()
            .zip(prev)
            .map(|(&a, &b)| (a - b).max(0.0) as f64)
            .sum::<f64>()
            / m as f64) as f32;
    }
    env
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TempoEstimate {
    pub bpm: f32,
    /// Mean normalized autocorrelation at the chosen lag (0..1): how strongly periodic
    /// the onsets are at that tempo.
    pub pulse_clarity: f32,
}

/// Global tempo from the mean of an 8-second autocorrelation tempogram, weighted by a
/// log-normal prior around 120 BPM (std 1 octave), capped at 320 BPM.
pub fn estimate_tempo(env: &[f32], sr: f64, hop: usize) -> TempoEstimate {
    const START_BPM: f64 = 120.0;
    const STD_BPM: f64 = 1.0;
    const MAX_TEMPO: f64 = 320.0;
    let win = ((8.0 * sr) as usize) / hop;
    let n = env.len();
    if n == 0 || win < 2 {
        return TempoEstimate::default();
    }

    // Center with linear ramps down to zero (np.pad mode="linear_ramp").
    let half = win / 2;
    let (first, last) = (env[0] as f64, env[n - 1] as f64);
    let mut padded = Vec::with_capacity(n + 2 * half);
    padded.extend((0..half).map(|i| (first * i as f64 / half as f64) as f32));
    padded.extend_from_slice(env);
    padded.extend((0..half).map(|j| (last * (half - 1 - j) as f64 / half as f64) as f32));

    let window = hann_periodic(win);
    let nfft = (2 * win - 1).next_power_of_two();
    let mut planner = RealFftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(nfft);
    let inv = planner.plan_fft_inverse(nfft);

    struct Bufs {
        acc: Vec<f64>,
        input: Vec<f32>,
        spec: Vec<Complex32>,
        output: Vec<f32>,
    }
    let acc = (0..n)
        .into_par_iter()
        .fold(
            || Bufs {
                acc: vec![0.0; win],
                input: fwd.make_input_vec(),
                spec: fwd.make_output_vec(),
                output: inv.make_output_vec(),
            },
            |mut b, t| {
                b.input.fill(0.0);
                for i in 0..win {
                    b.input[i] = padded[t + i] * window[i];
                }
                fwd.process(&mut b.input, &mut b.spec)
                    .expect("planner sizes");
                for c in b.spec.iter_mut() {
                    *c = Complex32::new(c.norm_sqr(), 0.0);
                }
                inv.process(&mut b.spec, &mut b.output)
                    .expect("power spectrum is real");
                let max = b.output[..win].iter().fold(0f32, |m, v| m.max(v.abs()));
                let norm = if max >= f32::MIN_POSITIVE { max } else { 1.0 };
                for (a, &v) in b.acc.iter_mut().zip(&b.output[..win]) {
                    *a += (v / norm) as f64;
                }
                b
            },
        )
        .map(|b| b.acc)
        .reduce(
            || vec![0.0; win],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                a
            },
        );
    let tg: Vec<f64> = acc.iter().map(|&v| v / n as f64).collect();

    let bpm_of = |lag: usize| 60.0 * sr / (hop as f64 * lag as f64);
    let min_lag = (1..win).find(|&lag| bpm_of(lag) < MAX_TEMPO).unwrap_or(win);
    let mut best: Option<(usize, f64)> = None;
    for (lag, &v) in tg.iter().enumerate().skip(min_lag) {
        let prior = -0.5 * ((bpm_of(lag).log2() - START_BPM.log2()) / STD_BPM).powi(2);
        let score = (1e6 * v).ln_1p() + prior;
        if score.is_finite() && best.is_none_or(|(_, s)| score > s) {
            best = Some((lag, score));
        }
    }
    match best {
        Some((lag, _)) => TempoEstimate {
            bpm: bpm_of(lag) as f32,
            pulse_clarity: tg[lag] as f32,
        },
        None => TempoEstimate::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn click_train_tempo() {
        let sr = 22050.0;
        let hop = 512;
        let frames_per_sec = sr / hop as f64;
        for bpm in [90.0, 120.0, 140.0] {
            let period = frames_per_sec * 60.0 / bpm;
            let n = (30.0 * frames_per_sec) as usize;
            // Onsets smeared over a couple of frames, as real onset envelopes are.
            let env: Vec<f32> = (0..n)
                .map(|i| {
                    let x = i as f64 / period;
                    let d = (x - x.round()) * period; // distance to nearest beat, in frames
                    (-0.5 * (d / 1.0).powi(2)).exp() as f32
                })
                .collect();
            let t = estimate_tempo(&env, sr, hop);
            assert!(
                ((t.bpm as f64 - bpm) / bpm).abs() < 0.04,
                "expected {bpm}, got {}",
                t.bpm
            );
            assert!(t.pulse_clarity > 0.3);
        }
    }

    #[test]
    fn onset_alignment() {
        // A single loud frame at index 10 → onset at 10 + 2 (librosa's centering shift).
        let mut lm = Matrix::zeros(20, 4);
        lm.row_mut(10).fill(1.0);
        let env = onset_envelope(&lm);
        assert_eq!(env.iter().position(|&v| v > 0.0), Some(12));
    }
}
