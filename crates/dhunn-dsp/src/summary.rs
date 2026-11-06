//! Whole-track analysis: runs every extractor and summarizes per-frame features into a
//! fixed, versioned, human-readable [`DspSummary`] plus a flat [`FeatureVec`].

use std::time::Instant;

use dhunn_core::FeatureVec;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    chroma::{self, Mode},
    hpss, mel, pitch, rhythm, spectral,
    stats::{self, Stat},
    stft::Stft,
};

/// Bump whenever any feature definition or the vector layout changes.
pub const DSP_VERSION: u32 = 1;

pub const N_FFT: usize = 2048;
pub const HOP: usize = 512;
pub const N_MELS: usize = 128;
pub const N_MFCC: usize = 20;
pub const N_CONTRAST_BANDS: usize = 6;
const HPSS_KERNEL: usize = 31;
/// Upper edge of the "thump" (kick / bass transient) band.
const THUMP_MAX_HZ: f64 = 150.0;
pub const PITCH_FMIN: f64 = 65.0;
pub const PITCH_FMAX: f64 = 2093.0;
const YIN_TROUGH_THRESHOLD: f64 = 0.1;
/// Frames with YIN aperiodicity below this count as voiced/pitched.
const VOICED_THRESHOLD: f32 = 0.25;
/// Frames quieter than (loudest frame − this) are treated as silence.
const SILENCE_DB: f32 = 40.0;

#[derive(Debug, thiserror::Error)]
pub enum DspError {
    #[error("signal too short to analyze: {samples} samples (need at least {min})")]
    TooShort { samples: usize, min: usize },
    #[error("unsupported sample rate {0} Hz")]
    InvalidSampleRate(u32),
    #[error("signal contains NaN or infinite samples")]
    NonFinite,
}

#[derive(Debug, Clone)]
pub struct AnalyzeOptions {
    /// Longer tracks are analyzed on a centered excerpt of this length (bounds memory/time,
    /// e.g. for hour-long mixes). `None` analyzes everything.
    pub max_seconds: Option<f64>,
    /// Compute extra, slower outputs used only to validate against reference implementations.
    pub validation_diagnostics: bool,
}

impl Default for AnalyzeOptions {
    fn default() -> Self {
        Self {
            max_seconds: Some(600.0),
            validation_diagnostics: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Spectral {
    /// Brightness: "centre of mass" of the spectrum.
    pub centroid_hz: Stat,
    pub bandwidth_hz: Stat,
    /// Frequency below which 85% of spectral magnitude lies (treble extent).
    pub rolloff_hz: Stat,
    /// 0 = tonal, 1 = noise-like.
    pub flatness: Stat,
    /// Peak-vs-valley contrast per octave band (dB), lowest band first.
    pub contrast_db: Vec<Stat>,
    pub zcr: Stat,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Timbre {
    pub mfcc: Vec<Stat>,
    pub mfcc_delta: Vec<Stat>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Harmony {
    /// Mean chroma, index 0 = C.
    pub chroma_mean: [f32; 12],
    /// Mean chroma rotated so index 0 is the tonic (key-invariant).
    pub chroma_relative: [f32; 12],
    /// Tonic pitch class, 0 = C.
    pub key: u8,
    pub key_name: String,
    pub mode: Mode,
    /// Correlation with the best key profile; higher = more clearly in one key.
    pub key_clarity: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Percussion {
    /// Share of spectral energy in the harmonic (sustained) part, 0..1.
    pub harmonic_ratio: f32,
    pub onset: Stat,
    /// Share of total energy that is percussive and below 150 Hz (kick/bass punch).
    pub thump_ratio: f32,
    /// Positive dB flux of low-band percussive energy: how hard the low end hits.
    pub thump_flux: Stat,
    pub thump_flux_p90: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rhythm {
    pub tempo_bpm: f32,
    pub pulse_clarity: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dynamics {
    pub rms_mean: f32,
    pub rms_db: Stat,
    /// Spread between loud and quiet passages (p95 − p10 of frame dB, non-silent frames).
    pub dynamic_range_db: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pitch {
    /// Median f0 of voiced frames (Hz, 0 if none).
    pub f0_median_hz: f32,
    /// p90 − p10 of voiced f0, in semitones.
    pub f0_range_semitones: f32,
    /// Share of non-silent frames with a clear pitch.
    pub voiced_fraction: f32,
    /// 1 − median YIN aperiodicity of non-silent frames.
    pub periodicity: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DspSummary {
    pub version: u32,
    pub duration_s: f32,
    pub analyzed_s: f32,
    pub spectral: Spectral,
    pub timbre: Timbre,
    pub harmony: Harmony,
    pub percussion: Percussion,
    pub rhythm: Rhythm,
    pub dynamics: Dynamics,
    pub pitch: Pitch,
}

#[derive(Debug, Clone, Serialize)]
pub struct StageTiming {
    pub stage: &'static str,
    pub ms: f64,
}

/// Extra outputs useful for debugging and validating against reference implementations.
#[derive(Debug, Clone, Serialize)]
pub struct Diagnostics {
    pub n_frames: usize,
    pub stages: Vec<StageTiming>,
    /// Only with [`AnalyzeOptions::validation_diagnostics`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation: Option<ValidationDiagnostics>,
}

/// Well-defined intermediate values to compare against librosa (see `ml/validate_dsp.py`).
#[derive(Debug, Clone, Serialize)]
pub struct ValidationDiagnostics {
    /// Median YIN f0 of the *original* signal over non-silent frames.
    pub yin_raw_f0_median_nonsilent_hz: f32,
    /// Per-frame YIN f0 of the original signal (all frames).
    pub yin_raw_f0: Vec<f32>,
    /// RMS of the re-synthesized harmonic signal (checks HPSS masks + inverse STFT).
    pub harmonic_rms: f32,
}

impl DspSummary {
    /// Flat `(name, value)` list in a fixed order; the basis of [`Self::feature_vec`].
    pub fn named_values(&self) -> Vec<(String, f32)> {
        let mut v: Vec<(String, f32)> = Vec::with_capacity(140);
        let stat = |v: &mut Vec<(String, f32)>, name: &str, s: Stat| {
            v.push((format!("{name}_mean"), s.mean));
            v.push((format!("{name}_std"), s.std));
        };
        let s = &self.spectral;
        stat(&mut v, "centroid_hz", s.centroid_hz);
        stat(&mut v, "bandwidth_hz", s.bandwidth_hz);
        stat(&mut v, "rolloff_hz", s.rolloff_hz);
        stat(&mut v, "flatness", s.flatness);
        stat(&mut v, "zcr", s.zcr);
        for (i, c) in s.contrast_db.iter().enumerate() {
            stat(&mut v, &format!("contrast{i}_db"), *c);
        }
        for (i, c) in self.timbre.mfcc.iter().enumerate() {
            stat(&mut v, &format!("mfcc{i:02}"), *c);
        }
        for (i, c) in self.timbre.mfcc_delta.iter().enumerate() {
            stat(&mut v, &format!("mfcc_delta{i:02}"), *c);
        }
        let h = &self.harmony;
        for (i, c) in h.chroma_relative.iter().enumerate() {
            v.push((format!("chroma_rel{i:02}"), *c));
        }
        v.push((
            "mode_major".into(),
            if h.mode == Mode::Major { 1.0 } else { 0.0 },
        ));
        v.push(("key_clarity".into(), h.key_clarity));
        let p = &self.percussion;
        v.push(("harmonic_ratio".into(), p.harmonic_ratio));
        stat(&mut v, "onset", p.onset);
        v.push(("thump_ratio".into(), p.thump_ratio));
        stat(&mut v, "thump_flux", p.thump_flux);
        v.push(("thump_flux_p90".into(), p.thump_flux_p90));
        let r = &self.rhythm;
        v.push((
            "log2_tempo".into(),
            if r.tempo_bpm > 0.0 {
                r.tempo_bpm.log2()
            } else {
                0.0
            },
        ));
        v.push(("pulse_clarity".into(), r.pulse_clarity));
        let d = &self.dynamics;
        stat(&mut v, "rms_db", d.rms_db);
        v.push(("dynamic_range_db".into(), d.dynamic_range_db));
        let pi = &self.pitch;
        v.push((
            "f0_midi_median".into(),
            if pi.f0_median_hz > 0.0 {
                pitch::hz_to_midi(pi.f0_median_hz)
            } else {
                0.0
            },
        ));
        v.push(("f0_range_semitones".into(), pi.f0_range_semitones));
        v.push(("voiced_fraction".into(), pi.voiced_fraction));
        v.push(("periodicity".into(), pi.periodicity));
        v
    }

    pub fn feature_vec(&self) -> FeatureVec {
        let (names, values) = self.named_values().into_iter().unzip();
        FeatureVec {
            kind: "dsp".into(),
            version: self.version,
            names,
            values,
        }
    }
}

struct Timer {
    last: Instant,
    stages: Vec<StageTiming>,
}

impl Timer {
    fn new() -> Self {
        Self {
            last: Instant::now(),
            stages: Vec::new(),
        }
    }

    fn lap(&mut self, stage: &'static str) {
        let now = Instant::now();
        self.stages.push(StageTiming {
            stage,
            ms: (now - self.last).as_secs_f64() * 1e3,
        });
        self.last = now;
    }
}

/// Analyzes mono audio. Designed for [`dhunn_core::ANALYSIS_SAMPLE_RATE`] (22.05 kHz) but
/// works at any rate ≥ 8 kHz.
pub fn analyze(
    y_full: &[f32],
    sr: u32,
    opts: &AnalyzeOptions,
) -> Result<(DspSummary, Diagnostics), DspError> {
    if sr < 8000 {
        return Err(DspError::InvalidSampleRate(sr));
    }
    if y_full.len() < N_FFT {
        return Err(DspError::TooShort {
            samples: y_full.len(),
            min: N_FFT,
        });
    }
    if y_full.iter().any(|v| !v.is_finite()) {
        return Err(DspError::NonFinite);
    }
    let srf = sr as f64;
    let y = match opts.max_seconds {
        Some(max) if y_full.len() as f64 > max * srf => {
            let n = (max * srf) as usize;
            let start = (y_full.len() - n) / 2;
            &y_full[start..start + n]
        }
        _ => y_full,
    };

    let mut timer = Timer::new();
    let stft = Stft::new(N_FFT, HOP);
    let mag = stft.magnitude(y);
    let n_frames = mag.rows();
    let power = mag.map(|v| v * v);
    timer.lap("stft");

    let freqs = spectral::fft_frequencies(srf, N_FFT);
    let shapes: Vec<spectral::FrameShape> = (0..n_frames)
        .into_par_iter()
        .map(|t| spectral::frame_shape(mag.row(t), &freqs))
        .collect();
    let pick =
        |f: fn(&spectral::FrameShape) -> f32| Stat::of(&shapes.iter().map(f).collect::<Vec<_>>());
    let contrast =
        spectral::spectral_contrast(&mag, &freqs, 200.0, N_CONTRAST_BANDS, 0.02).transpose();
    let spectral_out = Spectral {
        centroid_hz: pick(|s| s.centroid),
        bandwidth_hz: pick(|s| s.bandwidth),
        rolloff_hz: pick(|s| s.rolloff),
        flatness: pick(|s| s.flatness),
        contrast_db: (0..contrast.rows())
            .map(|b| Stat::of(contrast.row(b)))
            .collect(),
        zcr: Stat::of(&spectral::zero_crossing_rate(y, N_FFT, HOP)),
    };
    timer.lap("spectral");

    let mel_fb = mel::mel_filterbank(srf, N_FFT, N_MELS, 0.0, srf / 2.0);
    let mut log_mel = mel_fb.apply(&power);
    mel::power_to_db_inplace(log_mel.data_mut(), Some(80.0));
    let mfcc = mel::mfcc(&log_mel, N_MFCC).transpose();
    let timbre = Timbre {
        mfcc: (0..N_MFCC).map(|k| Stat::of(mfcc.row(k))).collect(),
        mfcc_delta: (0..N_MFCC)
            .map(|k| Stat::of(&mel::delta(mfcc.row(k))))
            .collect(),
    };
    timer.lap("timbre");

    let chroma_fb = chroma::chroma_filterbank(srf, N_FFT);
    let chroma_mean_v = chroma::chroma_frames(&power, &chroma_fb).column_means();
    let chroma_mean: [f32; 12] = std::array::from_fn(|i| chroma_mean_v[i]);
    let key = chroma::estimate_key(&chroma_mean);
    let harmony = Harmony {
        chroma_mean,
        chroma_relative: chroma::relative_to_tonic(&chroma_mean, key.tonic),
        key: key.tonic as u8,
        key_name: key.name(),
        mode: key.mode,
        key_clarity: key.clarity,
    };
    drop(power);
    timer.lap("harmony");

    let onset = rhythm::onset_envelope(&log_mel);
    let tempo = rhythm::estimate_tempo(&onset, srf, HOP);
    drop(log_mel);
    timer.lap("rhythm");

    let low_bins = 1..freqs
        .iter()
        .position(|&f| f >= THUMP_MAX_HZ)
        .unwrap_or(freqs.len());
    let hp = hpss::hpss(&mag, HPSS_KERNEL, low_bins);
    drop(mag);
    let total_energy = hp.harmonic_energy + hp.percussive_energy;
    let ratio = |x: f64| {
        if total_energy > 0.0 {
            (x / total_energy) as f32
        } else {
            0.0
        }
    };
    let mut low_db: Vec<f32> = hp.low_percussive_energy.iter().map(|&e| e as f32).collect();
    mel::power_to_db_inplace(&mut low_db, Some(80.0));
    let mut flux = vec![0f32; low_db.len()];
    for t in 1..low_db.len() {
        flux[t] = (low_db[t] - low_db[t - 1]).max(0.0);
    }
    let percussion = Percussion {
        harmonic_ratio: ratio(hp.harmonic_energy),
        onset: Stat::of(&onset),
        thump_ratio: ratio(hp.low_percussive_energy.iter().sum()),
        thump_flux: Stat::of(&flux),
        thump_flux_p90: stats::percentile(&flux, 90.0),
    };
    timer.lap("hpss");

    let rms = spectral::rms(y, N_FFT, HOP);
    let rms_db: Vec<f32> = rms.iter().map(|&r| 20.0 * r.max(1e-5).log10()).collect();
    let max_db = rms_db.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let nonsilent: Vec<bool> = rms_db.iter().map(|&d| d > max_db - SILENCE_DB).collect();
    let loud_db: Vec<f32> = rms_db
        .iter()
        .zip(&nonsilent)
        .filter(|(_, ns)| **ns)
        .map(|(d, _)| *d)
        .collect();
    let dynamics = Dynamics {
        rms_mean: stats::mean(&rms),
        rms_db: Stat::of(&rms_db),
        dynamic_range_db: stats::percentile(&loud_db, 95.0) - stats::percentile(&loud_db, 10.0),
    };

    let y_harm = stft.istft_masked(y, &hp.harmonic_mask);
    drop(hp);
    let yin = pitch::yin(
        &y_harm,
        srf,
        N_FFT,
        HOP,
        PITCH_FMIN,
        PITCH_FMAX,
        YIN_TROUGH_THRESHOLD,
    );
    let validation = opts.validation_diagnostics.then(|| {
        let raw = pitch::yin(
            y,
            srf,
            N_FFT,
            HOP,
            PITCH_FMIN,
            PITCH_FMAX,
            YIN_TROUGH_THRESHOLD,
        );
        let f0: Vec<f32> = raw
            .f0
            .iter()
            .zip(&nonsilent)
            .filter(|(_, ns)| **ns)
            .map(|(f, _)| *f)
            .collect();
        let harmonic_rms =
            (y_harm.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / y_harm.len() as f64).sqrt();
        ValidationDiagnostics {
            yin_raw_f0_median_nonsilent_hz: stats::median(&f0),
            yin_raw_f0: raw.f0,
            harmonic_rms: harmonic_rms as f32,
        }
    });
    let mut aper_nonsilent = Vec::new();
    let mut voiced_midi = Vec::new();
    for ((&f0, &ap), &ns) in yin.f0.iter().zip(&yin.aperiodicity).zip(&nonsilent) {
        if !ns {
            continue;
        }
        aper_nonsilent.push(ap);
        if ap < VOICED_THRESHOLD && f0 > 0.0 {
            voiced_midi.push(pitch::hz_to_midi(f0));
        }
    }
    let pitch_out = Pitch {
        f0_median_hz: if voiced_midi.is_empty() {
            0.0
        } else {
            440.0 * 2f32.powf((stats::median(&voiced_midi) - 69.0) / 12.0)
        },
        f0_range_semitones: stats::percentile(&voiced_midi, 90.0)
            - stats::percentile(&voiced_midi, 10.0),
        voiced_fraction: if aper_nonsilent.is_empty() {
            0.0
        } else {
            voiced_midi.len() as f32 / aper_nonsilent.len() as f32
        },
        periodicity: (1.0 - stats::median(&aper_nonsilent)).clamp(0.0, 1.0),
    };
    timer.lap("pitch");

    let summary = DspSummary {
        version: DSP_VERSION,
        duration_s: (y_full.len() as f64 / srf) as f32,
        analyzed_s: (y.len() as f64 / srf) as f32,
        spectral: spectral_out,
        timbre,
        harmony,
        percussion,
        rhythm: Rhythm {
            tempo_bpm: tempo.bpm,
            pulse_clarity: tempo.pulse_clarity,
        },
        dynamics,
        pitch: pitch_out,
    };
    let diagnostics = Diagnostics {
        n_frames,
        stages: timer.stages,
        validation,
    };
    Ok((summary, diagnostics))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// 10 s of a I–IV–V–I progression in C major plus a kick-like thump at 120 BPM.
    fn synthetic_track(sr: f32) -> Vec<f32> {
        let n = (10.0 * sr) as usize;
        let chords = [
            [261.63f32, 329.63, 392.0], // C
            [349.23, 440.0, 523.25],    // F
            [392.0, 493.88, 587.33],    // G
            [261.63, 329.63, 392.0],    // C
        ];
        (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                let chord = chords[((t / 2.5) as usize).min(3)];
                let tonal: f32 = chord
                    .iter()
                    .flat_map(|&f| (1..=2).map(move |h| (TAU * f * h as f32 * t).sin() / h as f32))
                    .sum::<f32>()
                    * 0.08;
                let beat_t = t % 0.5;
                let kick = (TAU * 60.0 * beat_t).sin() * (-beat_t * 30.0).exp() * 0.6;
                tonal + kick
            })
            .collect()
    }

    #[test]
    fn analyzes_synthetic_track() {
        let sr = 22050;
        let y = synthetic_track(sr as f32);
        let (s, d) = analyze(&y, sr, &AnalyzeOptions::default()).unwrap();
        assert_eq!(d.n_frames, 1 + y.len() / HOP);
        assert_eq!(s.harmony.key_name, "C major");
        assert!(
            (s.rhythm.tempo_bpm - 120.0).abs() < 5.0,
            "tempo {}",
            s.rhythm.tempo_bpm
        );
        assert!(
            s.percussion.thump_ratio > 0.05,
            "thump {}",
            s.percussion.thump_ratio
        );
        assert!(s.pitch.voiced_fraction > 0.0);

        let v = s.feature_vec();
        assert_eq!(v.names.len(), v.values.len());
        assert!(v.values.iter().all(|x| x.is_finite()));
        let unique: std::collections::HashSet<_> = v.names.iter().collect();
        assert_eq!(unique.len(), v.names.len(), "duplicate feature names");
    }

    #[test]
    fn silence_is_finite() {
        let y = vec![0f32; 22050 * 2];
        let (s, _) = analyze(&y, 22050, &AnalyzeOptions::default()).unwrap();
        assert!(s.feature_vec().values.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn rejects_bad_input() {
        assert!(matches!(
            analyze(&[0.0; 100], 22050, &Default::default()),
            Err(DspError::TooShort { .. })
        ));
        assert!(matches!(
            analyze(&[f32::NAN; 4096], 22050, &Default::default()),
            Err(DspError::NonFinite)
        ));
    }

    #[test]
    fn long_input_is_excerpted() {
        let y = vec![0.1f32; 22050 * 5];
        let (s, _) = analyze(
            &y,
            22050,
            &AnalyzeOptions {
                max_seconds: Some(2.0),
                ..Default::default()
            },
        )
        .unwrap();
        assert!((s.analyzed_s - 2.0).abs() < 1e-3);
        assert!((s.duration_s - 5.0).abs() < 1e-3);
    }
}
