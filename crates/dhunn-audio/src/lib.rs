//! Audio decoding, mono downmix and resampling.
//!
//! Decoding uses symphonia (pure Rust: MP3, FLAC, WAV/AIFF, AAC/ALAC in MP4, Ogg Vorbis),
//! resampling uses rubato's FFT resampler.

use std::{fs::File, path::Path};

use rubato::{FftFixedIn, Resampler};
use symphonia::core::{
    audio::SampleBuffer,
    codecs::{CODEC_TYPE_NULL, DecoderOptions},
    errors::Error as SymphoniaError,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported or corrupt audio: {0}")]
    Decode(#[from] SymphoniaError),
    #[error("no decodable audio track found")]
    NoTrack,
    #[error("stream does not declare a sample rate")]
    UnknownSampleRate,
    #[error("audio contains no samples")]
    Empty,
    #[error("resampling failed: {0}")]
    Resample(String),
}

/// Mono PCM audio in `[-1, 1]`.
#[derive(Debug, Clone)]
pub struct MonoAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl MonoAudio {
    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }
}

/// Decodes the first audio track of a file and downmixes it to mono by averaging channels.
pub fn decode_mono(path: &Path) -> Result<MonoAudio, AudioError> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or(AudioError::NoTrack)?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or(AudioError::UnknownSampleRate)?;
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut mono = Vec::new();
    let mut buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            // Chained streams (e.g. Ogg with new parameters); treat as end of first stream.
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // A corrupt packet: skip it, as players do.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        };
        let spec = *decoded.spec();
        let channels = spec.channels.count().max(1);
        let frames = decoded.capacity();
        let needs_new = buf
            .as_ref()
            .is_none_or(|b| b.capacity() < frames * channels);
        if needs_new {
            buf = Some(SampleBuffer::new(frames as u64, spec));
        }
        let b = buf.as_mut().expect("buffer allocated above");
        b.copy_interleaved_ref(decoded);
        let scale = 1.0 / channels as f32;
        mono.extend(
            b.samples()
                .chunks_exact(channels)
                .map(|f| f.iter().sum::<f32>() * scale),
        );
    }

    if mono.is_empty() {
        return Err(AudioError::Empty);
    }
    Ok(MonoAudio {
        samples: mono,
        sample_rate,
    })
}

/// Resamples mono audio. Output length is `ceil(len * to / from)`, with the resampler's
/// delay compensated so output is time-aligned with input.
pub fn resample(samples: &[f32], from: u32, to: u32) -> Result<Vec<f32>, AudioError> {
    if from == to || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    let err = |e: &dyn std::fmt::Display| AudioError::Resample(e.to_string());
    const CHUNK: usize = 4096;
    let mut r =
        FftFixedIn::<f32>::new(from as usize, to as usize, CHUNK, 2, 1).map_err(|e| err(&e))?;
    let delay = r.output_delay();
    let expected = (samples.len() as u64 * to as u64).div_ceil(from as u64) as usize;
    let mut out = Vec::with_capacity(expected + delay + CHUNK);

    let mut pos = 0;
    while samples.len() - pos >= r.input_frames_next() {
        let n = r.input_frames_next();
        let o = r
            .process(&[&samples[pos..pos + n]], None)
            .map_err(|e| err(&e))?;
        out.extend_from_slice(&o[0]);
        pos += n;
    }
    if pos < samples.len() {
        let o = r
            .process_partial(Some(&[&samples[pos..]][..]), None)
            .map_err(|e| err(&e))?;
        out.extend_from_slice(&o[0]);
    }
    // Flush the resampler until the delayed tail is out.
    let mut guard = 0;
    while out.len() < delay + expected && guard < 64 {
        let o = r
            .process_partial(None::<&[&[f32]]>, None)
            .map_err(|e| err(&e))?;
        if o[0].is_empty() {
            break;
        }
        out.extend_from_slice(&o[0]);
        guard += 1;
    }
    out.drain(..delay.min(out.len()));
    out.truncate(expected);
    Ok(out)
}

/// Decodes a file to mono and resamples it to `target_rate`.
pub fn load_mono(path: &Path, target_rate: u32) -> Result<MonoAudio, AudioError> {
    let a = decode_mono(path)?;
    let samples = resample(&a.samples, a.sample_rate, target_rate)?;
    Ok(MonoAudio {
        samples,
        sample_rate: target_rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// Writes a 16-bit PCM WAV file by hand (no extra dev-dependency needed).
    fn write_wav(path: &Path, rate: u32, channels: u16, interleaved: &[f32]) {
        let data: Vec<u8> = interleaved
            .iter()
            .flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes())
            .collect();
        let mut b = Vec::new();
        b.extend_from_slice(b"RIFF");
        b.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
        b.extend_from_slice(b"WAVEfmt ");
        b.extend_from_slice(&16u32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&channels.to_le_bytes());
        b.extend_from_slice(&rate.to_le_bytes());
        b.extend_from_slice(&(rate * channels as u32 * 2).to_le_bytes());
        b.extend_from_slice(&(channels * 2).to_le_bytes());
        b.extend_from_slice(&16u16.to_le_bytes());
        b.extend_from_slice(b"data");
        b.extend_from_slice(&(data.len() as u32).to_le_bytes());
        b.extend_from_slice(&data);
        std::fs::write(path, b).unwrap();
    }

    fn sine(freq: f32, rate: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.5 * (TAU * freq * i as f32 / rate as f32).sin())
            .collect()
    }

    /// Estimates frequency by counting upward zero crossings.
    fn est_freq(x: &[f32], rate: u32) -> f32 {
        let ups = x.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
        ups as f32 * rate as f32 / x.len() as f32
    }

    #[test]
    fn decodes_stereo_wav_to_mono() {
        let dir = std::env::temp_dir().join(format!("dhunn-audio-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stereo.wav");
        // Left = sine, right = silence → mono is half amplitude.
        let left = sine(440.0, 44_100, 44_100);
        let inter: Vec<f32> = left.iter().flat_map(|&l| [l, 0.0]).collect();
        write_wav(&path, 44_100, 2, &inter);

        let a = decode_mono(&path).unwrap();
        assert_eq!(a.sample_rate, 44_100);
        assert_eq!(a.samples.len(), 44_100);
        let peak = a.samples.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!((peak - 0.25).abs() < 0.01, "peak {peak}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resample_preserves_length_and_pitch() {
        let x = sine(440.0, 44_100, 44_100 * 2);
        let y = resample(&x, 44_100, 22_050).unwrap();
        assert_eq!(y.len(), 44_100);
        let f = est_freq(&y[1000..y.len() - 1000], 22_050);
        assert!((f - 440.0).abs() < 2.0, "freq {f}");
        // Delay compensated: the waveform starts in phase with the input.
        let x_ds: Vec<f32> = x.iter().step_by(2).copied().collect();
        let err: f32 = (2000..3000).map(|i| (x_ds[i] - y[i]).abs()).sum::<f32>() / 1000.0;
        assert!(err < 0.01, "alignment error {err}");
    }

    #[test]
    fn resample_is_identity_for_same_rate() {
        let x = sine(100.0, 22_050, 1000);
        assert_eq!(resample(&x, 22_050, 22_050).unwrap(), x);
    }
}

#[cfg(test)]
mod quality_probe {
    use super::*;

    /// Upsampling must not create content above the source Nyquist (imaging).
    #[test]
    fn upsampling_adds_no_images() {
        // Deterministic pseudo-noise at 11025 Hz.
        let mut s = 12345u32;
        let x: Vec<f32> = (0..11025 * 4)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5
            })
            .collect();
        let y = resample(&x, 11025, 22050).unwrap();
        // Energy above 5.6 kHz vs below, via a naive DFT on a slice.
        let n = 4096;
        let seg = &y[8000..8000 + n];
        let (mut lo, mut hi) = (0f64, 0f64);
        for k in 1..n / 2 {
            let (mut re, mut im) = (0f64, 0f64);
            for (i, &v) in seg.iter().enumerate() {
                let w = 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos();
                let ph = std::f64::consts::TAU * (k * i) as f64 / n as f64;
                re += v as f64 * w * ph.cos();
                im -= v as f64 * w * ph.sin();
            }
            let p = re * re + im * im;
            let f = k as f64 * 22050.0 / n as f64;
            if f < 5000.0 {
                lo += p
            } else if f > 5600.0 {
                hi += p
            }
        }
        let ratio_db = 10.0 * (hi / lo).log10();
        eprintln!("image energy relative to passband: {ratio_db:.1} dB");
        assert!(ratio_db < -60.0, "imaging at {ratio_db:.1} dB");
    }
}
