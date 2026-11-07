//! CLAP embeddings: audio and text in one shared space, via ONNX Runtime.
//!
//! The model files come from `ml/export_clap.py` (a one-time export of LAION's music CLAP):
//! `audio.onnx` takes 10 s of mono 48 kHz audio (its log-mel front end is inside the graph),
//! `text.onnx` takes RoBERTa token ids, and both return unit vectors in the same 512-d space,
//! so "how well does this text describe this track" is a dot product.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use dhunn_audio::MonoAudio;
use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::{AutoDevicePolicy, GraphOptimizationLevel};
use ort::value::Tensor;
use serde::Deserialize;
use tokenizers::Tokenizer;

/// The name embeddings are stored under.
pub const MODEL: &str = "clap";

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("no CLAP model in {0} (export one with `uv run ml/export_clap.py --out <dir>`)")]
    Missing(PathBuf),
    #[error("reading {0}: {1}")]
    Io(PathBuf, std::io::Error),
    #[error("bad model manifest: {0}")]
    Manifest(#[from] serde_json::Error),
    #[error("ONNX Runtime: {0}")]
    Ort(#[from] ort::Error),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error(transparent)]
    Audio(#[from] dhunn_audio::AudioError),
    #[error("model returned an unexpected shape: {0:?}")]
    Shape(Vec<i64>),
}

pub type Result<T> = std::result::Result<T, EmbedError>;

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub model: String,
    pub version: u32,
    pub dims: usize,
    pub sample_rate: u32,
    pub clip_samples: usize,
    pub max_tokens: usize,
    pub check: Check,
}

/// Reference values written by the export, for tests.
#[derive(Debug, Clone, Deserialize)]
pub struct Check {
    pub text: String,
    pub input_ids: Vec<i64>,
    pub embedding_head: Vec<f32>,
    /// Embedding of 10 s of a 440 Hz sine at amplitude 0.5.
    pub sine_440_head: Vec<f32>,
}

pub struct Clap {
    pub manifest: Manifest,
    /// `None` when loaded for text only.
    audio: Option<Mutex<Session>>,
    text: Mutex<Session>,
    tokenizer: Tokenizer,
}

/// Where the model lives: `$DHUNN_MODELS/clap`, else `<data dir>/dhunn/models/clap`, else
/// `./models/clap`.
pub fn default_dir(data_dir: Option<&Path>) -> PathBuf {
    if let Some(m) = std::env::var_os("DHUNN_MODELS") {
        return PathBuf::from(m).join("clap");
    }
    if let Some(d) = data_dir {
        let p = d.join("dhunn").join("models").join("clap");
        if p.join("clap.json").exists() {
            return p;
        }
    }
    PathBuf::from("models").join("clap")
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

/// Starts (in samples) of up to three clips spread over a track: early, middle and late
/// (skipping the very start and end, which are often silence or applause).
pub fn clip_starts(len: usize, clip: usize) -> Vec<usize> {
    if len <= clip {
        return vec![0];
    }
    let room = len - clip;
    let mut v: Vec<usize> = [0.15, 0.45, 0.75]
        .iter()
        .map(|f| (room as f64 * f) as usize)
        .collect();
    v.dedup();
    v
}

/// One clip of exactly `clip` samples: cut, or (if the audio is shorter) repeated and then
/// zero-padded — the same "repeatpad" the model was trained with.
pub fn clip_at(samples: &[f32], start: usize, clip: usize) -> Vec<f32> {
    if samples.is_empty() {
        return vec![0.0; clip];
    }
    if samples.len() >= start + clip {
        return samples[start..start + clip].to_vec();
    }
    let mut out = Vec::with_capacity(clip);
    let repeats = (clip / samples.len()).max(1);
    for _ in 0..repeats {
        out.extend_from_slice(samples);
    }
    out.truncate(clip);
    out.resize(clip, 0.0);
    out
}

/// ONNX Runtime allows one environment per process; every model shares it.
fn environment() -> Result<&'static Environment> {
    static ENV: OnceLock<Environment> = OnceLock::new();
    if let Some(e) = ENV.get() {
        return Ok(e);
    }
    let env = ort::init().with_name("dhunn").build()?;
    Ok(ENV.get_or_init(|| env))
}

fn session(env: &Environment, path: &Path) -> Result<Session> {
    if !path.exists() {
        return Err(EmbedError::Missing(
            path.parent().unwrap_or(path).to_owned(),
        ));
    }
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    // CPU only: `ort` asks ONNX Runtime to pick the "most efficient" device by default,
    // which on macOS is CoreML — and CoreML can't run this model. CPU results are also the
    // ones the export was checked against.
    Ok(Session::builder(env)?
        .with_auto_device(AutoDevicePolicy::PreferCPU)
        .map_err(ort::Error::from)?
        .with_no_environment_execution_providers()
        .map_err(ort::Error::from)?
        .with_execution_providers([ort::ep::CPU::default().build()])
        .map_err(ort::Error::from)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort::Error::from)?
        .with_intra_threads(threads)
        .map_err(ort::Error::from)?
        .commit_from_file(path)?)
}

impl Clap {
    /// Loads both encoders.
    pub fn load(dir: &Path) -> Result<Self> {
        Self::open(dir, true)
    }

    /// Loads only the text encoder (enough for vibe search; faster, half the memory).
    pub fn load_text(dir: &Path) -> Result<Self> {
        Self::open(dir, false)
    }

    fn open(dir: &Path, with_audio: bool) -> Result<Self> {
        let mpath = dir.join("clap.json");
        let manifest: Manifest = serde_json::from_str(
            &std::fs::read_to_string(&mpath).map_err(|_| EmbedError::Missing(dir.to_owned()))?,
        )?;
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;
        // The saved tokenizer pads batches itself; padding and truncation are done here
        // instead, so the attention mask always matches.
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;
        let env = environment()?;
        Ok(Self {
            audio: if with_audio {
                Some(Mutex::new(session(env, &dir.join("audio.onnx"))?))
            } else {
                None
            },
            text: Mutex::new(session(env, &dir.join("text.onnx"))?),
            tokenizer,
            manifest,
        })
    }

    pub fn version(&self) -> u32 {
        self.manifest.version
    }

    fn rows(&self, shape: &[i64], data: &[f32], n: usize) -> Result<Vec<Vec<f32>>> {
        let d = self.manifest.dims;
        if shape != [n as i64, d as i64] {
            return Err(EmbedError::Shape(shape.to_vec()));
        }
        Ok(data.chunks(d).map(<[f32]>::to_vec).collect())
    }

    /// Embeds clips of exactly `clip_samples` mono samples at the model's rate.
    pub fn embed_clips(&self, clips: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if clips.is_empty() {
            return Ok(Vec::new());
        }
        let n = self.manifest.clip_samples;
        let flat: Vec<f32> = clips.iter().flat_map(|c| c.iter().copied()).collect();
        let input = Tensor::from_array(([clips.len(), n], flat))?;
        let audio = self
            .audio
            .as_ref()
            .ok_or_else(|| EmbedError::Tokenizer("CLAP was loaded for text only".into()))?;
        let mut s = audio.lock().expect("audio session poisoned");
        let out = s.run(ort::inputs!["waveform" => input])?;
        let (shape, data) = out["embedding"].try_extract_tensor::<f32>()?;
        self.rows(shape, data, clips.len())
    }

    /// The model's 48 kHz clips for a decoded track (any sample rate).
    pub fn clips_for(&self, audio: &MonoAudio) -> Result<Vec<Vec<f32>>> {
        let sr = self.manifest.sample_rate;
        let samples = dhunn_audio::resample(&audio.samples, audio.sample_rate, sr)?;
        let n = self.manifest.clip_samples;
        Ok(clip_starts(samples.len(), n)
            .into_iter()
            .map(|s| clip_at(&samples, s, n))
            .collect())
    }

    /// One unit vector per track: the mean of its clips' embeddings.
    pub fn embed_track(&self, audio: &MonoAudio) -> Result<Vec<f32>> {
        let embs = self.embed_clips(&self.clips_for(audio)?)?;
        let mut mean = vec![0f32; self.manifest.dims];
        for e in &embs {
            mean.iter_mut().zip(e).for_each(|(m, x)| *m += x);
        }
        normalize(&mut mean);
        Ok(mean)
    }

    /// Token ids and attention mask, padded to the longest text (truncated to the model's
    /// token limit).
    pub fn tokenize(&self, texts: &[&str]) -> Result<(Vec<Vec<i64>>, usize)> {
        let enc = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;
        let max = self.manifest.max_tokens;
        let ids: Vec<Vec<i64>> = enc
            .iter()
            .map(|e| {
                let mut v: Vec<i64> = e.get_ids().iter().map(|&i| i as i64).collect();
                if v.len() > max {
                    // Keep the end-of-text token.
                    let last = *v.last().expect("not empty");
                    v.truncate(max);
                    v[max - 1] = last;
                }
                v
            })
            .collect();
        let len = ids.iter().map(Vec::len).max().unwrap_or(0);
        Ok((ids, len))
    }

    pub fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let (ids, len) = self.tokenize(texts)?;
        // RoBERTa's padding token (masked out, so its value doesn't change the result).
        let pad = self.tokenizer.token_to_id("<pad>").map_or(1, i64::from);
        let mut flat_ids = Vec::with_capacity(ids.len() * len);
        let mut mask = Vec::with_capacity(ids.len() * len);
        for row in &ids {
            flat_ids.extend(row.iter().copied());
            mask.extend(std::iter::repeat_n(1i64, row.len()));
            flat_ids.extend(std::iter::repeat_n(pad, len - row.len()));
            mask.extend(std::iter::repeat_n(0i64, len - row.len()));
        }
        let ids_t = Tensor::from_array(([ids.len(), len], flat_ids))?;
        let mask_t = Tensor::from_array(([ids.len(), len], mask))?;
        let mut s = self.text.lock().expect("text session poisoned");
        let out = s.run(ort::inputs!["input_ids" => ids_t, "attention_mask" => mask_t])?;
        let (shape, data) = out["embedding"].try_extract_tensor::<f32>()?;
        self.rows(shape, data, ids.len())
    }

    pub fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self
            .embed_texts(&[text])?
            .pop()
            .expect("one text, one embedding"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clips_cover_the_track() {
        assert_eq!(clip_starts(100, 200), vec![0]);
        assert_eq!(clip_starts(1200, 200), vec![150, 450, 750]);
        let short = clip_at(&[1.0, 2.0, 3.0], 0, 8);
        assert_eq!(short, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
        assert_eq!(clip_at(&[1.0; 10], 2, 4), vec![1.0; 4]);
        assert_eq!(clip_at(&[], 0, 3), vec![0.0; 3]);
    }

    /// Needs the exported model (`uv run ml/export_clap.py`); skipped otherwise.
    #[test]
    fn matches_the_export() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/clap");
        if !dir.join("clap.json").exists() {
            eprintln!("skipping: no model in {}", dir.display());
            return;
        }
        let clap = Clap::load(&dir).unwrap();
        let check = clap.manifest.check.clone();
        let (ids, _) = clap.tokenize(&[&check.text]).unwrap();
        assert_eq!(ids[0], check.input_ids, "tokenizer differs from the export");
        let e = clap.embed_text(&check.text).unwrap();
        for (a, b) in e.iter().zip(&check.embedding_head) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
        // Batched with a longer text, padding must not change the result.
        let both = clap
            .embed_texts(&[&check.text, "a much longer description of some music"])
            .unwrap();
        let cos: f32 = both[0].iter().zip(&e).map(|(a, b)| a * b).sum();
        assert!(cos > 0.9999, "padding changed the embedding: {cos}");

        let n = clap.manifest.clip_samples;
        let sr = clap.manifest.sample_rate as f64;
        let sine: Vec<f32> = (0..n)
            .map(|i| (0.5 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sr).sin()) as f32)
            .collect();
        let a = clap.embed_clips(&[sine]).unwrap().pop().unwrap();
        for (x, y) in a.iter().zip(&check.sine_440_head) {
            assert!((x - y).abs() < 1e-3, "audio: {x} vs {y}");
        }
        // Text and audio share the space: a pure tone is closer to "a sine tone" than to
        // "heavy metal".
        let t = clap
            .embed_texts(&[
                "a pure sine wave tone",
                "heavy metal with distorted guitars",
            ])
            .unwrap();
        let dot = |u: &[f32], v: &[f32]| u.iter().zip(v).map(|(p, q)| p * q).sum::<f32>();
        assert!(dot(&a, &t[0]) > dot(&a, &t[1]));
    }
}
