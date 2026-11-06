//! Hand-crafted audio features for dhunn.
//!
//! Per-frame features follow librosa's definitions (validated with `ml/validate_dsp.py`) and
//! are summarized per track into a [`DspSummary`]: spectral shape (brightness, treble
//! extent, noisiness, octave contrast), timbre (MFCC + deltas), harmony (chroma, key),
//! percussion (HPSS balance, onset strength, low-end "thump"), rhythm (tempo, pulse
//! clarity), dynamics and pitch (YIN on the harmonic part).
//!
//! Entry point: [`analyze`].

pub mod chroma;
pub mod hpss;
pub mod mel;
pub mod pitch;
pub mod rhythm;
pub mod spectral;
pub mod stats;
pub mod stft;
mod summary;

pub use chroma::{KEY_NAMES, Mode};
pub use stats::Stat;
pub use summary::*;
