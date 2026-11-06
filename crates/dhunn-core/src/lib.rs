//! Shared types used across the dhunn crates.
//!
//! Kept dependency-light (serde only) so every other crate — including the future
//! server and Android/desktop bindings — can depend on it cheaply.

use serde::{Deserialize, Serialize};

/// Sample rate (Hz) that all DSP features are computed at.
pub const ANALYSIS_SAMPLE_RATE: u32 = 22_050;

/// Stable identifier for a track in the library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TrackId(pub u64);

/// A fixed-order, versioned feature vector. `names[i]` describes `values[i]`.
///
/// Vectors are only comparable when `kind` and `version` match.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureVec {
    /// Which extractor produced this vector, e.g. `"dsp"`.
    pub kind: String,
    /// Schema version of the extractor.
    pub version: u32,
    pub names: Vec<String>,
    pub values: Vec<f32>,
}

impl FeatureVec {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Looks up a dimension by name.
    pub fn get(&self, name: &str) -> Option<f32> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.values[i])
    }
}
