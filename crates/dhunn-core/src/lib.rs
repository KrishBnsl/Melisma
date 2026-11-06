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

/// Identifier of a dhunn user (a person in the friend group).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct UserId(pub u64);

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

/// How an artist is credited on a track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ArtistRole {
    Primary,
    Featured,
}

impl ArtistRole {
    pub fn as_str(self) -> &'static str {
        match self {
            ArtistRole::Primary => "primary",
            ArtistRole::Featured => "featured",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtistCredit {
    pub name: String,
    pub role: ArtistRole,
    pub mb_artist_id: Option<String>,
}

/// Descriptive metadata for a track (from tags, later MusicBrainz).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackMeta {
    pub title: String,
    pub album: Option<String>,
    pub year: Option<i32>,
    pub duration_s: f64,
    pub mb_recording_id: Option<String>,
    pub artists: Vec<ArtistCredit>,
}

/// Normalizes a name for matching: case-folded, trimmed, inner whitespace collapsed.
pub fn normalize_name(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_names() {
        assert_eq!(normalize_name("  The   Beatles "), "the beatles");
    }
}
