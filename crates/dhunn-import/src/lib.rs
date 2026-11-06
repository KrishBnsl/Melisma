//! Listening-history import: parse streaming-service data exports, store the plays for a
//! user, and match them to library tracks.

use std::path::{Path, PathBuf};

use dhunn_core::UserId;
use dhunn_store::{PlayRecord, Store, StoreError};

pub mod apple;
pub mod matcher;
pub mod spotify;
pub mod time;

pub use matcher::{MatchReport, match_history};

#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path} is not valid JSON: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("{path} is not readable CSV: {source}")]
    Csv { path: PathBuf, source: csv::Error },
    #[error("no history files found in {path} (expected {expected})")]
    NoFiles {
        path: PathBuf,
        expected: &'static str,
    },
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// Plays parsed from an export, plus what was left out and why.
#[derive(Debug, Clone, Default)]
pub struct ImportBatch {
    pub files: usize,
    pub plays: Vec<PlayRecord>,
    /// Podcasts, audiobooks, zero-length plays, non-PLAY_END events.
    pub skipped_non_music: usize,
    /// Entries that looked like plays but couldn't be read.
    pub invalid: usize,
    pub first_error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Service {
    Spotify,
    Apple,
}

impl Service {
    pub fn parse(self, path: &Path) -> Result<ImportBatch, ImportError> {
        match self {
            Service::Spotify => spotify::parse(path),
            Service::Apple => apple::parse(path),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ImportReport {
    pub files: usize,
    pub new_plays: usize,
    pub already_imported: usize,
    pub skipped_non_music: usize,
    pub invalid: usize,
    pub first_error: Option<String>,
    pub matching: MatchReport,
}

/// Parses an export, stores its plays for `user` (re-importing is harmless) and matches
/// them against the library.
pub fn import(
    store: &mut Store,
    user: UserId,
    service: Service,
    path: &Path,
) -> Result<ImportReport, ImportError> {
    let batch = service.parse(path)?;
    let (new_plays, already_imported) = store.insert_plays(user, &batch.plays)?;
    let matching = match_history(store)?;
    Ok(ImportReport {
        files: batch.files,
        new_plays,
        already_imported,
        skipped_non_music: batch.skipped_non_music,
        invalid: batch.invalid,
        first_error: batch.first_error,
        matching,
    })
}
