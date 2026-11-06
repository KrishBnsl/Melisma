//! Spotify data exports.
//!
//! * **Extended streaming history** (`Streaming_History_Audio_*.json`, from "Download your
//!   data" → *Extended streaming history*): one object per play with `ts` (UTC, end of play),
//!   `ms_played`, track/artist/album names, `reason_start`/`reason_end`, `shuffle`, `skipped`.
//! * **Basic account data** (`StreamingHistory_music_*.json` / `StreamingHistory*.json`):
//!   `endTime`, `artistName`, `trackName`, `msPlayed` only — no skip information.
//!
//! Podcasts, audiobooks and video entries are ignored. Network fields such as `ip_addr` are
//! never read or stored.

use std::path::{Path, PathBuf};

use dhunn_store::PlayRecord;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{ImportBatch, ImportError, time::parse_timestamp};

pub const SOURCE: &str = "spotify";

/// One entry of the extended streaming history (the fields dhunn uses, plus the ones
/// needed to write realistic sample files).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExtendedRow {
    pub ts: String,
    #[serde(default)]
    pub platform: Option<String>,
    pub ms_played: i64,
    #[serde(default)]
    pub conn_country: Option<String>,
    #[serde(default)]
    pub master_metadata_track_name: Option<String>,
    #[serde(default)]
    pub master_metadata_album_artist_name: Option<String>,
    #[serde(default)]
    pub master_metadata_album_album_name: Option<String>,
    #[serde(default)]
    pub spotify_track_uri: Option<String>,
    #[serde(default)]
    pub episode_name: Option<String>,
    #[serde(default)]
    pub episode_show_name: Option<String>,
    #[serde(default)]
    pub spotify_episode_uri: Option<String>,
    #[serde(default)]
    pub reason_start: Option<String>,
    #[serde(default)]
    pub reason_end: Option<String>,
    #[serde(default)]
    pub shuffle: Option<bool>,
    #[serde(default)]
    pub skipped: Option<bool>,
    #[serde(default)]
    pub offline: Option<bool>,
    #[serde(default)]
    pub incognito_mode: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BasicRow {
    end_time: String,
    artist_name: String,
    track_name: String,
    ms_played: i64,
}

fn non_empty(s: Option<String>) -> Option<String> {
    s.map(|s| s.trim().to_owned()).filter(|s| !s.is_empty())
}

fn is_history_file(p: &Path) -> bool {
    let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
    name.ends_with(".json")
        && (name.starts_with("Streaming_History_Audio") || name.starts_with("StreamingHistory"))
}

fn collect_files(path: &Path) -> Result<Vec<PathBuf>, ImportError> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut files = Vec::new();
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).map_err(|source| ImportError::Io {
            path: dir.clone(),
            source,
        })?;
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if is_history_file(&p) {
                files.push(p);
            }
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(ImportError::NoFiles {
            path: path.to_path_buf(),
            expected: "Streaming_History_Audio_*.json or StreamingHistory*.json (unzip the export first)",
        });
    }
    Ok(files)
}

/// Converts one extended-history entry; `None` for non-music entries.
pub fn extended_to_play(row: ExtendedRow) -> Option<Result<PlayRecord, String>> {
    let title = non_empty(row.master_metadata_track_name)?;
    let artist = non_empty(row.master_metadata_album_artist_name)?;
    let Some(ts) = parse_timestamp(&row.ts) else {
        return Some(Err(format!("bad timestamp {:?}", row.ts)));
    };
    Some(Ok(PlayRecord {
        ts,
        source: SOURCE,
        raw_artist: artist,
        raw_title: title,
        raw_album: non_empty(row.master_metadata_album_album_name),
        ms_played: row.ms_played.max(0),
        skipped: row.skipped,
        reason_end: non_empty(row.reason_end),
        shuffle: row.shuffle,
        source_uri: non_empty(row.spotify_track_uri),
    }))
}

fn basic_to_play(row: BasicRow) -> Option<Result<PlayRecord, String>> {
    let title = non_empty(Some(row.track_name))?;
    let artist = non_empty(Some(row.artist_name))?;
    if row.ms_played <= 0 {
        return None;
    }
    let Some(ts) = parse_timestamp(&row.end_time) else {
        return Some(Err(format!("bad timestamp {:?}", row.end_time)));
    };
    Some(Ok(PlayRecord {
        ts,
        source: SOURCE,
        raw_artist: artist,
        raw_title: title,
        raw_album: None,
        ms_played: row.ms_played,
        skipped: None,
        reason_end: None,
        shuffle: None,
        source_uri: None,
    }))
}

/// Parses a Spotify export: one JSON file, or a folder (searched recursively).
pub fn parse(path: &Path) -> Result<ImportBatch, ImportError> {
    let mut batch = ImportBatch::default();
    for file in collect_files(path)? {
        let text = std::fs::read_to_string(&file).map_err(|source| ImportError::Io {
            path: file.clone(),
            source,
        })?;
        let rows: Vec<Value> = serde_json::from_str(&text).map_err(|source| ImportError::Json {
            path: file.clone(),
            source,
        })?;
        batch.files += 1;
        for v in rows {
            let converted = if v.get("ts").is_some() {
                serde_json::from_value::<ExtendedRow>(v)
                    .map_err(|e| e.to_string())
                    .map(extended_to_play)
            } else if v.get("endTime").is_some() {
                serde_json::from_value::<BasicRow>(v)
                    .map_err(|e| e.to_string())
                    .map(basic_to_play)
            } else {
                Err("unrecognized entry".to_owned())
            };
            match converted {
                Ok(Some(Ok(play))) => batch.plays.push(play),
                Ok(None) => batch.skipped_non_music += 1,
                Ok(Some(Err(e))) | Err(e) => {
                    batch.invalid += 1;
                    if batch.first_error.is_none() {
                        batch.first_error = Some(format!("{}: {e}", file.display()));
                    }
                }
            }
        }
    }
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    #[test]
    fn extended_history() {
        let b = parse(&fixtures().join("Streaming_History_Audio_2025-2026_0.json")).unwrap();
        assert_eq!((b.plays.len(), b.skipped_non_music, b.invalid), (2, 1, 0));
        let p = &b.plays[0];
        assert_eq!(p.raw_artist, "The Beatles");
        assert_eq!(p.raw_title, "Let It Be - Remastered 2009");
        assert_eq!(p.ts, 1_772_388_252);
        assert_eq!((p.ms_played, p.skipped), (214_532, Some(false)));
        assert_eq!(b.plays[1].skipped, Some(true));
        assert_eq!(b.plays[1].reason_end.as_deref(), Some("fwdbtn"));
    }

    #[test]
    fn basic_history() {
        let b = parse(&fixtures().join("StreamingHistory_music_0.json")).unwrap();
        // The zero-length play is dropped.
        assert_eq!((b.plays.len(), b.skipped_non_music), (2, 1));
        assert_eq!(b.plays[0].skipped, None);
    }

    #[test]
    fn folder_reads_all_spotify_files_only() {
        let b = parse(&fixtures()).unwrap();
        assert_eq!(b.files, 2);
        assert_eq!(b.plays.len(), 4);
    }

    #[test]
    fn empty_folder_explains_what_it_wants() {
        let dir = tempfile::tempdir().unwrap();
        let err = parse(dir.path()).unwrap_err().to_string();
        assert!(err.contains("Streaming_History_Audio"), "{err}");
    }
}
