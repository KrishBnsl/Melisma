//! Apple Music data exports (privacy.apple.com → *Apple Media Services information*).
//!
//! Two CSV layouts are understood, by header name, so extra or reordered columns are fine:
//! * **Play Activity** (`Apple Music Play Activity.csv`): one row per playback event; only
//!   `PLAY_END` rows are kept. Song/artist, start time, play and media duration, end reason.
//! * **Daily Tracks** (`Apple Music - Play History Daily Tracks.csv`): per-day aggregates with
//!   `Track Description` = "Artist - Title", play/skip counts and total duration; each row is
//!   expanded into its individual plays.
//!
//! Built from the documented column names and not yet checked against a real export, so the
//! importer is deliberately lenient and reports what it skipped.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use dhunn_store::PlayRecord;

use crate::{ImportBatch, ImportError, time::parse_timestamp};

pub const SOURCE: &str = "apple";

const FILE_NAMES: &[&str] = &[
    "Apple Music Play Activity.csv",
    "Apple Music - Play History Daily Tracks.csv",
];

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
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if p.is_dir() {
                stack.push(p);
            } else if FILE_NAMES.contains(&name) {
                files.push(p);
            }
        }
    }
    files.sort();
    if files.is_empty() {
        return Err(ImportError::NoFiles {
            path: path.to_path_buf(),
            expected: "Apple Music Play Activity.csv",
        });
    }
    Ok(files)
}

struct Columns(HashMap<String, usize>);

impl Columns {
    fn get<'r>(&self, row: &'r csv::StringRecord, names: &[&str]) -> Option<&'r str> {
        names
            .iter()
            .find_map(|n| self.0.get(&n.to_ascii_lowercase()))
            .and_then(|&i| row.get(i))
            .map(str::trim)
            .filter(|s| !s.is_empty())
    }

    fn has(&self, names: &[&str]) -> bool {
        names
            .iter()
            .any(|n| self.0.contains_key(&n.to_ascii_lowercase()))
    }
}

const TITLE: &[&str] = &["Song Name", "Track Name", "Title", "Content Name"];
const ARTIST: &[&str] = &["Artist Name", "Artist", "Container Artist Name"];
const DESCRIPTION: &[&str] = &["Track Description"];
const START: &[&str] = &[
    "Event Start Timestamp",
    "Event Timestamp",
    "Play Date Utc",
    "Date Played",
];
const PLAY_MS: &[&str] = &["Play Duration Milliseconds", "Play Duration Ms"];
const MEDIA_MS: &[&str] = &["Media Duration In Milliseconds"];
const END_REASON: &[&str] = &["End Reason Type"];
const EVENT_TYPE: &[&str] = &["Event Type"];
const ALBUM: &[&str] = &["Album Name", "Container Album Name"];
const PLAY_COUNT: &[&str] = &["Play Count"];
const SKIP_COUNT: &[&str] = &["Skip Count"];
const HOURS: &[&str] = &["Hours"];

fn int(s: Option<&str>) -> Option<i64> {
    s.and_then(|v| v.parse::<f64>().ok()).map(|v| v as i64)
}

fn parse_file(file: &Path, batch: &mut ImportBatch) -> Result<(), ImportError> {
    let csv_err = |source| ImportError::Csv {
        path: file.to_path_buf(),
        source,
    };
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(file)
        .map_err(csv_err)?;
    let headers = rdr.headers().map_err(csv_err)?.clone();
    let cols = Columns(
        headers
            .iter()
            .enumerate()
            .map(|(i, h)| (h.trim().to_ascii_lowercase(), i))
            .collect(),
    );
    let daily = cols.has(DESCRIPTION) && !cols.has(TITLE);
    batch.files += 1;

    for (line, row) in rdr.records().enumerate() {
        let row = row.map_err(csv_err)?;
        let mut invalid = |why: &str| {
            batch.invalid += 1;
            if batch.first_error.is_none() {
                batch.first_error = Some(format!("{} row {}: {why}", file.display(), line + 2));
            }
        };
        if let Some(ev) = cols.get(&row, EVENT_TYPE)
            && ev != "PLAY_END"
        {
            batch.skipped_non_music += 1;
            continue;
        }
        let (artist, title) = if daily {
            match cols
                .get(&row, DESCRIPTION)
                .and_then(|d| d.split_once(" - "))
            {
                Some((a, t)) => (a.trim().to_owned(), t.trim().to_owned()),
                None => {
                    batch.skipped_non_music += 1;
                    continue;
                }
            }
        } else {
            match (cols.get(&row, ARTIST), cols.get(&row, TITLE)) {
                (Some(a), Some(t)) => (a.to_owned(), t.to_owned()),
                _ => {
                    batch.skipped_non_music += 1;
                    continue;
                }
            }
        };
        let Some(mut ts) = cols.get(&row, START).and_then(parse_timestamp) else {
            invalid("missing or unreadable timestamp");
            continue;
        };
        let ms = int(cols.get(&row, PLAY_MS)).unwrap_or(0).max(0);
        if ms == 0 {
            batch.skipped_non_music += 1;
            continue;
        }
        let album = cols.get(&row, ALBUM).map(str::to_owned);

        if daily {
            // "Hours" lists the UTC hours played that day, e.g. "10, 11"; use the first.
            if let Some(h) = cols
                .get(&row, HOURS)
                .and_then(|h| h.split(',').next())
                .and_then(|h| h.trim().parse::<i64>().ok())
            {
                ts += h.clamp(0, 23) * 3600;
            }
            let plays = int(cols.get(&row, PLAY_COUNT)).unwrap_or(1).max(1);
            let skips = int(cols.get(&row, SKIP_COUNT)).unwrap_or(0).clamp(0, plays);
            for i in 0..plays {
                batch.plays.push(PlayRecord {
                    ts: ts + i * 60, // distinct timestamps for the same track on one day
                    source: SOURCE,
                    raw_artist: artist.clone(),
                    raw_title: title.clone(),
                    raw_album: album.clone(),
                    ms_played: ms / plays,
                    skipped: Some(i < skips),
                    reason_end: None,
                    shuffle: None,
                    source_uri: None,
                });
            }
        } else {
            let reason = cols.get(&row, END_REASON).map(str::to_owned);
            let media_ms = int(cols.get(&row, MEDIA_MS));
            let skipped = match reason.as_deref() {
                Some(r) if r.contains("SKIP") => Some(true),
                Some("NATURAL_END_OF_TRACK") => Some(false),
                _ => media_ms.map(|m| m > 0 && ms < m / 2),
            };
            batch.plays.push(PlayRecord {
                ts,
                source: SOURCE,
                raw_artist: artist,
                raw_title: title,
                raw_album: album,
                ms_played: ms,
                skipped,
                reason_end: reason,
                shuffle: None,
                source_uri: None,
            });
        }
    }
    Ok(())
}

/// Parses an Apple Music export: one CSV, or a folder searched for the known file names.
pub fn parse(path: &Path) -> Result<ImportBatch, ImportError> {
    let mut batch = ImportBatch::default();
    for file in collect_files(path)? {
        parse_file(&file, &mut batch)?;
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
    fn play_activity() {
        let b = parse(&fixtures().join("Apple Music Play Activity.csv")).unwrap();
        assert_eq!((b.plays.len(), b.skipped_non_music, b.invalid), (2, 1, 0));
        assert_eq!(b.plays[0].raw_title, "Let It Be");
        assert_eq!(b.plays[0].skipped, Some(false));
        assert_eq!(b.plays[1].skipped, Some(true));
        assert_eq!(
            b.plays[0].ts,
            parse_timestamp("2026-03-03T10:00:00Z").unwrap()
        );
    }

    #[test]
    fn daily_tracks_expand_into_plays() {
        let b = parse(&fixtures().join("Apple Music - Play History Daily Tracks.csv")).unwrap();
        assert_eq!(b.plays.len(), 3); // 2 plays of Dylan + 1 of Nirvana
        assert_eq!(b.plays[0].raw_artist, "Bob Dylan");
        assert_eq!(b.plays[0].raw_title, "Knockin' On Heaven's Door");
        assert_eq!(
            b.plays[0].ts,
            parse_timestamp("2026-03-04T10:00:00Z").unwrap()
        );
        assert_eq!(b.plays[0].ms_played, 175_000);
        assert_eq!(b.plays[2].skipped, Some(true));
    }

    #[test]
    fn folder_finds_both_files() {
        assert_eq!(parse(&fixtures()).unwrap().files, 2);
    }
}
