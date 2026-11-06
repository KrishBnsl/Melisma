//! SQLite storage for the library: tracks, the files that hold them, artists and features.
//!
//! One database file (WAL mode). Schema changes are numbered migrations tracked with
//! `PRAGMA user_version`. A *track* is a piece of audio identified by a hash of its decoded
//! samples; *track files* are paths on disk that contain it (renamed, re-tagged and duplicate
//! files all point at the same track and share its analysis).

use std::{
    collections::{HashMap, HashSet},
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use dhunn_core::{FeatureVec, TrackId, TrackMeta, normalize_name};
use rusqlite::{Connection, OptionalExtension, Transaction, params};

mod history;
mod sessions;
pub use history::*;
pub use sessions::*;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("database error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("could not create database directory: {0}")]
    Io(#[from] std::io::Error),
    #[error("feature vector layout for {kind} v{version} differs from the one already stored")]
    SchemaMismatch { kind: String, version: u32 },
    #[error("a user named {0:?} already exists")]
    UserExists(String),
    #[error("database was created by a newer dhunn (schema {found}, this build knows {known})")]
    TooNew { found: i64, known: i64 },
}

pub type Result<T> = std::result::Result<T, StoreError>;

/// Ordered migrations; index + 1 is the schema version it produces.
const MIGRATIONS: &[&str] = &[
    r#"
CREATE TABLE tracks (
    id              INTEGER PRIMARY KEY,
    content_hash    TEXT    NOT NULL UNIQUE,
    title           TEXT    NOT NULL,
    album           TEXT,
    year            INTEGER,
    duration_s      REAL    NOT NULL,
    mb_recording_id TEXT,
    added_at        INTEGER NOT NULL
);
CREATE TABLE track_files (
    path     TEXT    PRIMARY KEY,
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    size     INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL
);
CREATE INDEX track_files_by_track ON track_files(track_id);
CREATE TABLE artists (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    name_key     TEXT NOT NULL UNIQUE,
    mb_artist_id TEXT
);
CREATE TABLE track_artists (
    track_id  INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    artist_id INTEGER NOT NULL REFERENCES artists(id),
    role      TEXT    NOT NULL CHECK (role IN ('primary', 'featured')),
    position  INTEGER NOT NULL,
    PRIMARY KEY (track_id, artist_id)
);
CREATE INDEX track_artists_by_artist ON track_artists(artist_id);
CREATE TABLE feature_schemas (
    kind    TEXT    NOT NULL,
    version INTEGER NOT NULL,
    names   TEXT    NOT NULL,
    PRIMARY KEY (kind, version)
);
CREATE TABLE features (
    track_id    INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    kind        TEXT    NOT NULL,
    version     INTEGER NOT NULL,
    vec         BLOB    NOT NULL,
    summary     TEXT,
    computed_at INTEGER NOT NULL,
    PRIMARY KEY (track_id, kind, version)
);
CREATE TABLE scan_failures (
    path      TEXT    PRIMARY KEY,
    size      INTEGER NOT NULL,
    mtime_ns  INTEGER NOT NULL,
    error     TEXT    NOT NULL,
    failed_at INTEGER NOT NULL
);
"#,
    r#"
CREATE TABLE users (
    id              INTEGER PRIMARY KEY,
    name            TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    adventurousness REAL    NOT NULL DEFAULT 0.5 CHECK (adventurousness BETWEEN 0 AND 1),
    share_profile   INTEGER NOT NULL DEFAULT 1,
    created_at      INTEGER NOT NULL
);
CREATE TABLE history_plays (
    id         INTEGER PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ts         INTEGER NOT NULL,
    source     TEXT    NOT NULL,
    raw_artist TEXT    NOT NULL,
    raw_title  TEXT    NOT NULL,
    raw_album  TEXT,
    ms_played  INTEGER NOT NULL,
    skipped    INTEGER,
    reason_end TEXT,
    shuffle    INTEGER,
    source_uri TEXT,
    track_id   INTEGER REFERENCES tracks(id) ON DELETE SET NULL,
    artist_id  INTEGER REFERENCES artists(id) ON DELETE SET NULL,
    match_kind TEXT CHECK (match_kind IN ('exact', 'fuzzy', 'artist', 'none')),
    UNIQUE (user_id, source, ts, raw_artist, raw_title)
);
CREATE INDEX history_by_user_ts ON history_plays(user_id, ts);
CREATE INDEX history_by_track ON history_plays(track_id);
CREATE INDEX history_by_key ON history_plays(raw_artist, raw_title);
"#,
    r#"
CREATE TABLE sessions (
    id         INTEGER PRIMARY KEY,
    user_id    INTEGER REFERENCES users(id) ON DELETE CASCADE,
    started_at INTEGER NOT NULL,
    ended_at   INTEGER,
    mode       TEXT    NOT NULL,
    params     TEXT
);
CREATE TABLE events (
    id             INTEGER PRIMARY KEY,
    session_id     INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id        INTEGER REFERENCES users(id) ON DELETE CASCADE,
    ts_ms          INTEGER NOT NULL,
    track_id       INTEGER REFERENCES tracks(id) ON DELETE SET NULL,
    kind           TEXT    NOT NULL,
    position_ms    INTEGER,
    payload        TEXT,
    schema_version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX events_by_user_ts ON events(user_id, ts_ms);
CREATE INDEX events_by_session ON events(session_id);
"#,
];

/// Size and modification time of a file, used to detect changes without reading it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileStamp {
    pub size: u64,
    pub mtime_ns: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnownFile {
    pub stamp: FileStamp,
    pub track_id: TrackId,
}

/// Everything produced by analyzing one new piece of audio.
#[derive(Debug, Clone)]
pub struct NewAnalysis {
    pub content_hash: String,
    pub meta: TrackMeta,
    pub features: FeatureVec,
    /// Human-readable summary (JSON) kept alongside the vector for explanations.
    pub summary_json: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StoreStats {
    pub tracks: u64,
    pub files: u64,
    pub artists: u64,
    /// `(kind, version, tracks with that feature set)`.
    pub features: Vec<(String, u32, u64)>,
    pub failures: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrackRow {
    pub id: TrackId,
    pub title: String,
    pub artists: Vec<String>,
    pub album: Option<String>,
    pub year: Option<i32>,
    pub duration_s: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    pub path: String,
    pub error: String,
}

/// Dimension names plus one `(track, vector)` row per track.
pub type FeatureMatrix = (Vec<String>, Vec<(TrackId, Vec<f32>)>);

pub struct Store {
    conn: Connection,
}

pub(crate) fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn blob_to_vec(b: &[u8]) -> Vec<f32> {
    let (chunks, _) = b.as_chunks::<4>();
    chunks.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

/// `root` as a prefix that matches only paths inside it.
fn dir_prefix(root: &str) -> String {
    if root.ends_with(std::path::MAIN_SEPARATOR) {
        root.to_owned()
    } else {
        format!("{root}{}", std::path::MAIN_SEPARATOR)
    }
}

impl Store {
    /// Opens (creating if needed) the database at `path` and applies pending migrations.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        Self::init(conn)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        let mut store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&mut self) -> Result<()> {
        let current: i64 = self
            .conn
            .pragma_query_value(None, "user_version", |r| r.get(0))?;
        let known = MIGRATIONS.len() as i64;
        if current > known {
            return Err(StoreError::TooNew {
                found: current,
                known,
            });
        }
        for (i, sql) in MIGRATIONS.iter().enumerate().skip(current as usize) {
            let tx = self.conn.transaction()?;
            tx.execute_batch(sql)?;
            tx.pragma_update(None, "user_version", (i + 1) as i64)?;
            tx.commit()?;
        }
        Ok(())
    }

    pub fn schema_version(&self) -> Result<i64> {
        Ok(self
            .conn
            .pragma_query_value(None, "user_version", |r| r.get(0))?)
    }

    /// Content hashes of tracks that already have features of `kind`/`version`.
    pub fn hashes_with_features(&self, kind: &str, version: u32) -> Result<HashSet<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT t.content_hash FROM tracks t JOIN features f ON f.track_id = t.id
             WHERE f.kind = ?1 AND f.version = ?2",
        )?;
        let rows = stmt.query_map(params![kind, version], |r| r.get(0))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Files recorded under directory `root`.
    pub fn files_under(&self, root: &str) -> Result<HashMap<String, KnownFile>> {
        let prefix = dir_prefix(root);
        let mut stmt = self.conn.prepare(
            "SELECT path, size, mtime_ns, track_id FROM track_files WHERE substr(path, 1, ?2) = ?1",
        )?;
        let rows = stmt.query_map(params![prefix, prefix.chars().count() as i64], |r| {
            Ok((
                r.get::<_, String>(0)?,
                KnownFile {
                    stamp: FileStamp {
                        size: r.get::<_, i64>(1)? as u64,
                        mtime_ns: r.get(2)?,
                    },
                    track_id: TrackId(r.get::<_, i64>(3)? as u64),
                },
            ))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Failed files recorded under directory `root`.
    pub fn failures_under(&self, root: &str) -> Result<HashMap<String, FileStamp>> {
        let prefix = dir_prefix(root);
        let mut stmt = self.conn.prepare(
            "SELECT path, size, mtime_ns FROM scan_failures WHERE substr(path, 1, ?2) = ?1",
        )?;
        let rows = stmt.query_map(params![prefix, prefix.chars().count() as i64], |r| {
            Ok((
                r.get::<_, String>(0)?,
                FileStamp {
                    size: r.get::<_, i64>(1)? as u64,
                    mtime_ns: r.get(2)?,
                },
            ))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    fn ensure_feature_schema(tx: &Transaction<'_>, fv: &FeatureVec) -> Result<()> {
        let names = serde_json::to_string(&fv.names).expect("strings serialize");
        let existing: Option<String> = tx
            .query_row(
                "SELECT names FROM feature_schemas WHERE kind = ?1 AND version = ?2",
                params![fv.kind, fv.version],
                |r| r.get(0),
            )
            .optional()?;
        match existing {
            Some(e) if e != names => Err(StoreError::SchemaMismatch {
                kind: fv.kind.clone(),
                version: fv.version,
            }),
            Some(_) => Ok(()),
            None => {
                tx.execute(
                    "INSERT INTO feature_schemas (kind, version, names) VALUES (?1, ?2, ?3)",
                    params![fv.kind, fv.version, names],
                )?;
                Ok(())
            }
        }
    }

    fn upsert_artist(tx: &Transaction<'_>, name: &str, mb_id: Option<&str>) -> Result<i64> {
        let key = normalize_name(name);
        tx.execute(
            "INSERT INTO artists (name, name_key, mb_artist_id) VALUES (?1, ?2, ?3)
             ON CONFLICT(name_key) DO UPDATE SET mb_artist_id = COALESCE(artists.mb_artist_id, excluded.mb_artist_id)",
            params![name.trim(), key, mb_id],
        )?;
        Ok(
            tx.query_row("SELECT id FROM artists WHERE name_key = ?1", [key], |r| {
                r.get(0)
            })?,
        )
    }

    fn write_artists(tx: &Transaction<'_>, track: i64, m: &TrackMeta) -> Result<()> {
        tx.execute("DELETE FROM track_artists WHERE track_id = ?1", [track])?;
        let mut seen = HashSet::new();
        for (pos, credit) in m.artists.iter().enumerate() {
            let artist = Self::upsert_artist(tx, &credit.name, credit.mb_artist_id.as_deref())?;
            if seen.insert(artist) {
                tx.execute(
                    "INSERT INTO track_artists (track_id, artist_id, role, position) VALUES (?1, ?2, ?3, ?4)",
                    params![track, artist, credit.role.as_str(), pos as i64],
                )?;
            }
        }
        Ok(())
    }

    /// Replaces a track's descriptive metadata (e.g. after its file was re-tagged).
    pub fn update_meta(&mut self, id: TrackId, m: &TrackMeta) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "UPDATE tracks SET title = ?2, album = ?3, year = ?4,
                 mb_recording_id = COALESCE(?5, mb_recording_id) WHERE id = ?1",
            params![id.0 as i64, m.title, m.album, m.year, m.mb_recording_id],
        )?;
        Self::write_artists(&tx, id.0 as i64, m)?;
        tx.commit()?;
        Ok(())
    }

    /// Tracks that already have features of `kind`/`version`.
    pub fn track_ids_with_features(&self, kind: &str, version: u32) -> Result<HashSet<TrackId>> {
        let mut stmt = self
            .conn
            .prepare("SELECT track_id FROM features WHERE kind = ?1 AND version = ?2")?;
        let rows = stmt.query_map(params![kind, version], |r| {
            Ok(TrackId(r.get::<_, i64>(0)? as u64))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    fn put_file(tx: &Transaction<'_>, path: &str, stamp: FileStamp, track: i64) -> Result<()> {
        tx.execute(
            "INSERT INTO track_files (path, track_id, size, mtime_ns) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(path) DO UPDATE SET track_id = excluded.track_id, size = excluded.size,
                                             mtime_ns = excluded.mtime_ns",
            params![path, track, stamp.size as i64, stamp.mtime_ns],
        )?;
        tx.execute("DELETE FROM scan_failures WHERE path = ?1", [path])?;
        Ok(())
    }

    /// Saves a newly analyzed track (metadata, artists, features) and records `path` as
    /// one of its files. Re-saving the same audio updates it in place.
    pub fn save_analysis(
        &mut self,
        path: &str,
        stamp: FileStamp,
        a: &NewAnalysis,
    ) -> Result<TrackId> {
        let tx = self.conn.transaction()?;
        Self::ensure_feature_schema(&tx, &a.features)?;
        let m = &a.meta;
        tx.execute(
            "INSERT INTO tracks (content_hash, title, album, year, duration_s, mb_recording_id, added_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(content_hash) DO UPDATE SET title = excluded.title, album = excluded.album,
                 year = excluded.year, duration_s = excluded.duration_s,
                 mb_recording_id = COALESCE(excluded.mb_recording_id, tracks.mb_recording_id)",
            params![a.content_hash, m.title, m.album, m.year, m.duration_s, m.mb_recording_id, now_unix()],
        )?;
        let track: i64 = tx.query_row(
            "SELECT id FROM tracks WHERE content_hash = ?1",
            [&a.content_hash],
            |r| r.get(0),
        )?;

        Self::write_artists(&tx, track, m)?;

        tx.execute(
            "INSERT OR REPLACE INTO features (track_id, kind, version, vec, summary, computed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                track,
                a.features.kind,
                a.features.version,
                vec_to_blob(&a.features.values),
                a.summary_json,
                now_unix()
            ],
        )?;
        Self::put_file(&tx, path, stamp, track)?;
        tx.commit()?;
        Ok(TrackId(track as u64))
    }

    /// Records `path` as another file holding already-known audio. Returns `None` if no
    /// track has that content hash.
    pub fn link_file(
        &mut self,
        path: &str,
        stamp: FileStamp,
        content_hash: &str,
    ) -> Result<Option<TrackId>> {
        let tx = self.conn.transaction()?;
        let track: Option<i64> = tx
            .query_row(
                "SELECT id FROM tracks WHERE content_hash = ?1",
                [content_hash],
                |r| r.get(0),
            )
            .optional()?;
        let Some(track) = track else {
            return Ok(None);
        };
        Self::put_file(&tx, path, stamp, track)?;
        tx.commit()?;
        Ok(Some(TrackId(track as u64)))
    }

    /// Updates a file's size/mtime when its audio is unchanged (e.g. re-tagged).
    pub fn touch_file(&mut self, path: &str, stamp: FileStamp) -> Result<()> {
        self.conn.execute(
            "UPDATE track_files SET size = ?2, mtime_ns = ?3 WHERE path = ?1",
            params![path, stamp.size as i64, stamp.mtime_ns],
        )?;
        Ok(())
    }

    pub fn record_failure(&mut self, path: &str, stamp: FileStamp, error: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO scan_failures (path, size, mtime_ns, error, failed_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![path, stamp.size as i64, stamp.mtime_ns, error, now_unix()],
        )?;
        Ok(())
    }

    /// Forgets files (and failure records) at these paths. Tracks and their features are
    /// kept: the same audio may live elsewhere, and listening history refers to tracks.
    pub fn remove_paths(&mut self, paths: &[String]) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let mut n = 0;
        for p in paths {
            n += tx.execute("DELETE FROM track_files WHERE path = ?1", [p])?;
            tx.execute("DELETE FROM scan_failures WHERE path = ?1", [p])?;
        }
        tx.commit()?;
        Ok(n)
    }

    pub fn stats(&self) -> Result<StoreStats> {
        let count = |sql: &str| -> Result<u64> {
            Ok(self.conn.query_row(sql, [], |r| r.get::<_, i64>(0))? as u64)
        };
        let mut stmt = self
            .conn
            .prepare("SELECT kind, version, COUNT(*) FROM features GROUP BY kind, version ORDER BY kind, version")?;
        let features = stmt
            .query_map([], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get::<_, i64>(2)? as u64))
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(StoreStats {
            tracks: count("SELECT COUNT(*) FROM tracks")?,
            files: count("SELECT COUNT(*) FROM track_files")?,
            artists: count("SELECT COUNT(*) FROM artists")?,
            features,
            failures: count("SELECT COUNT(*) FROM scan_failures")?,
        })
    }

    pub fn failures(&self, limit: usize) -> Result<Vec<Failure>> {
        let mut stmt = self.conn.prepare(
            "SELECT path, error FROM scan_failures ORDER BY failed_at DESC, path LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit as i64], |r| {
            Ok(Failure {
                path: r.get(0)?,
                error: r.get(1)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn track(&self, id: TrackId) -> Result<Option<TrackRow>> {
        let row = self
            .conn
            .query_row(
                "SELECT title, album, year, duration_s FROM tracks WHERE id = ?1",
                [id.0 as i64],
                |r| Ok((r.get::<_, String>(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .optional()?;
        let Some((title, album, year, duration_s)) = row else {
            return Ok(None);
        };
        let mut stmt = self.conn.prepare(
            "SELECT a.name FROM track_artists ta JOIN artists a ON a.id = ta.artist_id
             WHERE ta.track_id = ?1 ORDER BY ta.position",
        )?;
        let artists = stmt
            .query_map([id.0 as i64], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        Ok(Some(TrackRow {
            id,
            title,
            artists,
            album,
            year,
            duration_s,
        }))
    }

    /// All stored vectors of one kind/version, with their dimension names.
    pub fn feature_matrix(&self, kind: &str, version: u32) -> Result<FeatureMatrix> {
        let names: Option<String> = self
            .conn
            .query_row(
                "SELECT names FROM feature_schemas WHERE kind = ?1 AND version = ?2",
                params![kind, version],
                |r| r.get(0),
            )
            .optional()?;
        let names: Vec<String> = names
            .map(|n| serde_json::from_str(&n).unwrap_or_default())
            .unwrap_or_default();
        let mut stmt = self.conn.prepare(
            "SELECT track_id, vec FROM features WHERE kind = ?1 AND version = ?2 ORDER BY track_id",
        )?;
        let rows = stmt.query_map(params![kind, version], |r| {
            Ok((
                TrackId(r.get::<_, i64>(0)? as u64),
                blob_to_vec(&r.get::<_, Vec<u8>>(1)?),
            ))
        })?;
        Ok((names, rows.collect::<rusqlite::Result<_>>()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dhunn_core::{ArtistCredit, ArtistRole};

    fn analysis(hash: &str, title: &str, artists: &[&str]) -> NewAnalysis {
        NewAnalysis {
            content_hash: hash.into(),
            meta: TrackMeta {
                title: title.into(),
                album: Some("Album".into()),
                year: Some(1970),
                duration_s: 180.0,
                mb_recording_id: None,
                artists: artists
                    .iter()
                    .enumerate()
                    .map(|(i, a)| ArtistCredit {
                        name: (*a).into(),
                        role: if i == 0 {
                            ArtistRole::Primary
                        } else {
                            ArtistRole::Featured
                        },
                        mb_artist_id: None,
                    })
                    .collect(),
            },
            features: FeatureVec {
                kind: "dsp".into(),
                version: 1,
                names: vec!["a".into(), "b".into()],
                values: vec![1.5, -2.0],
            },
            summary_json: Some("{}".into()),
        }
    }

    const STAMP: FileStamp = FileStamp {
        size: 10,
        mtime_ns: 20,
    };

    #[test]
    fn migrations_are_idempotent() {
        let dir = std::env::temp_dir().join(format!("dhunn-store-{}", std::process::id()));
        let db = dir.join("nested/test.db");
        drop(Store::open(&db).unwrap());
        let s = Store::open(&db).unwrap();
        assert_eq!(s.schema_version().unwrap(), MIGRATIONS.len() as i64);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn save_and_read_back() {
        let mut s = Store::open_in_memory().unwrap();
        let id = s
            .save_analysis(
                "/music/a.mp3",
                STAMP,
                &analysis("h1", "Song", &["The Band", "Guest"]),
            )
            .unwrap();
        let t = s.track(id).unwrap().unwrap();
        assert_eq!(t.title, "Song");
        assert_eq!(t.artists, vec!["The Band", "Guest"]);

        let (names, rows) = s.feature_matrix("dsp", 1).unwrap();
        assert_eq!(names, vec!["a", "b"]);
        assert_eq!(rows, vec![(id, vec![1.5, -2.0])]);
        assert!(s.hashes_with_features("dsp", 1).unwrap().contains("h1"));
        assert!(s.hashes_with_features("dsp", 2).unwrap().is_empty());
    }

    #[test]
    fn same_audio_is_one_track() {
        let mut s = Store::open_in_memory().unwrap();
        let a = s
            .save_analysis("/m/a.mp3", STAMP, &analysis("h1", "Song", &["X"]))
            .unwrap();
        let b = s.link_file("/m/copy/a.mp3", STAMP, "h1").unwrap();
        assert_eq!(b, Some(a));
        assert_eq!(s.link_file("/m/b.mp3", STAMP, "unknown").unwrap(), None);
        // Re-saving the same hash with new tags updates in place.
        let c = s
            .save_analysis(
                "/m/a.mp3",
                STAMP,
                &analysis("h1", "Song (Remaster)", &["x "]),
            )
            .unwrap();
        assert_eq!(c, a);
        let st = s.stats().unwrap();
        assert_eq!((st.tracks, st.files, st.artists), (1, 2, 1)); // "X" and "x " are one artist
        assert_eq!(s.track(a).unwrap().unwrap().title, "Song (Remaster)");
    }

    #[test]
    fn files_and_failures_by_directory() {
        let mut s = Store::open_in_memory().unwrap();
        s.save_analysis("/music/a.mp3", STAMP, &analysis("h1", "A", &[]))
            .unwrap();
        s.save_analysis("/music2/b.mp3", STAMP, &analysis("h2", "B", &[]))
            .unwrap();
        s.record_failure("/music/bad.mp3", STAMP, "corrupt")
            .unwrap();
        let files = s.files_under("/music").unwrap();
        assert_eq!(files.keys().collect::<Vec<_>>(), vec!["/music/a.mp3"]); // not /music2
        assert_eq!(s.failures_under("/music/").unwrap().len(), 1);

        // A successful save clears an earlier failure for the same path.
        s.save_analysis("/music/bad.mp3", STAMP, &analysis("h3", "Fixed", &[]))
            .unwrap();
        assert_eq!(s.stats().unwrap().failures, 0);

        assert_eq!(s.remove_paths(&["/music/a.mp3".into()]).unwrap(), 1);
        assert_eq!(s.stats().unwrap().files, 2);
        assert_eq!(s.stats().unwrap().tracks, 3); // tracks survive file removal
    }

    #[test]
    fn retag_updates_meta() {
        let mut s = Store::open_in_memory().unwrap();
        let id = s
            .save_analysis("/a", STAMP, &analysis("h1", "Old", &["A"]))
            .unwrap();
        let new = analysis("h1", "New", &["B", "C"]).meta;
        s.update_meta(id, &new).unwrap();
        let t = s.track(id).unwrap().unwrap();
        assert_eq!(
            (t.title.as_str(), t.artists.clone()),
            ("New", vec!["B".to_string(), "C".to_string()])
        );
        assert!(s.track_ids_with_features("dsp", 1).unwrap().contains(&id));
    }

    #[test]
    fn rejects_changed_vector_layout() {
        let mut s = Store::open_in_memory().unwrap();
        s.save_analysis("/a", STAMP, &analysis("h1", "A", &[]))
            .unwrap();
        let mut b = analysis("h2", "B", &[]);
        b.features.names = vec!["a".into(), "c".into()];
        assert!(matches!(
            s.save_analysis("/b", STAMP, &b),
            Err(StoreError::SchemaMismatch { .. })
        ));
    }
}
