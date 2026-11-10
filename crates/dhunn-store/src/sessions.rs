//! Listening sessions and the events inside them (plays, skips, likes, steering), plus the
//! track lookups a player needs.

use std::collections::{HashMap, HashSet};

use dhunn_core::{TrackId, UserId, normalize_name};
use rusqlite::{OptionalExtension, params};

use crate::{Result, Store, now_unix};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(pub u64);

/// Everything a listener can do during a session. Stored by name, so new kinds can be added
/// without a migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Play,
    Complete,
    Skip,
    Like,
    Dislike,
    Further,
    /// "Surprise me" pressed mid-session.
    Surprise,
    Back,
    Pause,
    Resume,
    Quit,
    /// Jumped within the track (position = where to).
    Seek,
    /// The file could not be played.
    Error,
    /// The listener chose a different track themselves (position = where the current
    /// one was left). Not a skip: it says nothing about the track that was playing.
    Choose,
    /// A like taken back.
    Unlike,
}

impl EventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EventKind::Play => "play",
            EventKind::Complete => "complete",
            EventKind::Skip => "skip",
            EventKind::Like => "like",
            EventKind::Dislike => "dislike",
            EventKind::Further => "further",
            EventKind::Surprise => "surprise",
            EventKind::Back => "back",
            EventKind::Pause => "pause",
            EventKind::Resume => "resume",
            EventKind::Quit => "quit",
            EventKind::Seek => "seek",
            EventKind::Error => "error",
            EventKind::Choose => "choose",
            EventKind::Unlike => "unlike",
        }
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// A library match for a free-text query.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    pub track: TrackId,
    pub title: String,
    pub artist: Option<String>,
}

/// One library track with what a browser shows.
#[derive(Debug, Clone, PartialEq)]
pub struct BrowseRow {
    pub track: TrackId,
    pub title: String,
    /// The first credited artist.
    pub artist: Option<String>,
    pub album: Option<String>,
    pub year: Option<i32>,
    pub duration_s: f64,
    pub path: Option<String>,
}

impl Store {
    /// Every track with its first artist, album and newest file, for browsing.
    pub fn browse_rows(&self) -> Result<Vec<BrowseRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT t.id, t.title, t.album, t.year, t.duration_s,
                    (SELECT a.name FROM track_artists ta JOIN artists a ON a.id = ta.artist_id
                     WHERE ta.track_id = t.id ORDER BY ta.position LIMIT 1),
                    (SELECT f.path FROM track_files f WHERE f.track_id = t.id
                     ORDER BY f.mtime_ns DESC LIMIT 1)
             FROM tracks t ORDER BY t.id",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(BrowseRow {
                track: TrackId(r.get::<_, i64>(0)? as u64),
                title: r.get(1)?,
                album: r.get(2)?,
                year: r.get(3)?,
                duration_s: r.get(4)?,
                artist: r.get(5)?,
                path: r.get(6)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn start_session(
        &mut self,
        user: Option<UserId>,
        mode: &str,
        params_json: &str,
    ) -> Result<SessionId> {
        self.conn.execute(
            "INSERT INTO sessions (user_id, started_at, mode, params) VALUES (?1, ?2, ?3, ?4)",
            params![user.map(|u| u.0 as i64), now_unix(), mode, params_json],
        )?;
        Ok(SessionId(self.conn.last_insert_rowid() as u64))
    }

    pub fn end_session(&mut self, session: SessionId) -> Result<()> {
        self.conn.execute(
            "UPDATE sessions SET ended_at = ?2 WHERE id = ?1",
            params![session.0 as i64, now_unix()],
        )?;
        Ok(())
    }

    /// Appends one event. Written immediately, so nothing is lost if the player is killed.
    pub fn log_event(
        &mut self,
        session: SessionId,
        user: Option<UserId>,
        track: Option<TrackId>,
        kind: EventKind,
        position_ms: Option<i64>,
        payload: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO events (session_id, user_id, ts_ms, track_id, kind, position_ms, payload)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session.0 as i64,
                user.map(|u| u.0 as i64),
                now_ms(),
                track.map(|t| t.0 as i64),
                kind.as_str(),
                position_ms,
                payload
            ],
        )?;
        Ok(())
    }

    /// `(kind, track)` events of a session in order (mostly for tests and debugging).
    pub fn session_events(&self, session: SessionId) -> Result<Vec<(String, Option<TrackId>)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT kind, track_id FROM events WHERE session_id = ?1 ORDER BY id")?;
        let rows = stmt.query_map([session.0 as i64], |r| {
            Ok((
                r.get(0)?,
                r.get::<_, Option<i64>>(1)?.map(|t| TrackId(t as u64)),
            ))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Tracks a user started playing in dhunn since `since_ms` (to avoid repeating them).
    pub fn recently_played(&self, user: UserId, since_ms: i64) -> Result<HashSet<TrackId>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT track_id FROM events WHERE user_id = ?1 AND kind = 'play' AND ts_ms >= ?2
             AND track_id IS NOT NULL",
        )?;
        let rows = stmt.query_map(params![user.0 as i64, since_ms], |r| {
            Ok(TrackId(r.get::<_, i64>(0)? as u64))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// A path on disk that holds this track (most recently seen file first).
    pub fn track_path(&self, track: TrackId) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row(
                "SELECT path FROM track_files WHERE track_id = ?1 ORDER BY mtime_ns DESC LIMIT 1",
                [track.0 as i64],
                |r| r.get(0),
            )
            .optional()?)
    }

    /// A portable identifier for the decoded audio, independent of paths and tags.
    pub fn recording_hash(&self, track: TrackId) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row(
                "SELECT content_hash FROM tracks WHERE id = ?1",
                [track.0 as i64],
                |r| r.get(0),
            )
            .optional()?)
    }

    pub fn recording_paths(&self, hash: &str) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.path FROM track_files f JOIN tracks t ON f.track_id = t.id
             WHERE t.content_hash = ?1 ORDER BY f.mtime_ns DESC, f.path LIMIT 32",
        )?;
        let rows = stmt.query_map([hash], |r| r.get(0))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// The track a file belongs to, by its path as analyzed.
    pub fn track_by_path(&self, path: &str) -> Result<Option<TrackId>> {
        Ok(self
            .conn
            .query_row(
                "SELECT track_id FROM track_files WHERE path = ?1",
                [path],
                |r| r.get::<_, i64>(0),
            )
            .optional()?
            .map(|id| TrackId(id as u64)))
    }

    /// Tracks a user likes now: liked, and not un-liked since.
    pub fn liked_tracks(&self, user: UserId) -> Result<HashSet<TrackId>> {
        let mut stmt = self.conn.prepare(
            "SELECT track_id, kind FROM events
             WHERE user_id = ?1 AND kind IN ('like', 'unlike') AND track_id IS NOT NULL
             ORDER BY id",
        )?;
        let rows = stmt.query_map([user.0 as i64], |r| {
            Ok((TrackId(r.get::<_, i64>(0)? as u64), r.get::<_, String>(1)?))
        })?;
        let mut liked = HashSet::new();
        for row in rows {
            let (t, kind) = row?;
            if kind == "like" {
                liked.insert(t);
            } else {
                liked.remove(&t);
            }
        }
        Ok(liked)
    }

    /// Every analyzed track's newest file path.
    pub fn track_paths(&self) -> Result<Vec<(TrackId, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT track_id, path FROM track_files f
             WHERE mtime_ns = (SELECT MAX(mtime_ns) FROM track_files g WHERE g.track_id = f.track_id)",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((TrackId(r.get::<_, i64>(0)? as u64), r.get::<_, String>(1)?))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// The stored human-readable feature summary (JSON) for a track.
    pub fn track_summary(
        &self,
        track: TrackId,
        kind: &str,
        version: u32,
    ) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row(
                "SELECT summary FROM features WHERE track_id = ?1 AND kind = ?2 AND version = ?3",
                params![track.0 as i64, kind, version],
                |r| r.get(0),
            )
            .optional()?
            .flatten())
    }

    /// Primary artist id of every track that has one.
    pub fn primary_artists(&self) -> Result<HashMap<TrackId, i64>> {
        let mut stmt = self
            .conn
            .prepare("SELECT track_id, artist_id FROM track_artists WHERE position = 0")?;
        let rows = stmt.query_map([], |r| Ok((TrackId(r.get::<_, i64>(0)? as u64), r.get(1)?)))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Tracks whose title or artist contains every word of `query` (case-insensitive),
    /// best matches first: exact title, then title prefix, then the rest.
    pub fn search_tracks(&self, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
        let words: Vec<String> = normalize_name(query)
            .split(' ')
            .filter(|w| !w.is_empty())
            .map(str::to_owned)
            .collect();
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let mut stmt = self.conn.prepare(
            "SELECT t.id, t.title, (SELECT a.name FROM track_artists ta JOIN artists a ON a.id = ta.artist_id
                                    WHERE ta.track_id = t.id ORDER BY ta.position LIMIT 1)
             FROM tracks t",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(SearchHit {
                track: TrackId(r.get::<_, i64>(0)? as u64),
                title: r.get(1)?,
                artist: r.get(2)?,
            })
        })?;
        let q = words.join(" ");
        let mut hits: Vec<(u8, SearchHit)> = Vec::new();
        for hit in rows {
            let hit = hit?;
            let title = normalize_name(&hit.title);
            let hay = format!(
                "{title} {}",
                hit.artist
                    .as_deref()
                    .map(normalize_name)
                    .unwrap_or_default()
            );
            if words.iter().all(|w| hay.contains(w.as_str())) {
                let rank = if title == q {
                    0
                } else if title.starts_with(&q) {
                    1
                } else {
                    2
                };
                hits.push((rank, hit));
            }
        }
        hits.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then(a.1.title.len().cmp(&b.1.title.len()))
                .then(a.1.track.cmp(&b.1.track))
        });
        Ok(hits.into_iter().take(limit).map(|h| h.1).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FileStamp, NewAnalysis};
    use dhunn_core::{ArtistCredit, ArtistRole, FeatureVec, TrackMeta};

    fn add(s: &mut Store, hash: &str, title: &str, artist: &str) -> TrackId {
        let a = NewAnalysis {
            content_hash: hash.into(),
            meta: TrackMeta {
                title: title.into(),
                album: None,
                year: None,
                duration_s: 100.0,
                mb_recording_id: None,
                artists: vec![ArtistCredit {
                    name: artist.into(),
                    role: ArtistRole::Primary,
                    mb_artist_id: None,
                }],
            },
            features: FeatureVec {
                kind: "dsp".into(),
                version: 1,
                names: vec!["x".into()],
                values: vec![0.0],
            },
            summary_json: Some(r#"{"rhythm":{"tempo_bpm":120.0}}"#.into()),
        };
        s.save_analysis(
            &format!("/m/{hash}.mp3"),
            FileStamp {
                size: 1,
                mtime_ns: 1,
            },
            &a,
        )
        .unwrap()
    }

    #[test]
    fn sessions_log_events() {
        let mut s = Store::open_in_memory().unwrap();
        let t = add(&mut s, "h1", "Let It Be", "The Beatles");
        let u = s.add_user("me", 0.5).unwrap().id;
        let sess = s.start_session(Some(u), "drift", "{}").unwrap();
        s.log_event(sess, Some(u), Some(t), EventKind::Play, Some(0), None)
            .unwrap();
        s.log_event(sess, Some(u), Some(t), EventKind::Skip, Some(4200), None)
            .unwrap();
        s.log_event(sess, Some(u), Some(t), EventKind::Like, None, None)
            .unwrap();
        assert!(s.liked_tracks(u).unwrap().contains(&t));
        s.log_event(sess, Some(u), Some(t), EventKind::Unlike, None, None)
            .unwrap();
        assert!(s.liked_tracks(u).unwrap().is_empty());
        assert_eq!(s.track_paths().unwrap(), vec![(t, "/m/h1.mp3".to_owned())]);
        s.end_session(sess).unwrap();
        assert_eq!(
            s.session_events(sess).unwrap()[..2],
            vec![("play".to_string(), Some(t)), ("skip".to_string(), Some(t))]
        );
        assert!(s.recently_played(u, 0).unwrap().contains(&t));
        assert!(s.recently_played(u, i64::MAX).unwrap().is_empty());
    }

    #[test]
    fn lookups_for_the_player() {
        let mut s = Store::open_in_memory().unwrap();
        let a = add(&mut s, "h1", "Let It Be", "The Beatles");
        let b = add(&mut s, "h2", "Let It Bleed", "The Rolling Stones");
        add(&mut s, "h3", "Paranoid", "Black Sabbath");
        assert_eq!(s.track_path(a).unwrap().as_deref(), Some("/m/h1.mp3"));
        assert_eq!(s.track_by_path("/m/h1.mp3").unwrap(), Some(a));
        assert_eq!(s.track_by_path("/m/nope.mp3").unwrap(), None);
        assert!(
            s.track_summary(a, "dsp", 1)
                .unwrap()
                .unwrap()
                .contains("tempo_bpm")
        );
        assert_eq!(s.primary_artists().unwrap().len(), 3);

        assert_eq!(s.search_tracks("let it be", 5).unwrap()[0].track, a); // exact title first
        let hits = s.search_tracks("let it", 5).unwrap();
        assert_eq!(hits.iter().map(|h| h.track).collect::<Vec<_>>(), vec![a, b]); // shorter title first
        assert_eq!(s.search_tracks("stones bleed", 5).unwrap()[0].track, b); // words in any field
        assert!(s.search_tracks("   ", 5).unwrap().is_empty());

        let rows = s.browse_rows().unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[1].artist.as_deref(), Some("The Rolling Stones"));
        assert_eq!(rows[1].path.as_deref(), Some("/m/h2.mp3"));
    }
}
