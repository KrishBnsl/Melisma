//! Users and imported listening history.

use std::collections::HashMap;

use dhunn_core::{TrackId, UserId};
use rusqlite::{OptionalExtension, params};

use crate::{Result, Store, StoreError, now_unix};

#[derive(Debug, Clone, PartialEq)]
pub struct User {
    pub id: UserId,
    pub name: String,
    /// 0 = stay close to known taste, 1 = explore boldly.
    pub adventurousness: f32,
    /// Whether friends' recommendations may use this user's taste.
    pub share_profile: bool,
}

/// One play as imported from a streaming service's data export.
#[derive(Debug, Clone, PartialEq)]
pub struct PlayRecord {
    /// Unix seconds (UTC) when the play ended (Spotify) or started (Apple).
    pub ts: i64,
    /// `"spotify"` or `"apple"`.
    pub source: &'static str,
    pub raw_artist: String,
    pub raw_title: String,
    pub raw_album: Option<String>,
    pub ms_played: i64,
    /// `None` when the export doesn't say.
    pub skipped: Option<bool>,
    pub reason_end: Option<String>,
    pub shuffle: Option<bool>,
    pub source_uri: Option<String>,
}

/// How an imported play was tied to the library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchKind {
    /// Same normalized artist and title as a library track.
    Exact,
    /// Close enough (spelling, "Remastered" suffixes, …).
    Fuzzy,
    /// The artist is in the library, this recording isn't.
    Artist,
    None,
}

impl MatchKind {
    pub fn as_str(self) -> &'static str {
        match self {
            MatchKind::Exact => "exact",
            MatchKind::Fuzzy => "fuzzy",
            MatchKind::Artist => "artist",
            MatchKind::None => "none",
        }
    }
}

/// `((raw artist, raw title), track, artist id, kind)` — the result of matching one
/// distinct song from the history.
pub type PlayMatch = ((String, String), Option<TrackId>, Option<i64>, MatchKind);

/// A library track as seen by the history matcher.
#[derive(Debug, Clone, PartialEq)]
pub struct LibraryEntry {
    pub track: TrackId,
    pub title: String,
    /// `(artist id, display name)`, primary artist first.
    pub artists: Vec<(i64, String)>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct HistorySummary {
    pub plays: u64,
    pub first_ts: Option<i64>,
    pub last_ts: Option<i64>,
    pub listened_hours: f64,
    /// Plays the export marks as skipped.
    pub skipped: u64,
    pub matched_exact: u64,
    pub matched_fuzzy: u64,
    pub matched_artist: u64,
    pub unmatched: u64,
    /// `(artist, listens)`, most listened first. A listen is a play that wasn't skipped.
    pub top_artists: Vec<(String, u64)>,
    /// `(artist – title, listens, in library?)`.
    pub top_tracks: Vec<(String, u64, bool)>,
}

/// When an export doesn't say whether a play was skipped, a play shorter than this counts as
/// a skip (the threshold streaming services use for counting streams). When it does say,
/// its flag wins — so a full play of a 20-second track is still a listen.
pub const LISTEN_MS: i64 = 30_000;

fn user_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<User> {
    Ok(User {
        id: UserId(r.get::<_, i64>(0)? as u64),
        name: r.get(1)?,
        adventurousness: r.get::<_, f64>(2)? as f32,
        share_profile: r.get::<_, i64>(3)? != 0,
    })
}

impl Store {
    pub fn add_user(&mut self, name: &str, adventurousness: f32) -> Result<User> {
        let name = name.trim();
        if self.user_by_name(name)?.is_some() {
            return Err(StoreError::UserExists(name.to_owned()));
        }
        self.conn.execute(
            "INSERT INTO users (name, adventurousness, created_at) VALUES (?1, ?2, ?3)",
            params![name, adventurousness.clamp(0.0, 1.0) as f64, now_unix()],
        )?;
        Ok(self.user_by_name(name)?.expect("just inserted"))
    }

    pub fn user_by_name(&self, name: &str) -> Result<Option<User>> {
        Ok(self
            .conn
            .query_row(
                "SELECT id, name, adventurousness, share_profile FROM users WHERE name = ?1",
                [name.trim()],
                user_from_row,
            )
            .optional()?)
    }

    pub fn users(&self) -> Result<Vec<User>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name, adventurousness, share_profile FROM users ORDER BY name")?;
        let rows = stmt.query_map([], user_from_row)?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Whether friends' recommendations may use this user's taste and listening.
    pub fn set_share_profile(&mut self, user: UserId, share: bool) -> Result<()> {
        self.conn.execute(
            "UPDATE users SET share_profile = ?2 WHERE id = ?1",
            params![user.0 as i64, share],
        )?;
        Ok(())
    }

    pub fn set_adventurousness(&mut self, user: UserId, value: f32) -> Result<()> {
        self.conn.execute(
            "UPDATE users SET adventurousness = ?2 WHERE id = ?1",
            params![user.0 as i64, value.clamp(0.0, 1.0) as f64],
        )?;
        Ok(())
    }

    /// Inserts plays, ignoring ones already imported. Returns `(new, already present)`.
    pub fn insert_plays(&mut self, user: UserId, plays: &[PlayRecord]) -> Result<(usize, usize)> {
        let tx = self.conn.transaction()?;
        let mut new = 0;
        {
            let mut stmt = tx.prepare(
                "INSERT OR IGNORE INTO history_plays
                 (user_id, ts, source, raw_artist, raw_title, raw_album, ms_played, skipped, reason_end,
                  shuffle, source_uri)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            )?;
            for p in plays {
                new += stmt.execute(params![
                    user.0 as i64,
                    p.ts,
                    p.source,
                    p.raw_artist,
                    p.raw_title,
                    p.raw_album,
                    p.ms_played,
                    p.skipped,
                    p.reason_end,
                    p.shuffle,
                    p.source_uri
                ])?;
            }
        }
        tx.commit()?;
        Ok((new, plays.len() - new))
    }

    /// Distinct `(artist, title)` pairs not yet tied to a library track — never matched, or
    /// matched only to an artist (the library may have grown since).
    pub fn unmatched_play_keys(&self) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT raw_artist, raw_title FROM history_plays
             WHERE match_kind IS NULL OR match_kind IN ('artist', 'none')",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Records the match for every unmatched play with this artist/title.
    pub fn set_play_matches(&mut self, matches: &[PlayMatch]) -> Result<()> {
        let tx = self.conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "UPDATE history_plays SET track_id = ?3, artist_id = ?4, match_kind = ?5
                 WHERE raw_artist = ?1 AND raw_title = ?2
                   AND (match_kind IS NULL OR match_kind IN ('artist', 'none'))",
            )?;
            for ((artist, title), track, artist_id, kind) in matches {
                stmt.execute(params![
                    artist,
                    title,
                    track.map(|t| t.0 as i64),
                    artist_id,
                    kind.as_str()
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Every library track with its artists, for history matching.
    pub fn library_entries(&self) -> Result<Vec<LibraryEntry>> {
        let mut artists: HashMap<i64, Vec<(i64, String)>> = HashMap::new();
        let mut stmt = self.conn.prepare(
            "SELECT ta.track_id, a.id, a.name FROM track_artists ta JOIN artists a ON a.id = ta.artist_id
             ORDER BY ta.track_id, ta.position",
        )?;
        for row in stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, String>(2)?,
            ))
        })? {
            let (t, a, n) = row?;
            artists.entry(t).or_default().push((a, n));
        }
        let mut stmt = self
            .conn
            .prepare("SELECT id, title FROM tracks ORDER BY id")?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
        let mut out = Vec::new();
        for row in rows {
            let (id, title) = row?;
            out.push(LibraryEntry {
                track: TrackId(id as u64),
                title,
                artists: artists.remove(&id).unwrap_or_default(),
            });
        }
        Ok(out)
    }

    /// `(artist id, name)` for every artist in the library.
    pub fn artists(&self) -> Result<Vec<(i64, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, name FROM artists ORDER BY id")?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Per-track statistics for a user: `(track, listens, skips, last ts)`.
    pub fn track_play_stats(&self, user: UserId) -> Result<Vec<(TrackId, u64, u64, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT track_id,
                    SUM(NOT COALESCE(skipped, ms_played < ?2)),
                    SUM(COALESCE(skipped, ms_played < ?2)),
                    MAX(ts)
             FROM history_plays WHERE user_id = ?1 AND track_id IS NOT NULL GROUP BY track_id",
        )?;
        let rows = stmt.query_map(params![user.0 as i64, LISTEN_MS], |r| {
            Ok((
                TrackId(r.get::<_, i64>(0)? as u64),
                r.get::<_, i64>(1)? as u64,
                r.get::<_, i64>(2)? as u64,
                r.get::<_, i64>(3)?,
            ))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn history_summary(&self, user: UserId, top: usize) -> Result<HistorySummary> {
        let u = user.0 as i64;
        let mut s = self.conn.query_row(
            "SELECT COUNT(*), MIN(ts), MAX(ts), COALESCE(SUM(ms_played), 0) / 3600000.0,
                    COALESCE(SUM(skipped = 1), 0),
                    COALESCE(SUM(match_kind = 'exact'), 0), COALESCE(SUM(match_kind = 'fuzzy'), 0),
                    COALESCE(SUM(match_kind = 'artist'), 0),
                    COALESCE(SUM(match_kind IS NULL OR match_kind = 'none'), 0)
             FROM history_plays WHERE user_id = ?1",
            [u],
            |r| {
                Ok(HistorySummary {
                    plays: r.get::<_, i64>(0)? as u64,
                    first_ts: r.get(1)?,
                    last_ts: r.get(2)?,
                    listened_hours: r.get(3)?,
                    skipped: r.get::<_, i64>(4)? as u64,
                    matched_exact: r.get::<_, i64>(5)? as u64,
                    matched_fuzzy: r.get::<_, i64>(6)? as u64,
                    matched_artist: r.get::<_, i64>(7)? as u64,
                    unmatched: r.get::<_, i64>(8)? as u64,
                    ..Default::default()
                })
            },
        )?;
        let mut stmt = self.conn.prepare(
            "SELECT raw_artist, COUNT(*) AS n FROM history_plays
             WHERE user_id = ?1 AND NOT COALESCE(skipped, ms_played < ?2) GROUP BY raw_artist ORDER BY n DESC, raw_artist LIMIT ?3",
        )?;
        s.top_artists = stmt
            .query_map(params![u, LISTEN_MS, top as i64], |r| {
                Ok((r.get(0)?, r.get::<_, i64>(1)? as u64))
            })?
            .collect::<rusqlite::Result<_>>()?;
        let mut stmt = self.conn.prepare(
            "SELECT raw_artist || ' – ' || raw_title, COUNT(*) AS n, MAX(track_id IS NOT NULL) FROM history_plays
             WHERE user_id = ?1 AND NOT COALESCE(skipped, ms_played < ?2) GROUP BY raw_artist, raw_title
             ORDER BY n DESC, raw_artist, raw_title LIMIT ?3",
        )?;
        s.top_tracks = stmt
            .query_map(params![u, LISTEN_MS, top as i64], |r| {
                Ok((
                    r.get(0)?,
                    r.get::<_, i64>(1)? as u64,
                    r.get::<_, i64>(2)? != 0,
                ))
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play(ts: i64, artist: &str, title: &str, ms: i64, skipped: Option<bool>) -> PlayRecord {
        PlayRecord {
            ts,
            source: "spotify",
            raw_artist: artist.into(),
            raw_title: title.into(),
            raw_album: None,
            ms_played: ms,
            skipped,
            reason_end: None,
            shuffle: None,
            source_uri: None,
        }
    }

    #[test]
    fn users_are_unique_case_insensitively() {
        let mut s = Store::open_in_memory().unwrap();
        let u = s.add_user("Krish", 0.7).unwrap();
        assert_eq!((u.name.as_str(), u.adventurousness), ("Krish", 0.7));
        assert!(matches!(
            s.add_user("krish", 0.5),
            Err(StoreError::UserExists(_))
        ));
        assert_eq!(s.user_by_name("KRISH").unwrap().unwrap().id, u.id);
        s.set_adventurousness(u.id, 2.0).unwrap();
        assert_eq!(s.users().unwrap()[0].adventurousness, 1.0);
    }

    #[test]
    fn import_is_idempotent_and_summarized() {
        let mut s = Store::open_in_memory().unwrap();
        let u = s.add_user("a", 0.5).unwrap().id;
        let plays = vec![
            play(100, "X", "One", 200_000, Some(false)),
            play(200, "X", "One", 5_000, Some(true)),
            play(300, "Y", "Two", 180_000, None),
        ];
        assert_eq!(s.insert_plays(u, &plays).unwrap(), (3, 0));
        assert_eq!(s.insert_plays(u, &plays).unwrap(), (0, 3));

        let keys = s.unmatched_play_keys().unwrap();
        assert_eq!(keys.len(), 2);
        s.set_play_matches(&[(("X".into(), "One".into()), None, Some(7), MatchKind::Artist)])
            .unwrap_err();
        // ^ artist 7 doesn't exist: foreign keys are enforced.
        s.set_play_matches(&[(("Y".into(), "Two".into()), None, None, MatchKind::None)])
            .unwrap();

        let h = s.history_summary(u, 5).unwrap();
        assert_eq!((h.plays, h.skipped, h.unmatched), (3, 1, 3));
        assert_eq!((h.first_ts, h.last_ts), (Some(100), Some(300)));
        assert_eq!(
            h.top_artists,
            vec![("X".to_string(), 1), ("Y".to_string(), 1)]
        );
    }
}
