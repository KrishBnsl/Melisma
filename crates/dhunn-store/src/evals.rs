//! Data for evaluation: plays in time order, logged player sessions, and saved eval reports.

use dhunn_core::{TrackId, UserId};
use rusqlite::{OptionalExtension, params};

use crate::{LISTEN_MS, Result, Store, now_unix};

/// One imported play of a library track.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchedPlay {
    /// Unix seconds.
    pub ts: i64,
    pub track: TrackId,
    pub ms_played: i64,
    /// Played long enough (or not flagged as skipped) to count as a listen.
    pub listened: bool,
}

/// One event from a `dhunn play` session.
#[derive(Debug, Clone, PartialEq)]
pub struct LoggedEvent {
    pub kind: String,
    pub track: Option<TrackId>,
    pub position_ms: Option<i64>,
}

/// A `dhunn play` session with its events in order.
#[derive(Debug, Clone, PartialEq)]
pub struct LoggedSession {
    pub id: u64,
    /// Unix seconds.
    pub started_at: i64,
    pub mode: String,
    pub events: Vec<LoggedEvent>,
}

/// A saved `dhunn eval` report.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalRun {
    pub id: u64,
    pub created_at: i64,
    pub label: Option<String>,
    pub report_json: String,
}

impl Store {
    /// A user's imported plays that matched a library track, oldest first.
    pub fn matched_plays(&self, user: UserId) -> Result<Vec<MatchedPlay>> {
        let mut stmt = self.conn.prepare(
            "SELECT ts, track_id, ms_played, NOT COALESCE(skipped, ms_played < ?2)
             FROM history_plays WHERE user_id = ?1 AND track_id IS NOT NULL ORDER BY ts, id",
        )?;
        let rows = stmt.query_map(params![user.0 as i64, LISTEN_MS], |r| {
            Ok(MatchedPlay {
                ts: r.get(0)?,
                track: TrackId(r.get::<_, i64>(1)? as u64),
                ms_played: r.get(2)?,
                listened: r.get(3)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// A user's `dhunn play` sessions, oldest first.
    pub fn logged_sessions(&self, user: UserId) -> Result<Vec<LoggedSession>> {
        let mut stmt = self.conn.prepare(
            "SELECT s.id, s.started_at, s.mode, e.kind, e.track_id, e.position_ms
             FROM sessions s LEFT JOIN events e ON e.session_id = s.id
             WHERE s.user_id = ?1 ORDER BY s.id, e.id",
        )?;
        let mut out: Vec<LoggedSession> = Vec::new();
        let mut rows = stmt.query([user.0 as i64])?;
        while let Some(r) = rows.next()? {
            let id = r.get::<_, i64>(0)? as u64;
            if out.last().is_none_or(|s| s.id != id) {
                out.push(LoggedSession {
                    id,
                    started_at: r.get(1)?,
                    mode: r.get(2)?,
                    events: Vec::new(),
                });
            }
            if let Some(kind) = r.get::<_, Option<String>>(3)? {
                let session = out.last_mut().expect("pushed above");
                session.events.push(LoggedEvent {
                    kind,
                    track: r.get::<_, Option<i64>>(4)?.map(|t| TrackId(t as u64)),
                    position_ms: r.get(5)?,
                });
            }
        }
        Ok(out)
    }

    pub fn save_eval_run(&mut self, label: Option<&str>, report_json: &str) -> Result<u64> {
        self.conn.execute(
            "INSERT INTO eval_runs (created_at, label, report) VALUES (?1, ?2, ?3)",
            params![now_unix(), label, report_json],
        )?;
        Ok(self.conn.last_insert_rowid() as u64)
    }

    /// The most recent saved report.
    pub fn last_eval_run(&self) -> Result<Option<EvalRun>> {
        Ok(self
            .conn
            .query_row(
                "SELECT id, created_at, label, report FROM eval_runs ORDER BY id DESC LIMIT 1",
                [],
                |r| {
                    Ok(EvalRun {
                        id: r.get::<_, i64>(0)? as u64,
                        created_at: r.get(1)?,
                        label: r.get(2)?,
                        report_json: r.get(3)?,
                    })
                },
            )
            .optional()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EventKind, PlayRecord};

    #[test]
    fn logged_sessions_group_events() {
        let mut s = Store::open_in_memory().unwrap();
        let u = s.add_user("a", 0.5).unwrap().id;
        let a = s.start_session(Some(u), "drift", "{}").unwrap();
        s.log_event(a, Some(u), None, EventKind::Play, None, None)
            .unwrap();
        s.log_event(a, Some(u), None, EventKind::Skip, Some(4_000), None)
            .unwrap();
        s.start_session(Some(u), "drift", "{}").unwrap();
        let got = s.logged_sessions(u).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].events.len(), 2);
        assert_eq!(got[0].events[1].kind, "skip");
        assert_eq!(got[0].events[1].position_ms, Some(4_000));
        assert!(got[1].events.is_empty());
    }

    #[test]
    fn matched_plays_and_eval_runs() {
        let mut s = Store::open_in_memory().unwrap();
        let u = s.add_user("a", 0.5).unwrap().id;
        let play = |ts, ms| PlayRecord {
            ts,
            source: "spotify",
            raw_artist: "x".into(),
            raw_title: format!("t{ts}"),
            raw_album: None,
            ms_played: ms,
            skipped: None,
            reason_end: None,
            shuffle: None,
            source_uri: None,
        };
        s.insert_plays(u, &[play(20, 5_000), play(10, 200_000)])
            .unwrap();
        // Unmatched plays are not returned.
        assert!(s.matched_plays(u).unwrap().is_empty());

        assert!(s.last_eval_run().unwrap().is_none());
        s.save_eval_run(None, "{\"a\":1}").unwrap();
        s.save_eval_run(Some("tuned"), "{\"a\":2}").unwrap();
        let last = s.last_eval_run().unwrap().unwrap();
        assert_eq!(last.label.as_deref(), Some("tuned"));
        assert_eq!(last.report_json, "{\"a\":2}");
    }
}
