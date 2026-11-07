//! What actually happened in your `dhunn play` sessions: the real-world score.

use dhunn_store::LoggedSession;
use serde::{Deserialize, Serialize};

/// A skip before this point says "not this" rather than "enough of this".
const EARLY_SKIP_MS: i64 = 30_000;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Counts {
    pub sessions: usize,
    pub plays: usize,
    pub completes: usize,
    pub skips: usize,
    pub early_skips: usize,
    pub likes: usize,
    pub dislikes: usize,
    pub further: usize,
    pub back: usize,
}

impl Counts {
    pub fn skip_rate(&self) -> f64 {
        self.skips as f64 / self.plays.max(1) as f64
    }

    pub fn early_skip_rate(&self) -> f64 {
        self.early_skips as f64 / self.plays.max(1) as f64
    }

    pub fn like_rate(&self) -> f64 {
        self.likes as f64 / self.plays.max(1) as f64
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoggedReport {
    pub all: Counts,
    pub last_7_days: Counts,
}

fn count(sessions: &[&LoggedSession]) -> Counts {
    let mut c = Counts::default();
    for s in sessions {
        if s.events.iter().any(|e| e.kind == "play") {
            c.sessions += 1;
        }
        for e in &s.events {
            match e.kind.as_str() {
                "play" => c.plays += 1,
                "complete" => c.completes += 1,
                "skip" => {
                    c.skips += 1;
                    if e.position_ms.is_some_and(|p| p < EARLY_SKIP_MS) {
                        c.early_skips += 1;
                    }
                }
                "like" => c.likes += 1,
                "dislike" => c.dislikes += 1,
                "further" => c.further += 1,
                "back" => c.back += 1,
                _ => {}
            }
        }
    }
    c
}

/// `None` when nothing has been played in dhunn yet.
pub fn summarize(sessions: &[LoggedSession], now: i64) -> Option<LoggedReport> {
    // Automated audio checks (`dhunn play --smoke`) aren't listening.
    let all: Vec<&LoggedSession> = sessions.iter().filter(|s| s.mode != "smoke").collect();
    let recent: Vec<&LoggedSession> = all
        .iter()
        .copied()
        .filter(|s| s.started_at >= now - 7 * 86_400)
        .collect();
    let report = LoggedReport {
        all: count(&all),
        last_7_days: count(&recent),
    };
    (report.all.plays > 0).then_some(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dhunn_store::LoggedEvent;

    fn ev(kind: &str, pos: Option<i64>) -> LoggedEvent {
        LoggedEvent {
            kind: kind.into(),
            track: None,
            position_ms: pos,
        }
    }

    #[test]
    fn counts_recent_and_early_skips() {
        let old = LoggedSession {
            id: 1,
            started_at: 0,
            mode: "drift".into(),
            events: vec![ev("play", None), ev("complete", None)],
        };
        let new = LoggedSession {
            id: 2,
            started_at: 1_000_000,
            mode: "drift".into(),
            events: vec![
                ev("play", None),
                ev("skip", Some(5_000)),
                ev("play", None),
                ev("like", None),
                ev("skip", Some(120_000)),
            ],
        };
        let smoke = LoggedSession {
            id: 3,
            started_at: 1_000_050,
            mode: "smoke".into(),
            events: vec![ev("play", None), ev("skip", Some(2_000))],
        };
        let r = summarize(&[old, new, smoke], 1_000_100).unwrap();
        assert_eq!(r.all.plays, 3);
        assert_eq!(r.all.sessions, 2);
        assert_eq!(r.last_7_days.sessions, 1);
        assert_eq!(r.last_7_days.skips, 2);
        assert_eq!(r.last_7_days.early_skips, 1);
        assert!((r.all.skip_rate() - 2.0 / 3.0).abs() < 1e-12);
        assert!(summarize(&[], 0).is_none());
    }
}
