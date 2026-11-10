//! Listening data as sessions of observations, from imported history and `dhunn play` logs.

use dhunn_core::TrackId;
use dhunn_store::{LISTEN_MS, LoggedSession, MatchedPlay};

/// One track heard in a session.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Obs {
    pub track: TrackId,
    /// Unix seconds.
    pub ts: i64,
    pub listened: bool,
    pub liked: bool,
    pub disliked: bool,
}

/// Cuts plays (oldest first) into sessions: a pause longer than `gap_s` beyond a track's own
/// length starts a new one. (`ts` is when a play ended, as in Spotify exports.)
pub fn history_sessions(plays: &[MatchedPlay], gap_s: i64) -> Vec<Vec<Obs>> {
    let mut out: Vec<Vec<Obs>> = Vec::new();
    let mut prev_ts: Option<i64> = None;
    for p in plays {
        if prev_ts.is_none_or(|t| p.ts - p.ms_played / 1000 - t > gap_s) {
            out.push(Vec::new());
        }
        prev_ts = Some(p.ts);
        out.last_mut().expect("pushed").push(Obs {
            track: p.track,
            ts: p.ts,
            listened: p.listened,
            liked: false,
            disliked: false,
        });
    }
    out
}

/// `dhunn play` sessions as observations (automated `smoke` sessions are left out). A skip
/// in the first 30 s counts as not listened, like in imported history.
pub fn logged_sessions(sessions: &[LoggedSession]) -> Vec<Vec<Obs>> {
    observations(sessions, true)
}

/// Like [`logged_sessions`], without `--private` sessions: what friends may learn from.
pub fn shared_sessions(sessions: &[LoggedSession]) -> Vec<Vec<Obs>> {
    observations(sessions, false)
}

fn observations(sessions: &[LoggedSession], include_private: bool) -> Vec<Vec<Obs>> {
    // Whether each track's latest like/unlike (in any session) was a like.
    let mut likes: std::collections::HashMap<TrackId, bool> = Default::default();
    for e in sessions.iter().flat_map(|s| &s.events) {
        if let (Some(t), "like" | "unlike") = (e.track, e.kind.as_str()) {
            likes.insert(t, e.kind == "like");
        }
    }
    let mut out = Vec::new();
    for s in sessions
        .iter()
        .filter(|s| s.mode != "smoke" && (include_private || !s.private))
    {
        let mut obs: Vec<Obs> = Vec::new();
        for e in &s.events {
            let Some(track) = e.track else { continue };
            match e.kind.as_str() {
                "play" => obs.push(Obs {
                    track,
                    ts: s.started_at,
                    listened: true,
                    liked: false,
                    disliked: false,
                }),
                // A like from another player's library (no play here): a strong yes.
                "like" if !obs.iter().any(|o| o.track == track) => obs.push(Obs {
                    track,
                    ts: s.started_at,
                    listened: true,
                    liked: true,
                    disliked: false,
                }),
                kind => {
                    let Some(o) = obs.iter_mut().rev().find(|o| o.track == track) else {
                        continue;
                    };
                    match kind {
                        "skip" => o.listened = e.position_ms.is_some_and(|p| p >= LISTEN_MS),
                        "like" => o.liked = true,
                        "dislike" => {
                            o.disliked = true;
                            o.listened = false;
                        }
                        "error" => o.listened = false,
                        _ => {}
                    }
                }
            }
        }
        // A track that errored before playing tells us nothing.
        obs.retain(|o| {
            o.listened || o.liked || o.disliked || {
                s.events
                    .iter()
                    .any(|e| e.track == Some(o.track) && e.kind == "skip")
            }
        });
        // Likes taken back since don't count.
        for o in &mut obs {
            if o.liked && likes.get(&o.track) == Some(&false) {
                o.liked = false;
            }
        }
        if !obs.is_empty() {
            out.push(obs);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use dhunn_store::LoggedEvent;

    #[test]
    fn logged_events_become_observations() {
        let ev = |kind: &str, t: u64, pos: Option<i64>| LoggedEvent {
            kind: kind.into(),
            track: Some(TrackId(t)),
            position_ms: pos,
        };
        let s = LoggedSession {
            id: 1,
            started_at: 100,
            mode: "drift".into(),
            private: false,
            events: vec![
                ev("play", 1, None),
                ev("like", 1, Some(10_000)),
                ev("complete", 1, None),
                ev("play", 2, None),
                ev("skip", 2, Some(4_000)),
                ev("play", 3, None),
                ev("skip", 3, Some(95_000)),
                ev("play", 4, None),
                ev("error", 4, None),
            ],
        };
        let smoke = LoggedSession {
            mode: "smoke".into(),
            private: false,
            ..s.clone()
        };
        let private = LoggedSession {
            id: 3,
            private: true,
            ..s.clone()
        };
        assert_eq!(logged_sessions(std::slice::from_ref(&private)).len(), 1);
        assert!(shared_sessions(&[private]).is_empty());
        let got = logged_sessions(&[s, smoke]);
        assert_eq!(got.len(), 1);
        let o = &got[0];
        assert_eq!(o.len(), 3, "{o:?}");
        assert!(o[0].listened && o[0].liked);
        assert!(!o[1].listened);
        assert!(o[2].listened, "a late skip still counts as a listen");
    }

    #[test]
    fn likes_from_elsewhere_and_unlikes() {
        let ev = |kind: &str, t: u64| LoggedEvent {
            kind: kind.into(),
            track: Some(TrackId(t)),
            position_ms: None,
        };
        let s = |id, events| LoggedSession {
            id,
            started_at: 100,
            mode: "drift".into(),
            private: false,
            events,
        };
        let got = logged_sessions(&[
            s(1, vec![ev("play", 1), ev("like", 1), ev("like", 7)]),
            s(2, vec![ev("unlike", 1)]),
        ]);
        let o = &got[0];
        // Track 7 was liked without a play here; track 1's like was taken back later.
        assert!(o.iter().any(|o| o.track == TrackId(7) && o.liked));
        assert!(
            o.iter()
                .any(|o| o.track == TrackId(1) && !o.liked && o.listened)
        );
    }
}
