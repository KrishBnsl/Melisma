//! Ties imported plays to library tracks by normalized artist + title.
//!
//! Order of attempts per distinct (artist, title):
//! 1. **exact** — same normalized artist and title;
//! 2. **fuzzy** — same artist with a near-identical title, or same title with a
//!    near-identical artist (Jaro–Winkler);
//! 3. **artist** — the recording isn't in the library but the artist is (still a taste
//!    signal, and a "you might want to add this" candidate);
//! 4. **none**.
//!
//! Plays matched only to an artist (or not at all) are retried on every run, so they get
//! picked up when the library grows.

use std::collections::HashMap;

use dhunn_core::TrackId;
use dhunn_store::{MatchKind, Store, StoreError};

/// Similarity needed for a fuzzy title match (same artist).
const TITLE_SIMILARITY: f64 = 0.92;
/// Similarity needed for a fuzzy artist match (same title).
const ARTIST_SIMILARITY: f64 = 0.90;

/// Parenthesized or " - " suffix notes that don't change which recording it is.
const VERSION_NOTES: &[&str] = &[
    "remaster",
    "mono",
    "stereo",
    "single version",
    "album version",
    "radio edit",
    "deluxe",
    "bonus track",
    "explicit",
    "clean",
    "original mix",
    "feat",
    "ft.",
    "with ",
];

fn is_version_note(s: &str) -> bool {
    let s = s.trim().to_lowercase();
    VERSION_NOTES.iter().any(|n| s.contains(n))
        || s.chars().all(|c| c.is_ascii_digit() || c.is_whitespace())
}

fn squash(s: &str) -> String {
    let mapped: String = s
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c
            } else if c == '&' {
                '&'
            } else {
                ' '
            }
        })
        .collect();
    mapped
        .split_whitespace()
        .map(|w| if w == "&" { "and" } else { w })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Lowercases and strips version notes ("- Remastered 2009", "(feat. X)", "[Mono]") and
/// punctuation. Keeps "live", "acoustic", "remix": those are different recordings.
pub fn normalize_title(title: &str) -> String {
    let s = title.to_lowercase();
    // Bracketed notes anywhere (first, so a " - " inside brackets isn't mistaken for a suffix).
    let mut out = String::with_capacity(s.len());
    let mut rest = s.as_str();
    while let Some(open) = rest.find(['(', '[']) {
        let close_ch = if rest.as_bytes()[open] == b'(' {
            ')'
        } else {
            ']'
        };
        let Some(close) = rest[open..].find(close_ch).map(|c| open + c) else {
            break;
        };
        out.push_str(&rest[..open]);
        let inner = &rest[open + 1..close];
        if !is_version_note(inner) {
            out.push(' ');
            out.push_str(inner);
            out.push(' ');
        }
        rest = &rest[close + 1..];
    }
    out.push_str(rest);
    // " - Remastered 2009" style suffixes (repeatedly: "Song - 2011 Remaster - Mono").
    while let Some(i) = out.rfind(" - ") {
        if is_version_note(&out[i + 3..]) {
            out.truncate(i);
        } else {
            break;
        }
    }
    squash(&out)
}

/// Lowercase, punctuation-free, "&" → "and", leading "the " dropped.
pub fn normalize_artist(name: &str) -> String {
    let s = squash(&name.to_lowercase());
    s.strip_prefix("the ").map(str::to_owned).unwrap_or(s)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MatchReport {
    /// Distinct (artist, title) pairs examined.
    pub pairs: usize,
    pub exact: usize,
    pub fuzzy: usize,
    pub artist: usize,
    pub none: usize,
}

struct LibraryIndex {
    exact: HashMap<(String, String), TrackId>,
    by_artist: HashMap<String, Vec<(String, TrackId)>>,
    by_title: HashMap<String, Vec<(String, TrackId)>>,
    /// Distinctive titles of tracks that have no artist at all (untagged files).
    artistless: HashMap<String, TrackId>,
    artists: HashMap<String, i64>,
}

/// An artistless track only matches on an exact title at least this long, so that
/// generic names ("Intro", "Track 1") never match.
const ARTISTLESS_MIN_TITLE: usize = 12;

impl LibraryIndex {
    fn build(store: &Store) -> Result<Self, StoreError> {
        let mut idx = LibraryIndex {
            exact: HashMap::new(),
            by_artist: HashMap::new(),
            by_title: HashMap::new(),
            artistless: HashMap::new(),
            artists: HashMap::new(),
        };
        for (id, name) in store.artists()? {
            idx.artists.entry(normalize_artist(&name)).or_insert(id);
        }
        for e in store.library_entries()? {
            let t = normalize_title(&e.title);
            if t.is_empty() {
                continue;
            }
            if e.artists.is_empty() && t.chars().count() >= ARTISTLESS_MIN_TITLE {
                idx.artistless.entry(t.clone()).or_insert(e.track);
            }
            for (_, name) in &e.artists {
                let a = normalize_artist(name);
                idx.exact.entry((a.clone(), t.clone())).or_insert(e.track);
                idx.by_artist
                    .entry(a.clone())
                    .or_default()
                    .push((t.clone(), e.track));
                idx.by_title
                    .entry(t.clone())
                    .or_default()
                    .push((a, e.track));
            }
        }
        Ok(idx)
    }

    fn best(cands: &[(String, TrackId)], key: &str, threshold: f64) -> Option<TrackId> {
        cands
            .iter()
            .map(|(k, t)| (strsim::jaro_winkler(k, key), *t))
            .filter(|(s, _)| *s >= threshold)
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, t)| t)
    }

    fn lookup(
        &self,
        raw_artist: &str,
        raw_title: &str,
    ) -> (Option<TrackId>, Option<i64>, MatchKind) {
        let title = normalize_title(raw_title);
        // Every credited name counts: "A feat. B" can be filed under A or B.
        let mut artists: Vec<String> = dhunn_meta::parse_artist_credits(raw_artist)
            .iter()
            .map(|c| normalize_artist(&c.name))
            .collect();
        artists.push(normalize_artist(raw_artist));
        artists.retain(|a| !a.is_empty());
        artists.dedup();
        let artist_id = artists.iter().find_map(|a| self.artists.get(a).copied());

        for a in &artists {
            if let Some(&t) = self.exact.get(&(a.clone(), title.clone())) {
                return (Some(t), artist_id, MatchKind::Exact);
            }
        }
        for a in &artists {
            if let Some(t) = self
                .by_artist
                .get(a)
                .and_then(|c| Self::best(c, &title, TITLE_SIMILARITY))
            {
                return (Some(t), artist_id, MatchKind::Fuzzy);
            }
        }
        if let Some(cands) = self.by_title.get(&title) {
            for a in &artists {
                if let Some(t) = Self::best(cands, a, ARTIST_SIMILARITY) {
                    return (Some(t), artist_id, MatchKind::Fuzzy);
                }
            }
        }
        if let Some(&t) = self.artistless.get(&title) {
            return (Some(t), artist_id, MatchKind::Fuzzy);
        }
        match artist_id {
            Some(_) => (None, artist_id, MatchKind::Artist),
            None => (None, None, MatchKind::None),
        }
    }
}

/// Matches every not-yet-matched play (all users) against the current library.
pub fn match_history(store: &mut Store) -> Result<MatchReport, StoreError> {
    let keys = store.unmatched_play_keys()?;
    if keys.is_empty() {
        return Ok(MatchReport::default());
    }
    let idx = LibraryIndex::build(store)?;
    let mut report = MatchReport {
        pairs: keys.len(),
        ..Default::default()
    };
    let mut updates = Vec::with_capacity(keys.len());
    for (artist, title) in keys {
        let (track, artist_id, kind) = idx.lookup(&artist, &title);
        match kind {
            MatchKind::Exact => report.exact += 1,
            MatchKind::Fuzzy => report.fuzzy += 1,
            MatchKind::Artist => report.artist += 1,
            MatchKind::None => report.none += 1,
        }
        updates.push(((artist, title), track, artist_id, kind));
    }
    store.set_play_matches(&updates)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dhunn_core::{ArtistCredit, ArtistRole, FeatureVec, TrackMeta};
    use dhunn_store::{FileStamp, NewAnalysis, PlayRecord};

    #[test]
    fn title_normalization() {
        let n = normalize_title;
        assert_eq!(n("Let It Be - Remastered 2009"), "let it be");
        assert_eq!(n("Let It Be (2009 Remaster)"), "let it be");
        assert_eq!(n("Get Lucky (feat. Pharrell Williams)"), "get lucky");
        assert_eq!(n("Song - 2011 Remaster - Mono"), "song");
        assert_eq!(n("Knockin' On Heaven's Door"), "knockin on heaven s door");
        assert_eq!(n("Rock & Roll"), "rock and roll");
        // Different recordings stay different.
        assert_eq!(
            n("Song - Live at Wembley"),
            "song - live at wembley".replace(" - ", " ").to_string()
        );
        assert_eq!(n("Song (Acoustic)"), "song acoustic");
        assert_eq!(n("Song [Remix]"), "song remix");
    }

    #[test]
    fn artist_normalization() {
        assert_eq!(normalize_artist("The Beatles"), "beatles");
        assert_eq!(normalize_artist("Simon & Garfunkel"), "simon and garfunkel");
        assert_eq!(normalize_artist("AC/DC"), "ac dc");
    }

    fn add(s: &mut Store, hash: &str, title: &str, artist: &str) -> TrackId {
        let a = NewAnalysis {
            content_hash: hash.into(),
            meta: TrackMeta {
                title: title.into(),
                album: None,
                year: None,
                duration_s: 200.0,
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
            summary_json: None,
        };
        s.save_analysis(
            &format!("/m/{hash}"),
            FileStamp {
                size: 1,
                mtime_ns: 1,
            },
            &a,
        )
        .unwrap()
    }

    fn play(artist: &str, title: &str, ts: i64) -> PlayRecord {
        PlayRecord {
            ts,
            source: "spotify",
            raw_artist: artist.into(),
            raw_title: title.into(),
            raw_album: None,
            ms_played: 100_000,
            skipped: Some(false),
            reason_end: None,
            shuffle: None,
            source_uri: None,
        }
    }

    #[test]
    fn matches_against_library_and_rematches_later() {
        let mut s = Store::open_in_memory().unwrap();
        let let_it_be = add(&mut s, "h1", "Let It Be", "The Beatles");
        let paranoid = add(&mut s, "h2", "Paranoid", "Black Sabbath");
        let u = s.add_user("me", 0.5).unwrap().id;
        s.insert_plays(
            u,
            &[
                play("Beatles", "Let It Be - Remastered 2009", 1), // exact after normalization
                play("Black Sabath", "Paranoid", 2),               // fuzzy artist
                play("The Beatles", "Hey Jude", 3),                // artist only
                play("Daft Punk", "Get Lucky (feat. Pharrell Williams)", 4), // none
                play("Black Sabbath", "Paranoid (2012 - Remaster)", 5), // exact
            ],
        )
        .unwrap();
        let r = match_history(&mut s).unwrap();
        assert_eq!(
            (r.pairs, r.exact, r.fuzzy, r.artist, r.none),
            (5, 2, 1, 1, 1),
            "{r:?}"
        );
        let stats = s.track_play_stats(u).unwrap();
        let ids: Vec<TrackId> = stats.iter().map(|x| x.0).collect();
        assert!(ids.contains(&let_it_be) && ids.contains(&paranoid));

        // Nothing new to match → nothing re-examined except artist/none rows.
        let r = match_history(&mut s).unwrap();
        assert_eq!((r.pairs, r.exact + r.fuzzy), (2, 0));

        // The library grows: the artist-only play now matches.
        add(&mut s, "h3", "Hey Jude", "The Beatles");
        let r = match_history(&mut s).unwrap();
        assert_eq!((r.exact, r.none), (1, 1), "{r:?}");
    }
}
