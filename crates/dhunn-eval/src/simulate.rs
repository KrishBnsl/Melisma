//! Simulated listeners: synthetic Spotify-format history with a *known* hidden taste.
//!
//! A listener's taste is a few "home" tracks in the DSP feature space (z-scored). Each
//! library track gets an affinity = cosine similarity to the nearest home track. Sessions
//! pick tracks with probability rising with affinity and with similarity to the previous
//! track (listening sessions drift smoothly), and skips become likely as affinity drops.
//! Some plays are of popular songs not in the library, as in a real streaming history, and
//! some library titles/artists are spelled the way streaming services spell them
//! ("- Remastered 2009", "The …", "&" ↔ "and") to exercise the matcher.
//!
//! The hidden taste is written out as ground truth so evaluation can later check whether
//! recommendations find what this listener likes.

use dhunn_core::TrackId;
use dhunn_dsp::DSP_VERSION;
use dhunn_import::{spotify::ExtendedRow, time::format_timestamp};
use dhunn_store::{Store, StoreError};
use serde::{Deserialize, Serialize};

use crate::rng::Rng;
use crate::taste::{Taste, TasteSpace, skip_probability};

#[derive(Debug, thiserror::Error)]
pub enum SimError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(
        "the library has {0} analyzed tracks; a simulation needs at least 10 (run `dhunn analyze` first)"
    )]
    TooFewTracks(usize),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimConfig {
    pub seed: u64,
    pub days: u32,
    /// Unix seconds of the last day simulated.
    pub end_ts: i64,
    /// Number of "home" regions in the listener's taste.
    pub taste_regions: usize,
    /// Share of plays that are of songs not in the library.
    pub unowned_share: f64,
    /// Share of library plays whose names are re-spelled the streaming-service way.
    pub respell_share: f64,
    /// Lower = pickier listener (more weight on affinity).
    pub temperature: f64,
    /// Share of library picks that ignore taste (shuffle, radio, a friend's playlist).
    pub random_share: f64,
    /// Which features the listener's taste is defined in.
    #[serde(default)]
    pub taste_space: TasteSpace,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            days: 120,
            end_ts: 1_778_025_600, // 2026-05-06
            taste_regions: 2,
            unowned_share: 0.25,
            respell_share: 0.3,
            temperature: 0.12,
            random_share: 0.12,
            taste_space: TasteSpace::Dsp,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomeTrack {
    pub track: TrackId,
    pub title: String,
    pub artist: Option<String>,
}

/// What the simulated listener secretly likes — for evaluating recommendations later.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruth {
    pub config: SimConfig,
    pub feature_kind: String,
    pub feature_version: u32,
    pub home_tracks: Vec<HomeTrack>,
    /// Every library track's affinity (−1..1) for this listener, highest first.
    pub affinity: Vec<(TrackId, f32)>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimStats {
    pub plays: usize,
    pub library_plays: usize,
    pub unowned_plays: usize,
    pub skips: usize,
    pub respelled: usize,
}

/// Popular songs that are (probably) not in the library: realistic unmatched history.
const UNOWNED: &[(&str, &str, u32)] = &[
    ("Daft Punk", "Get Lucky (feat. Pharrell Williams)", 369),
    ("Fleetwood Mac", "Dreams", 257),
    ("Arctic Monkeys", "Do I Wanna Know?", 272),
    ("Tame Impala", "The Less I Know The Better", 216),
    ("Radiohead", "Weird Fishes/Arpeggi", 318),
    ("Kendrick Lamar", "HUMBLE.", 177),
    ("Frank Ocean", "Pink + White", 184),
    ("A. R. Rahman", "Kun Faya Kun", 473),
    ("Prateek Kuhad", "cold/mess", 323),
    ("Lucky Ali", "O Sanam", 382),
    ("Nusrat Fateh Ali Khan", "Afreen Afreen", 570),
    ("The Weeknd", "Blinding Lights", 200),
    ("Billie Eilish", "bad guy", 194),
    ("Bon Iver", "Holocene", 337),
    ("Nujabes", "Aruarian Dance", 247),
    ("Massive Attack", "Teardrop", 330),
    ("Portishead", "Glory Box", 306),
    ("Miles Davis", "So What", 562),
    ("Nina Simone", "Feeling Good", 177),
    ("Queen", "Bohemian Rhapsody - Remastered 2011", 355),
    ("Led Zeppelin", "Kashmir - Remaster", 508),
    ("Pink Floyd", "Time", 413),
    ("Coldplay", "Yellow", 266),
    ("Gorillaz", "Feel Good Inc.", 221),
    ("Khruangbin", "Maria También", 196),
    ("Anoushka Shankar", "Traces of You", 266),
    ("The Local Train", "Aaoge Tum Kabhi", 288),
    ("Indian Ocean", "Kandisa", 410),
    ("Aphex Twin", "Avril 14th", 125),
    ("Four Tet", "Baby", 259),
];

/// Re-spells a library title/artist the way a streaming service might list it.
fn respell(rng: &mut Rng, artist: &str, title: &str) -> (String, String) {
    match rng.int(0, 3) {
        0 => (
            artist.to_owned(),
            format!("{title} - Remastered {}", rng.int(2005, 2023)),
        ),
        1 if !artist.to_lowercase().starts_with("the ") && !artist.is_empty() => {
            (format!("The {artist}"), title.to_owned())
        }
        2 if artist.contains(" & ") => (artist.replace(" & ", " and "), title.to_owned()),
        _ => (artist.to_owned(), format!("{title} (Mono)")),
    }
}

/// Simulates one listener's history in Spotify's extended-history format.
pub fn simulate_spotify_history(
    store: &Store,
    cfg: &SimConfig,
) -> Result<(Vec<ExtendedRow>, GroundTruth, SimStats), SimError> {
    let mut rng = Rng::new(cfg.seed);
    let taste = Taste::hidden_in(store, cfg.taste_space, &mut rng, cfg.taste_regions)?;
    let ids = taste.ids.clone();

    // Names and durations for every track.
    let mut meta = Vec::with_capacity(ids.len());
    for id in &ids {
        let t = store
            .track(*id)?
            .expect("feature rows reference existing tracks");
        meta.push((t.artists.first().cloned(), t.title, t.album, t.duration_s));
    }

    let n = ids.len();
    let (homes, affinity) = (&taste.homes, &taste.affinity);
    let base_weight: Vec<f64> = affinity
        .iter()
        .map(|a| (a / cfg.temperature).exp())
        .collect();

    let mut out = Vec::new();
    let mut stats = SimStats::default();
    let first_day = cfg.end_ts.div_euclid(86_400) - cfg.days as i64 + 1;
    for day in 0..cfg.days as i64 {
        let sessions = rng.weighted(&[0.15, 0.45, 0.3, 0.1]);
        for _ in 0..sessions {
            // Mostly evenings, some mornings/commutes (UTC hours; good enough for a sample).
            let hour = *rng.pick(&[7.5, 8.5, 13.0, 18.0, 19.5, 21.0, 22.0, 23.0]);
            let mut ts = (first_day + day) * 86_400 + (hour * 3600.0) as i64 + rng.int(0, 1800);
            let len = rng.int(6, 22);
            let mut prev: Option<usize> = None;
            for _ in 0..len {
                let (artist, title, album, dur_s, aff, uri) = if rng.chance(cfg.unowned_share) {
                    let (a, t, d) = *rng.pick(UNOWNED);
                    stats.unowned_plays += 1;
                    (
                        a.to_owned(),
                        t.to_owned(),
                        None,
                        d as f64,
                        0.2 + rng.range(-0.3, 0.3),
                        None,
                    )
                } else {
                    // Continuity: weight by affinity and by similarity to the previous track.
                    let weights: Vec<f64> = match prev {
                        _ if rng.chance(cfg.random_share) => vec![1.0; n],
                        Some(p) => (0..n)
                            .map(|i| {
                                base_weight[i]
                                    * ((taste.cosine(i, p) * 0.5) / cfg.temperature).exp()
                            })
                            .collect(),
                        None => base_weight.clone(),
                    };
                    let i = rng.weighted(&weights);
                    prev = Some(i);
                    stats.library_plays += 1;
                    let (artist, title, album, dur) = &meta[i];
                    let artist = artist
                        .clone()
                        .unwrap_or_else(|| "Unknown Artist".to_owned());
                    let (artist, title) = if rng.chance(cfg.respell_share) {
                        stats.respelled += 1;
                        respell(&mut rng, &artist, title)
                    } else {
                        (artist, title.clone())
                    };
                    let uri = Some(format!("spotify:track:dhunnsim{}", ids[i].0));
                    (artist, title, album.clone(), *dur, affinity[i], uri)
                };
                // Skips get likely as affinity drops.
                let p_skip = skip_probability(aff);
                let skipped = rng.chance(p_skip);
                let dur_ms = (dur_s * 1000.0) as i64;
                let ms_played = if skipped {
                    rng.int(1_500, 30_000).min(dur_ms.max(1_500))
                } else {
                    (dur_ms as f64 * rng.range(0.93, 1.0)) as i64
                };
                ts += ms_played / 1000;
                stats.plays += 1;
                stats.skips += usize::from(skipped);
                out.push(ExtendedRow {
                    ts: format_timestamp(ts),
                    platform: Some("android".into()),
                    ms_played,
                    conn_country: Some("IN".into()),
                    master_metadata_track_name: Some(title),
                    master_metadata_album_artist_name: Some(artist),
                    master_metadata_album_album_name: album,
                    spotify_track_uri: uri,
                    reason_start: Some(
                        if out.is_empty() {
                            "clickrow"
                        } else {
                            "trackdone"
                        }
                        .into(),
                    ),
                    reason_end: Some(if skipped { "fwdbtn" } else { "trackdone" }.into()),
                    shuffle: Some(false),
                    skipped: Some(skipped),
                    offline: Some(false),
                    incognito_mode: Some(false),
                    ..Default::default()
                });
                ts += rng.int(0, 3);
            }
        }
    }

    let mut ranked: Vec<(TrackId, f32)> = ids
        .iter()
        .zip(affinity)
        .map(|(&t, &a)| (t, a as f32))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let truth = GroundTruth {
        config: cfg.clone(),
        feature_kind: "dsp".into(),
        feature_version: DSP_VERSION,
        home_tracks: homes
            .iter()
            .map(|&h| HomeTrack {
                track: ids[h],
                title: meta[h].1.clone(),
                artist: meta[h].0.clone(),
            })
            .collect(),
        affinity: ranked,
    };
    Ok((out, truth, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dhunn_core::{ArtistCredit, ArtistRole, FeatureVec, TrackMeta};
    use dhunn_store::{FileStamp, NewAnalysis};

    fn library(n: usize) -> Store {
        let mut s = Store::open_in_memory().unwrap();
        for i in 0..n {
            // Two clusters along the first dimension.
            let x = if i % 2 == 0 { 1.0 } else { -1.0 };
            let a = NewAnalysis {
                content_hash: format!("h{i}"),
                meta: TrackMeta {
                    title: format!("Song {i}"),
                    album: None,
                    year: None,
                    duration_s: 180.0,
                    mb_recording_id: None,
                    artists: vec![ArtistCredit {
                        name: format!("Artist {}", i % 5),
                        role: ArtistRole::Primary,
                        mb_artist_id: None,
                    }],
                },
                features: FeatureVec {
                    kind: "dsp".into(),
                    version: DSP_VERSION,
                    names: vec!["a".into(), "b".into()],
                    values: vec![x + 0.01 * i as f32, (i as f32 * 0.37).sin()],
                },
                summary_json: None,
            };
            s.save_analysis(
                &format!("/m/{i}"),
                FileStamp {
                    size: 1,
                    mtime_ns: 1,
                },
                &a,
            )
            .unwrap();
        }
        s
    }

    #[test]
    fn listener_prefers_and_finishes_their_taste() {
        let store = library(40);
        let cfg = SimConfig {
            days: 60,
            taste_regions: 1,
            ..Default::default()
        };
        let (rows, truth, stats) = simulate_spotify_history(&store, &cfg).unwrap();
        assert_eq!(rows.len(), stats.plays);
        assert!(stats.plays > 300, "{stats:?}");
        assert!(stats.unowned_plays > 0 && stats.respelled > 0);

        // Plays of tracks in the listener's half of the library dominate and are skipped less.
        let liked: std::collections::HashSet<String> = truth
            .affinity
            .iter()
            .filter(|(_, a)| *a > 0.0)
            .map(|(t, _)| format!("spotify:track:dhunnsim{}", t.0))
            .collect();
        let lib: Vec<&ExtendedRow> = rows
            .iter()
            .filter(|r| r.spotify_track_uri.is_some())
            .collect();
        let (near, far): (Vec<&&ExtendedRow>, Vec<&&ExtendedRow>) = lib
            .iter()
            .partition(|r| liked.contains(r.spotify_track_uri.as_ref().unwrap()));
        assert!(
            near.len() > 3 * far.len(),
            "near {} far {}",
            near.len(),
            far.len()
        );
        let skip_rate = |v: &[&&ExtendedRow]| {
            v.iter().filter(|r| r.skipped == Some(true)).count() as f64 / v.len().max(1) as f64
        };
        assert!(
            skip_rate(&near) < skip_rate(&far),
            "near {} skip {:.2}, far {} skip {:.2}",
            near.len(),
            skip_rate(&near),
            far.len(),
            skip_rate(&far)
        );
    }

    #[test]
    fn deterministic_for_a_seed() {
        let store = library(20);
        let cfg = SimConfig {
            days: 5,
            ..Default::default()
        };
        let a = simulate_spotify_history(&store, &cfg).unwrap().0;
        let b = simulate_spotify_history(&store, &cfg).unwrap().0;
        assert_eq!(a, b);
    }

    #[test]
    fn needs_a_library() {
        assert!(matches!(
            simulate_spotify_history(&library(3), &SimConfig::default()),
            Err(SimError::TooFewTracks(3))
        ));
    }
}
