//! Library → simulated history → import → every eval section.

use dhunn_core::{ArtistCredit, ArtistRole, FeatureVec, TrackMeta};
use dhunn_dsp::DSP_VERSION;
use dhunn_eval::{
    ListenerConfig, ReplayConfig, Report, SimConfig, params_list, render, replay,
    simulate_spotify_history, simulated_listeners, summarize,
};
use dhunn_import::Service;
use dhunn_index::{AspectWeights, SoundSpace};
use dhunn_steer::{SessionConfig, SteerParams};
use dhunn_store::{FileStamp, NewAnalysis, Store};

/// 120 tracks on a ring in a small timbre/rhythm/brightness space: neighbours sound alike.
fn library() -> Store {
    let mut s = Store::open_in_memory().unwrap();
    let names: Vec<String> = ["mfcc00_mean", "mfcc01_mean", "log2_tempo", "centroid_mean"]
        .iter()
        .map(|n| n.to_string())
        .collect();
    for i in 0..120 {
        let a = i as f32 / 120.0 * std::f32::consts::TAU;
        let a = NewAnalysis {
            content_hash: format!("h{i}"),
            meta: TrackMeta {
                title: format!("Song {i}"),
                album: None,
                year: None,
                duration_s: 200.0,
                mb_recording_id: None,
                artists: vec![ArtistCredit {
                    name: format!("Artist {}", i / 4),
                    role: ArtistRole::Primary,
                    mb_artist_id: None,
                }],
            },
            features: FeatureVec {
                kind: "dsp".into(),
                version: DSP_VERSION,
                names: names.clone(),
                values: vec![
                    a.cos(),
                    a.sin(),
                    (2.0 * a).cos() * 0.3,
                    (i as f32 * 0.7).sin() * 0.1,
                ],
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
fn eval_runs_on_simulated_history() {
    let mut store = library();
    let user = store.add_user("sim", 0.5).unwrap().id;
    let (rows, _, _) = simulate_spotify_history(
        &store,
        &SimConfig {
            days: 60,
            end_ts: 1_778_025_600,
            ..Default::default()
        },
    )
    .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("Streaming_History_Audio_sim.json");
    std::fs::write(&path, serde_json::to_string(&rows).unwrap()).unwrap();
    dhunn_import::import(&mut store, user, Service::Spotify, &path).unwrap();
    dhunn_import::matcher::match_history(&mut store).unwrap();

    let space = SoundSpace::from_store(&store, &AspectWeights::default()).unwrap();
    let steer = SessionConfig {
        artist_of: store.primary_artists().unwrap(),
        ..Default::default()
    };
    let rep = replay(
        &store,
        &space,
        user,
        &ReplayConfig {
            holdout_days: 14,
            session_gap_s: 1800,
            steer: steer.clone(),
        },
    )
    .unwrap()
    .expect("enough history");
    assert!(rep.steps > 20, "{rep:?}");
    let row = |n: &str| rep.rows.iter().find(|r| r.name == n).unwrap();
    // Sound-based models beat chance by a wide margin on a listener who drifts by sound.
    assert!(
        row("previous track").hit10 > 3.0 * row("random").hit10,
        "{rep:?}"
    );
    assert!(row("steer").hit10 > 3.0 * row("random").hit10, "{rep:?}");
    // Liked sound is listened to, disliked sound skipped.
    assert!(rep.skip_auc.iter().all(|(_, a)| *a > 0.55), "{rep:?}");

    let lis = simulated_listeners(
        &store,
        &space,
        &ListenerConfig {
            listeners: 4,
            sessions: 3,
            length: 15,
            steer,
            ..Default::default()
        },
    )
    .unwrap();
    let r = |n: &str| lis.rows.iter().find(|r| r.name == n).unwrap();
    assert_eq!(lis.rows.len(), 5);
    // Steering finds the listener's taste and keeps transitions smooth; random does neither.
    assert!(r("steer a=0.5").taste > r("random").taste + 15.0, "{lis:?}");
    assert!(r("steer a=0.5").skip < r("random").skip, "{lis:?}");
    assert!(
        r("steer a=0.5").smooth > r("random").smooth + 0.3,
        "{lis:?}"
    );
    // The artist-run limit holds: at most 2 in a row, so at most half the pairs match.
    assert!(r("steer a=0.5").same_artist <= 0.5, "{lis:?}");
    assert!(
        r("nearest chain").same_artist > r("steer a=0.5").same_artist,
        "{lis:?}"
    );

    let report = Report {
        created_at: 0,
        label: None,
        library_tracks: space.len(),
        user: Some("sim".into()),
        params: params_list(&SteerParams::default()),
        replay: Some(rep),
        listeners: Some(lis),
        logged: summarize(&store.logged_sessions(user).unwrap(), 0),
    };
    let text = render(&report, None);
    assert!(
        text.contains("Next-track prediction on sim's history"),
        "{text}"
    );
    assert!(text.contains("steer a=0.8"), "{text}");
    assert!(text.contains("none yet"), "{text}");
}
