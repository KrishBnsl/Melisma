//! `dhunn user`, `dhunn import`, `dhunn history` and `dhunn dev`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Subcommand, ValueEnum};
use dhunn_import::{Service, time::format_timestamp};
use dhunn_store::{Store, User};

use crate::library::open;

#[derive(Subcommand)]
pub enum UserAction {
    /// Add a person.
    Add {
        name: String,
        /// 0 = stay close to what you know, 1 = explore boldly.
        #[arg(long, default_value_t = 0.5)]
        adventure: f32,
    },
    /// List everyone.
    List,
    /// Change how adventurous recommendations are for someone.
    Set {
        name: String,
        #[arg(long)]
        adventure: f32,
    },
}

#[derive(Clone, Copy, ValueEnum)]
pub enum ServiceArg {
    Spotify,
    Apple,
}

#[derive(Subcommand)]
pub enum DevAction {
    /// Write a simulated listener's history in Spotify's export format (plus the hidden
    /// taste as ground truth), using tracks from the library.
    SampleHistory {
        /// Output folder (created if missing).
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 120)]
        days: u32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Number of distinct taste regions.
        #[arg(long, default_value_t = 2)]
        regions: usize,
    },
}

fn require_user(store: &Store, name: &str) -> Result<User> {
    match store.user_by_name(name)? {
        Some(u) => Ok(u),
        None => bail!("no user named {name:?}; add one with `dhunn user add {name}`"),
    }
}

pub fn user(db: &Path, action: UserAction) -> Result<()> {
    let mut store = open(db)?;
    match action {
        UserAction::Add { name, adventure } => {
            let u = store.add_user(&name, adventure)?;
            println!(
                "Added {} (adventurousness {:.2})",
                u.name, u.adventurousness
            );
        }
        UserAction::List => {
            let users = store.users()?;
            if users.is_empty() {
                println!("No users yet. Add one with `dhunn user add <name>`.");
            }
            for u in users {
                let plays = store.history_summary(u.id, 0)?.plays;
                println!(
                    "{:<16} adventurousness {:.2}  {} imported plays{}",
                    u.name,
                    u.adventurousness,
                    plays,
                    if u.share_profile { "" } else { "  (private)" }
                );
            }
        }
        UserAction::Set { name, adventure } => {
            let u = require_user(&store, &name)?;
            store.set_adventurousness(u.id, adventure)?;
            println!(
                "{}: adventurousness {:.2}",
                u.name,
                adventure.clamp(0.0, 1.0)
            );
        }
    }
    Ok(())
}

pub fn import(db: &Path, service: ServiceArg, path: &Path, user: &str) -> Result<()> {
    let mut store = open(db)?;
    let u = require_user(&store, user)?;
    let service = match service {
        ServiceArg::Spotify => Service::Spotify,
        ServiceArg::Apple => Service::Apple,
    };
    let r = dhunn_import::import(&mut store, u.id, service, path)
        .with_context(|| format!("importing {}", path.display()))?;
    println!(
        "Read {} file(s): {} new plays, {} already imported, {} non-music entries skipped",
        r.files, r.new_plays, r.already_imported, r.skipped_non_music
    );
    if r.invalid > 0 {
        println!(
            "{} entries could not be read; first: {}",
            r.invalid,
            r.first_error.as_deref().unwrap_or("?")
        );
    }
    let m = r.matching;
    if m.pairs > 0 {
        println!(
            "Matched {} distinct songs: {} in library ({} exact, {} fuzzy), {} by artist only, {} not in library",
            m.pairs,
            m.exact + m.fuzzy,
            m.exact,
            m.fuzzy,
            m.artist,
            m.none
        );
    }
    println!("See `dhunn history --user {}`", u.name);
    Ok(())
}

fn date(ts: Option<i64>) -> String {
    ts.map(|t| format_timestamp(t)[..10].to_owned())
        .unwrap_or_else(|| "—".into())
}

pub fn history(db: &Path, user: &str, top: usize) -> Result<()> {
    let store = open(db)?;
    let u = require_user(&store, user)?;
    let h = store.history_summary(u.id, top)?;
    if h.plays == 0 {
        println!(
            "No history for {} yet. Import some with `dhunn import spotify <folder> --user {}`.",
            u.name, u.name
        );
        return Ok(());
    }
    let pct = |n: u64| 100.0 * n as f64 / h.plays as f64;
    println!(
        "{}: {} plays, {} → {}, {:.1} hours",
        u.name,
        h.plays,
        date(h.first_ts),
        date(h.last_ts),
        h.listened_hours
    );
    println!(
        "  skipped (per export)   {:>6}  ({:.0}%)",
        h.skipped,
        pct(h.skipped)
    );
    println!(
        "  in library             {:>6}  ({:.0}%: {} exact, {} fuzzy)",
        h.matched_exact + h.matched_fuzzy,
        pct(h.matched_exact + h.matched_fuzzy),
        h.matched_exact,
        h.matched_fuzzy
    );
    println!(
        "  artist in library only {:>6}  ({:.0}%)",
        h.matched_artist,
        pct(h.matched_artist)
    );
    println!(
        "  not in library         {:>6}  ({:.0}%)",
        h.unmatched,
        pct(h.unmatched)
    );
    println!("\nTop artists (listens = plays not skipped)");
    for (a, n) in &h.top_artists {
        println!("  {n:>5}  {a}");
    }
    println!("\nTop tracks (✓ = in library)");
    for (t, n, owned) in &h.top_tracks {
        println!("  {n:>5}  {} {t}", if *owned { "✓" } else { " " });
    }
    Ok(())
}

pub fn dev(db: &Path, action: DevAction) -> Result<()> {
    match action {
        DevAction::SampleHistory {
            out,
            days,
            seed,
            regions,
        } => {
            let store = open(db)?;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs() as i64;
            let cfg = dhunn_eval::SimConfig {
                seed,
                days,
                end_ts: now,
                taste_regions: regions,
                ..Default::default()
            };
            let (rows, truth, stats) = dhunn_eval::simulate_spotify_history(&store, &cfg)?;
            std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;
            let first = rows.first().map(|r| &r.ts[..4]).unwrap_or("0000");
            let last = rows.last().map(|r| &r.ts[..4]).unwrap_or("0000");
            let history = out.join(format!("Streaming_History_Audio_{first}-{last}_0.json"));
            std::fs::write(&history, serde_json::to_string_pretty(&rows)?)?;
            let truth_path = out.join("dhunn_ground_truth.json");
            std::fs::write(&truth_path, serde_json::to_string_pretty(&truth)?)?;
            println!(
                "Wrote {} plays ({} of library tracks, {} of songs not in the library, {} skipped, {} re-spelled)",
                stats.plays, stats.library_plays, stats.unowned_plays, stats.skips, stats.respelled
            );
            println!("  history:      {}", history.display());
            println!(
                "  ground truth: {}  (the listener's hidden taste)",
                truth_path.display()
            );
            println!("Hidden taste centered on:");
            for h in &truth.home_tracks {
                println!("  {} – {}", h.artist.as_deref().unwrap_or("?"), h.title);
            }
        }
    }
    Ok(())
}
