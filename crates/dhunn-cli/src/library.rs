//! `dhunn analyze` and `dhunn stats`.

use std::{
    io::{IsTerminal, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use dhunn_analysis::{FileOutcome, ScanEvent, ScanOptions};
use dhunn_store::Store;

pub fn db_path(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p);
    }
    let base = dirs::data_dir().context("no data directory on this platform; pass --db")?;
    Ok(base.join("dhunn").join("dhunn.db"))
}

pub fn open(db: &Path) -> Result<Store> {
    Store::open(db).with_context(|| format!("opening library {}", db.display()))
}

fn short(path: &str, root: &str) -> String {
    path.strip_prefix(root)
        .map(|p| p.trim_start_matches('/'))
        .unwrap_or(path)
        .to_owned()
}

pub fn analyze(db: &Path, dir: &Path, force: bool, jobs: Option<usize>, quiet: bool) -> Result<()> {
    let mut store = open(db)?;
    let opts = ScanOptions {
        force,
        jobs: jobs.unwrap_or_else(dhunn_analysis::default_jobs),
        ..Default::default()
    };
    let root = dir
        .canonicalize()
        .with_context(|| format!("opening {}", dir.display()))?;
    let root_str = root.to_string_lossy().into_owned();
    let tty = std::io::stderr().is_terminal();
    let mut err = std::io::stderr();
    eprintln!("Library: {}", db.display());

    let report = dhunn_analysis::scan(&mut store, &root, &opts, |ev| match ev {
        ScanEvent::Planned { found, to_process } => {
            eprintln!(
                "Found {found} audio files in {}; {to_process} to process with {} workers",
                root.display(),
                opts.jobs
            );
        }
        ScanEvent::File {
            done,
            total,
            path,
            outcome,
        } => {
            let name = short(path, &root_str);
            let line = match outcome {
                FileOutcome::Analyzed { meta, seconds } => {
                    let artist = meta
                        .artists
                        .first()
                        .map(|a| format!("{} – ", a.name))
                        .unwrap_or_default();
                    format!("analyzed  {artist}{}  ({seconds:.1}s)", meta.title)
                }
                FileOutcome::Reused => format!("reused    {name}"),
                FileOutcome::Failed { error } => format!("FAILED    {name}: {error}"),
            };
            let failed = matches!(outcome, FileOutcome::Failed { .. });
            if !quiet || failed {
                let _ = writeln!(err, "[{done:>5}/{total}] {line}");
            } else if tty {
                let _ = write!(err, "\r[{done:>5}/{total}]");
                let _ = err.flush();
            }
        }
    })?;
    if quiet && tty {
        eprintln!();
    }

    let r = &report;
    println!(
        "\nDone in {:.1}s: {} analyzed, {} reused, {} unchanged, {} failed{}{}{}",
        r.elapsed.as_secs_f64(),
        r.analyzed,
        r.reused,
        r.unchanged,
        r.failed,
        if r.skipped_failed > 0 {
            format!(", {} skipped (failed before)", r.skipped_failed)
        } else {
            String::new()
        },
        if r.removed > 0 {
            format!(", {} removed (gone from disk)", r.removed)
        } else {
            String::new()
        },
        if r.unreadable > 0 {
            format!(", {} unreadable entries", r.unreadable)
        } else {
            String::new()
        },
    );
    if !store.users()?.is_empty() {
        let m = dhunn_import::match_history(&mut store)?;
        if m.exact + m.fuzzy > 0 {
            println!(
                "Matched {} more listening-history entries to library tracks",
                m.exact + m.fuzzy
            );
        }
    }
    if r.analyzed > 0 {
        let secs = r.elapsed.as_secs_f64() / r.analyzed as f64;
        println!(
            "≈ {secs:.2}s per analyzed track (wall clock, {} workers)",
            opts.jobs
        );
    }
    print_totals(&store)?;
    Ok(())
}

fn print_totals(store: &Store) -> Result<()> {
    let s = store.stats()?;
    println!(
        "Library: {} tracks in {} files, {} artists",
        s.tracks, s.files, s.artists
    );
    for (kind, version, n) in &s.features {
        println!("  features {kind} v{version}: {n} tracks");
    }
    if s.failures > 0 {
        println!(
            "  {} files failed to analyze (see `dhunn stats`)",
            s.failures
        );
    }
    Ok(())
}

pub fn stats(db: &Path, failures: usize) -> Result<()> {
    if !db.exists() {
        println!(
            "No library yet at {}. Run `dhunn analyze <folder>` first.",
            db.display()
        );
        return Ok(());
    }
    let store = open(db)?;
    println!("Library: {}", db.display());
    print_totals(&store)?;
    let list = store.failures(failures)?;
    if !list.is_empty() {
        println!("\nRecent failures:");
        for f in list {
            println!("  {}\n    {}", f.path, f.error);
        }
    }
    Ok(())
}
