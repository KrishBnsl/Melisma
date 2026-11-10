//! `dhunn` — command-line front end for the dhunn recommendation engine.

mod audio;
mod embed;
mod engine;
mod eval;
mod friends;
mod history;
mod library;
mod listen;
mod model;
mod profile;

use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use dhunn_core::ANALYSIS_SAMPLE_RATE;
use dhunn_dsp::{AnalyzeOptions, DspSummary, Stat};
use serde::Serialize;

#[derive(Parser)]
#[command(
    name = "dhunn",
    version,
    about = "Music recommendation engine that listens to the music"
)]
struct Cli {
    /// Library database file. Defaults to the platform data directory
    /// (macOS: ~/Library/Application Support/dhunn/dhunn.db).
    #[arg(long, global = true, env = "DHUNN_DB")]
    db: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Decode one audio file and print its DSP features.
    Inspect {
        /// Audio file (mp3, flac, wav, m4a, ogg, …).
        path: PathBuf,
        /// Print machine-readable JSON instead of a report.
        #[arg(long)]
        json: bool,
        /// Also print the flat feature vector (name = value per line, or in the JSON).
        #[arg(long)]
        vector: bool,
        /// Analyze the whole file even if it is longer than 10 minutes.
        #[arg(long)]
        full: bool,
        /// Include extra intermediate values used by `ml/validate_dsp.py` (slower).
        #[arg(long, hide = true)]
        validation: bool,
    },
    /// Scan a music folder and analyze new or changed tracks into the library.
    Analyze {
        /// Folder to scan recursively.
        dir: PathBuf,
        /// Re-analyze every file, even unchanged ones.
        #[arg(long)]
        force: bool,
        /// Tracks analyzed in parallel [default: half the CPU cores, max 6].
        #[arg(short, long)]
        jobs: Option<usize>,
        /// Only print the summary, not one line per file.
        #[arg(short, long)]
        quiet: bool,
    },
    /// Compute learned (CLAP) embeddings for analyzed tracks that don't have them yet.
    Embed {
        /// The exported CLAP model folder [default: $DHUNN_MODELS/clap, the data folder, or
        /// ./models/clap].
        #[arg(long)]
        models: Option<PathBuf>,
        /// Re-embed every track.
        #[arg(long)]
        force: bool,
        #[arg(short, long)]
        quiet: bool,
    },
    /// The CLAP model `embed` and vibe search use: where it is, and getting it.
    Model {
        #[command(subcommand)]
        action: Option<ModelAction>,
    },
    /// Show what is in the library.
    Stats {
        /// How many recent failures to list.
        #[arg(long, default_value_t = 10)]
        failures: usize,
    },
    /// Manage the people who use this library.
    User {
        #[command(subcommand)]
        action: history::UserAction,
    },
    /// Import listening history from a Spotify or Apple Music data export.
    Import {
        #[arg(value_enum)]
        service: history::ServiceArg,
        /// The unzipped export folder, or a single history file.
        path: PathBuf,
        #[arg(long, short)]
        user: String,
    },
    /// Summarize a user's imported listening history.
    History {
        #[arg(long, short)]
        user: String,
        /// How many top artists/tracks to show.
        #[arg(long, default_value_t = 10)]
        top: usize,
    },
    /// List the tracks that sound most like a song (search by title/artist words).
    Similar {
        query: String,
        #[arg(short, default_value_t = 10)]
        n: usize,
    },
    /// List the tracks that best match a description, e.g. "warm, slow and melancholic"
    /// (needs `dhunn embed`).
    Vibe {
        text: String,
        #[arg(short, default_value_t = 10)]
        n: usize,
    },
    /// Play audio for another program (Elsewhere on Linux and Windows): JSON requests on
    /// stdin, one reply per line on stdout.
    #[command(hide = true)]
    Audio,
    /// Run the recommendation engine for another player (such as Elsewhere): JSON
    /// requests on stdin, one reply per line on stdout.
    Engine {
        #[command(flatten)]
        steer: listen::SteerArgs,
    },
    /// Print a planned session without playing it.
    Session {
        #[command(flatten)]
        steer: listen::SteerArgs,
        #[arg(long, default_value_t = 15)]
        len: usize,
        /// Random seed, for reproducible sessions.
        #[arg(long, default_value_t = 1)]
        rng: u64,
    },
    /// Listen: plays a steered session with keyboard controls; every action is logged.
    Play {
        #[command(flatten)]
        steer: listen::SteerArgs,
        /// Non-interactive audio check: play N seconds of 3 tracks.
        #[arg(long, hide = true)]
        smoke: Option<u64>,
    },
    /// Who shares their taste with you, how close it is, and what they love that you've
    /// never played.
    Friends {
        /// Who is listening [default: $DHUNN_USER, or the only user].
        #[arg(long, short, env = "DHUNN_USER")]
        user: Option<String>,
        #[arg(short, default_value_t = 10)]
        n: usize,
    },
    /// Show what dhunn has learned about you: taste regions and the skip model.
    Profile {
        /// Who is listening [default: $DHUNN_USER, or the only user].
        #[arg(long, short, env = "DHUNN_USER")]
        user: Option<String>,
    },
    /// Measure recommendation quality: next-track prediction on your history, simulated
    /// listeners, and your logged sessions. Each run is saved and compared with the last.
    Eval {
        /// Who is listening [default: $DHUNN_USER, or the only user].
        #[arg(long, short, env = "DHUNN_USER")]
        user: Option<String>,
        /// Hold out the most recent N days of history for testing.
        #[arg(long, default_value_t = 14)]
        holdout_days: u32,
        /// Simulated listeners (0 to skip the simulation).
        #[arg(long, default_value_t = 16)]
        listeners: usize,
        /// Sessions per simulated listener.
        #[arg(long, default_value_t = 6)]
        sessions: usize,
        /// Tracks per simulated session.
        #[arg(long, default_value_t = 25)]
        len: usize,
        /// Try a steering parameter: name=value (repeatable), e.g. smoothness=0.5.
        #[arg(long = "param")]
        params: Vec<String>,
        /// Name this run (shown when later runs compare against it).
        #[arg(long)]
        label: Option<String>,
        /// Share of "sounds like" from learned (CLAP) embeddings, 0..1 [default: 0.25 when
        /// embeddings exist].
        #[arg(long)]
        clap_share: Option<f32>,
        /// Define the simulated listeners' tastes in DSP features or CLAP embeddings.
        #[arg(long, value_enum, default_value_t = JudgeArg::Dsp)]
        judge: JudgeArg,
        /// Don't save this run.
        #[arg(long)]
        no_save: bool,
        #[arg(long)]
        json: bool,
    },
    /// Developer tools.
    Dev {
        #[command(subcommand)]
        action: history::DevAction,
    },
}

#[derive(Subcommand)]
enum ModelAction {
    /// Download the ready-made model (about 790 MB) and check every file.
    Download,
    /// Install an export made with `uv run ml/export_clap.py --out <folder>`.
    Install {
        folder: PathBuf,
        /// Link to the folder instead of copying it.
        #[arg(long)]
        link: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum JudgeArg {
    Dsp,
    Clap,
}

#[derive(Serialize)]
struct InspectOutput<'a> {
    path: &'a str,
    source_sample_rate: u32,
    decode_ms: f64,
    analyze_ms: f64,
    summary: &'a DspSummary,
    diagnostics: &'a dhunn_dsp::Diagnostics,
    #[serde(skip_serializing_if = "Option::is_none")]
    vector: Option<dhunn_core::FeatureVec>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let db = cli.db.clone();
    match cli.command {
        Command::Analyze {
            dir,
            force,
            jobs,
            quiet,
        } => library::analyze(&library::db_path(db)?, &dir, force, jobs, quiet),
        Command::Embed {
            models,
            force,
            quiet,
        } => embed::embed(&library::db_path(db)?, models.as_deref(), force, quiet),
        Command::Model { action } => match action {
            None => model::status(),
            Some(ModelAction::Download) => model::download(),
            Some(ModelAction::Install { folder, link }) => model::install(&folder, link),
        },
        Command::Stats { failures } => library::stats(&library::db_path(db)?, failures),
        Command::User { action } => history::user(&library::db_path(db)?, action),
        Command::Import {
            service,
            path,
            user,
        } => history::import(&library::db_path(db)?, service, &path, &user),
        Command::History { user, top } => history::history(&library::db_path(db)?, &user, top),
        Command::Dev { action } => history::dev(&library::db_path(db)?, action),
        Command::Friends { user, n } => {
            friends::friends(&library::db_path(db)?, user.as_deref(), n)
        }
        Command::Profile { user } => profile::profile(&library::db_path(db)?, user.as_deref()),
        Command::Eval {
            user,
            holdout_days,
            listeners,
            sessions,
            len,
            params,
            label,
            clap_share,
            judge,
            no_save,
            json,
        } => eval::eval(
            &library::db_path(db)?,
            eval::EvalOptions {
                user: user.as_deref(),
                holdout_days,
                listeners,
                sessions,
                len,
                params: &params,
                label: label.as_deref(),
                clap_share,
                judge: match judge {
                    JudgeArg::Dsp => dhunn_eval::TasteSpace::Dsp,
                    JudgeArg::Clap => dhunn_eval::TasteSpace::Clap,
                },
                save: !no_save,
                json,
            },
        ),
        Command::Vibe { text, n } => listen::vibe(&library::db_path(db)?, &text, n),
        Command::Similar { query, n } => listen::similar(&library::db_path(db)?, &query, n),
        Command::Audio => audio::run(),
        Command::Engine { steer } => engine::run(&library::db_path(db)?, &steer),
        Command::Session { steer, len, rng } => {
            listen::session(&library::db_path(db)?, &steer, len, rng)
        }
        Command::Play { steer, smoke } => listen::play(&library::db_path(db)?, &steer, smoke),
        Command::Inspect {
            path,
            json,
            vector,
            full,
            validation,
        } => inspect(&path, json, vector, full, validation),
    }
}

fn inspect(path: &Path, json: bool, vector: bool, full: bool, validation: bool) -> Result<()> {
    let t0 = Instant::now();
    let decoded =
        dhunn_audio::decode_mono(path).with_context(|| format!("decoding {}", path.display()))?;
    let source_rate = decoded.sample_rate;
    let samples = dhunn_audio::resample(&decoded.samples, source_rate, ANALYSIS_SAMPLE_RATE)?;
    drop(decoded);
    let decode_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let opts = AnalyzeOptions {
        max_seconds: if full {
            None
        } else {
            AnalyzeOptions::default().max_seconds
        },
        validation_diagnostics: validation,
    };
    let (summary, diag) = dhunn_dsp::analyze(&samples, ANALYSIS_SAMPLE_RATE, &opts)
        .with_context(|| format!("analyzing {}", path.display()))?;
    let analyze_ms = t1.elapsed().as_secs_f64() * 1e3;

    if json {
        let out = InspectOutput {
            path: &path.to_string_lossy(),
            source_sample_rate: source_rate,
            decode_ms,
            analyze_ms,
            summary: &summary,
            diagnostics: &diag,
            vector: vector.then(|| summary.feature_vec()),
        };
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    print_report(path, source_rate, decode_ms, analyze_ms, &summary, &diag);
    if vector {
        let v = summary.feature_vec();
        println!(
            "\nFeature vector ({} dims, {} v{})",
            v.len(),
            v.kind,
            v.version
        );
        for (n, x) in v.names.iter().zip(&v.values) {
            println!("  {n:<22} {x:>12.5}");
        }
    }
    Ok(())
}

fn ms(s: Stat) -> String {
    format!("{:>9.2} ± {:<8.2}", s.mean, s.std)
}

fn print_report(
    path: &std::path::Path,
    source_rate: u32,
    decode_ms: f64,
    analyze_ms: f64,
    s: &DspSummary,
    d: &dhunn_dsp::Diagnostics,
) {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy())
        .unwrap_or_default();
    println!("{name}");
    println!(
        "  {:.1} s at {} Hz (analyzed {:.1} s) · decode {:.0} ms · analysis {:.0} ms · {} frames",
        s.duration_s, source_rate, s.analyzed_s, decode_ms, analyze_ms, d.n_frames
    );

    println!("\nRhythm");
    println!("  tempo            {:>9.1} BPM", s.rhythm.tempo_bpm);
    println!("  pulse clarity    {:>9.2}", s.rhythm.pulse_clarity);

    println!("\nHarmony");
    println!(
        "  key              {:>9}   (clarity {:.2})",
        s.harmony.key_name, s.harmony.key_clarity
    );
    let chroma: Vec<String> = dhunn_dsp::KEY_NAMES
        .iter()
        .zip(&s.harmony.chroma_mean)
        .map(|(n, v)| format!("{n}:{:.2}", v))
        .collect();
    println!("  chroma           {}", chroma.join(" "));

    println!("\nSpectral (brightness / treble)");
    println!("  centroid         {} Hz", ms(s.spectral.centroid_hz));
    println!("  rolloff (85%)    {} Hz", ms(s.spectral.rolloff_hz));
    println!("  bandwidth        {} Hz", ms(s.spectral.bandwidth_hz));
    println!("  flatness         {}", ms(s.spectral.flatness));
    println!("  zero-crossings   {}", ms(s.spectral.zcr));
    let contrast: Vec<String> = s
        .spectral
        .contrast_db
        .iter()
        .map(|c| format!("{:.1}", c.mean))
        .collect();
    println!("  contrast (dB)    {}", contrast.join(" "));

    println!("\nPercussion / low end");
    println!(
        "  harmonic ratio   {:>9.2}   (1 = all sustained, 0 = all percussive)",
        s.percussion.harmonic_ratio
    );
    println!("  onset strength   {}", ms(s.percussion.onset));
    println!(
        "  thump ratio      {:>9.3}   (percussive energy < 150 Hz)",
        s.percussion.thump_ratio
    );
    println!(
        "  thump flux       {} dB   p90 {:.2}",
        ms(s.percussion.thump_flux),
        s.percussion.thump_flux_p90
    );

    println!("\nDynamics");
    println!("  loudness         {} dBFS", ms(s.dynamics.rms_db));
    println!("  dynamic range    {:>9.1} dB", s.dynamics.dynamic_range_db);

    println!("\nPitch (harmonic part)");
    if s.pitch.f0_median_hz > 0.0 {
        println!("  median f0        {:>9.1} Hz", s.pitch.f0_median_hz);
    } else {
        println!("  median f0              —");
    }
    println!(
        "  range            {:>9.1} semitones",
        s.pitch.f0_range_semitones
    );
    println!(
        "  voiced           {:>8.0} %",
        s.pitch.voiced_fraction * 100.0
    );
    println!("  periodicity      {:>9.2}", s.pitch.periodicity);

    println!("\nTimbre (MFCC means)");
    let m: Vec<String> = s
        .timbre
        .mfcc
        .iter()
        .take(13)
        .map(|c| format!("{:.1}", c.mean))
        .collect();
    println!("  {}", m.join(" "));

    let stages: Vec<String> = d
        .stages
        .iter()
        .map(|t| format!("{} {:.0}", t.stage, t.ms))
        .collect();
    println!("\nStage timings (ms): {}", stages.join(" · "));
}
