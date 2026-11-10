//! `dhunn embed`: learned (CLAP) embeddings for every analyzed track.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use dhunn_dsp::DSP_VERSION;
use dhunn_embed::{Clap, MODEL};
use rayon::prelude::*;

use crate::library::open;

/// The CLAP model folder: `--models`, else `$DHUNN_MODELS/clap`, else the data folder, else
/// `./models/clap`.
pub fn model_dir(explicit: Option<&Path>) -> PathBuf {
    match explicit {
        Some(p) => p.to_owned(),
        None => dhunn_embed::default_dir(dirs::data_dir().as_deref()),
    }
}

pub fn load_clap(explicit: Option<&Path>) -> Result<Clap> {
    let dir = model_dir(explicit);
    if explicit.is_none() && !dir.join("clap.json").exists() {
        anyhow::bail!(crate::model::missing_help());
    }
    Clap::load(&dir).with_context(|| format!("loading the CLAP model from {}", dir.display()))
}

/// The text encoder only (for vibe search), loaded once per run.
pub fn text_clap() -> Result<&'static Clap> {
    static CLAP: std::sync::OnceLock<Clap> = std::sync::OnceLock::new();
    if let Some(c) = CLAP.get() {
        return Ok(c);
    }
    let dir = model_dir(None);
    if !dir.join("clap.json").exists() {
        anyhow::bail!(crate::model::missing_help());
    }
    let c = Clap::load_text(&dir)
        .with_context(|| format!("loading the CLAP text model from {}", dir.display()))?;
    Ok(CLAP.get_or_init(|| c))
}

/// Tracks decoded at once (decoding is parallel; the model then runs on the batch).
const BATCH: usize = 8;

pub fn embed(db: &Path, models: Option<&Path>, force: bool, quiet: bool) -> Result<()> {
    let mut store = open(db)?;
    let clap = load_clap(models)?;
    let version = clap.version();
    let mut todo: Vec<_> = store
        .track_ids_with_features("dsp", DSP_VERSION)?
        .into_iter()
        .collect();
    if !force {
        let done = store.tracks_with_embedding(MODEL, version)?;
        todo.retain(|t| !done.contains(t));
    }
    todo.sort();
    if todo.is_empty() {
        println!("All analyzed tracks already have {MODEL} embeddings.");
        return Ok(());
    }
    let paths: Vec<(dhunn_core::TrackId, Option<String>)> = todo
        .iter()
        .map(|&t| Ok((t, store.track_path(t)?)))
        .collect::<Result<_>>()?;
    println!(
        "Embedding {} tracks with {} …",
        paths.len(),
        clap.manifest.model
    );
    let start = Instant::now();
    let (mut done, mut failed) = (0usize, 0usize);
    for chunk in paths.chunks(BATCH) {
        let decoded: Vec<_> = chunk
            .par_iter()
            .map(|(t, p)| {
                let r = match p {
                    Some(p) => dhunn_audio::decode_mono(Path::new(p))
                        .map_err(anyhow::Error::from)
                        .and_then(|a| clap.clips_for(&a).map_err(anyhow::Error::from)),
                    None => Err(anyhow::anyhow!("no file on disk")),
                };
                (*t, p.clone(), r)
            })
            .collect();
        // One model run for the whole batch's clips.
        let mut clips = Vec::new();
        let mut owners = Vec::new();
        for (i, (_, p, r)) in decoded.iter().enumerate() {
            match r {
                Ok(c) => {
                    owners.extend(std::iter::repeat_n(i, c.len()));
                    clips.extend(c.iter().cloned());
                }
                Err(e) => {
                    failed += 1;
                    if !quiet {
                        println!("  ✗ {}: {e}", p.as_deref().unwrap_or("?"));
                    }
                }
            }
        }
        let embs = clap.embed_clips(&clips)?;
        let dims = clap.manifest.dims;
        for (i, (t, _, r)) in decoded.iter().enumerate() {
            if r.is_err() {
                continue;
            }
            let mut mean = vec![0f32; dims];
            for (e, _) in embs.iter().zip(&owners).filter(|(_, o)| **o == i) {
                mean.iter_mut().zip(e).for_each(|(m, x)| *m += x);
            }
            let n = mean.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            mean.iter_mut().for_each(|x| *x /= n);
            store.save_embedding(*t, MODEL, version, &mean)?;
            done += 1;
        }
        if !quiet {
            let rate = done as f64 / start.elapsed().as_secs_f64().max(1e-9);
            println!("  {done}/{} embedded ({rate:.1} tracks/s)", paths.len());
        }
    }
    println!(
        "Embedded {done} tracks in {:.0} s{}.",
        start.elapsed().as_secs_f64(),
        if failed > 0 {
            format!(" ({failed} could not be read)")
        } else {
            String::new()
        }
    );
    Ok(())
}
