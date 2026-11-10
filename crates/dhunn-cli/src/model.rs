//! `dhunn model`: where the CLAP model is, and getting it there. The model lives in one
//! place per user (`$DHUNN_MODELS/clap`, else the data folder), so `dhunn embed` and vibe
//! search work from any folder.
//!
//! - `dhunn model`: where it is, or where it was looked for.
//! - `dhunn model install <folder> [--link]`: copy (or link) an export from
//!   `uv run ml/export_clap.py --out <folder>` into place.
//! - `dhunn model download`: fetch the ready-made export (about 790 MB), checking every file.

use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

/// The files of the export, with their size and SHA-256.
const FILES: [(&str, u64, &str); 4] = [
    (
        "clap.json",
        726,
        "79ce8b6b6a88f211acb6de83a46795132feb3439e97bb2b269c566b814176188",
    ),
    (
        "tokenizer.json",
        3_558_802,
        "4fd1d86b4f5b53f40a609fcd11c1f34024b735f870a07439d70202b98493661a",
    ),
    (
        "audio.onnx",
        285_696_981,
        "659d48a58f18fa0431734a70d85c3ba44512ce06408aeee53cef68f1fb53d42f",
    ),
    (
        "text.onnx",
        501_450_191,
        "2392316b280249131c800de0c5fb014cfa4b4d2f2122fd5383e7106da051019a",
    ),
];

/// Where `dhunn model download` fetches from ($DHUNN_MODEL_URL overrides it).
const RELEASE: &str = "https://github.com/KrishBnsl/Melisma/releases/download/clap-v1";

/// Where the model belongs: `$DHUNN_MODELS/clap`, else `<data dir>/dhunn/models/clap`.
pub fn home() -> Result<PathBuf> {
    if let Some(m) = std::env::var_os("DHUNN_MODELS") {
        return Ok(PathBuf::from(m).join("clap"));
    }
    let data = dirs::data_dir().context("no data folder on this system; set DHUNN_MODELS")?;
    Ok(data.join("dhunn").join("models").join("clap"))
}

/// Every place the model is looked for, in order.
pub fn searched() -> Vec<PathBuf> {
    let mut places = Vec::new();
    if let Ok(h) = home() {
        places.push(h);
    }
    places.push(PathBuf::from("models").join("clap"));
    places
}

/// What to say when there's no model.
pub fn missing_help() -> String {
    let places: Vec<String> = searched().iter().map(|p| p.display().to_string()).collect();
    format!(
        "no CLAP model found (looked in {}).\n\
         Get one with `dhunn model download` (about 790 MB), or install your own export with\n\
         `dhunn model install <folder>` (from `uv run ml/export_clap.py --out <folder>`).",
        places.join(", ")
    )
}

fn has_model(dir: &Path) -> bool {
    FILES.iter().all(|(name, ..)| dir.join(name).is_file())
}

pub fn status() -> Result<()> {
    let found = searched().into_iter().find(|p| has_model(p));
    match found {
        Some(dir) => {
            let size: u64 = FILES
                .iter()
                .filter_map(|(name, ..)| fs::metadata(dir.join(name)).ok())
                .map(|m| m.len())
                .sum();
            let shown = fs::canonicalize(&dir).unwrap_or(dir.clone());
            println!("CLAP model: {} ({} MB)", shown.display(), size / 1_000_000);
            if dir.is_relative() {
                println!(
                    "It's only found from this folder. Make it work from anywhere with\n\
                     `dhunn model install {} --link`.",
                    dir.display()
                );
            }
        }
        None => println!("{}", missing_help()),
    }
    Ok(())
}

/// Copies (or links) an exported model into place, after checking it loads.
pub fn install(from: &Path, link: bool) -> Result<()> {
    if !has_model(from) {
        bail!(
            "{} isn't a CLAP export (it needs {})",
            from.display(),
            FILES.map(|f| f.0).join(", ")
        );
    }
    dhunn_embed::Clap::load_text(from)
        .with_context(|| format!("{} doesn't load as a CLAP model", from.display()))?;
    let to = home()?;
    let from = fs::canonicalize(from)?;
    if fs::canonicalize(&to).ok().as_ref() == Some(&from) {
        println!("The model in {} is already in place.", from.display());
        return Ok(());
    }
    replace_with(&to, |tmp| {
        if link {
            Ok(link_dir(&from, tmp)?)
        } else {
            fs::create_dir_all(tmp)?;
            for (name, ..) in FILES {
                fs::copy(from.join(name), tmp.join(name))?;
            }
            Ok(())
        }
    })?;
    let how = if link { "linked to" } else { "copied from" };
    println!(
        "CLAP model installed in {} ({how} {}).",
        to.display(),
        from.display()
    );
    Ok(())
}

#[cfg(unix)]
fn link_dir(from: &Path, to: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(from, to)
}

#[cfg(windows)]
fn link_dir(from: &Path, to: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_dir(from, to)
}

/// Builds the new model next to `to` with `make`, then swaps it in (keeping the old one until
/// the new one is complete).
fn replace_with(to: &Path, make: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    let parent = to.parent().context("no parent folder")?;
    fs::create_dir_all(parent)?;
    let tmp = parent.join("clap.new");
    remove(&tmp)?;
    if let Err(e) = make(&tmp) {
        let _ = remove(&tmp);
        return Err(e);
    }
    let old = parent.join("clap.old");
    remove(&old)?;
    if to.symlink_metadata().is_ok() {
        fs::rename(to, &old)?;
    }
    fs::rename(&tmp, to)?;
    remove(&old)?;
    Ok(())
}

/// Removes a file, folder or link, if it's there.
fn remove(p: &Path) -> Result<()> {
    match p.symlink_metadata() {
        Ok(m) if m.is_dir() => fs::remove_dir_all(p)?,
        Ok(_) => fs::remove_file(p)?,
        Err(_) => {}
    }
    Ok(())
}

/// Downloads the ready-made export, checking each file's size and SHA-256.
pub fn download() -> Result<()> {
    let to = home()?;
    if has_model(&to) && verify(&to).is_ok() {
        println!("The CLAP model is already in {}.", to.display());
        return Ok(());
    }
    let base = std::env::var("DHUNN_MODEL_URL").unwrap_or_else(|_| RELEASE.to_owned());
    let total: u64 = FILES.iter().map(|f| f.1).sum();
    println!(
        "Downloading the CLAP model ({} MB) into {}",
        total / 1_000_000,
        to.display()
    );
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(15)))
        .timeout_recv_response(Some(Duration::from_secs(30)))
        .user_agent("dhunn (https://github.com/KrishBnsl/Melisma)")
        .build()
        .into();
    replace_with(&to, |tmp| {
        fs::create_dir_all(tmp)?;
        let mut done = 0u64;
        let start = Instant::now();
        for (name, size, sum) in FILES {
            let url = format!("{base}/{name}");
            let resp = agent
                .get(&url)
                .call()
                .with_context(|| format!("downloading {url}"))?;
            let mut body = resp.into_body().into_reader();
            let mut out = File::create(tmp.join(name))?;
            let mut hasher = Sha256::new();
            let mut buf = vec![0u8; 1 << 16];
            let (mut got, mut shown) = (0u64, 0u64);
            loop {
                let n = body.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                out.write_all(&buf[..n])?;
                hasher.update(&buf[..n]);
                got += n as u64;
                if got > size {
                    bail!("{name} is bigger than expected; the download isn't the right file");
                }
                let pct = (done + got) * 100 / total;
                if pct >= shown + 5 || done + got == total {
                    shown = pct;
                    let mb_s = (done + got) as f64 / 1e6 / start.elapsed().as_secs_f64().max(0.1);
                    print!(
                        "\r  {pct:>3}%  {:>4} MB  ({mb_s:.0} MB/s)   ",
                        (done + got) / 1_000_000
                    );
                    std::io::stdout().flush().ok();
                }
            }
            if got != size {
                bail!("{name} was cut short ({got} of {size} bytes); try again");
            }
            if hex(&hasher.finalize()) != sum {
                bail!("{name} is damaged (checksum mismatch); try again");
            }
            done += got;
        }
        println!();
        dhunn_embed::Clap::load_text(tmp).context("the downloaded model doesn't load")?;
        Ok(())
    })?;
    println!("CLAP model installed. Next: `dhunn embed` (vibe search then works too).");
    Ok(())
}

/// Checks each file's size and checksum.
fn verify(dir: &Path) -> Result<()> {
    for (name, size, sum) in FILES {
        let path = dir.join(name);
        if fs::metadata(&path)?.len() != size {
            bail!("{name} has the wrong size");
        }
        let mut f = BufReader::new(File::open(&path)?);
        let mut hasher = Sha256::new();
        std::io::copy(&mut f, &mut hasher)?;
        if hex(&hasher.finalize()) != sum {
            bail!("{name} doesn't match");
        }
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swapping_in_a_folder_keeps_nothing_half_done() {
        let dir = tempfile::tempdir().unwrap();
        let to = dir.path().join("models").join("clap");
        replace_with(&to, |tmp| {
            fs::create_dir_all(tmp)?;
            fs::write(tmp.join("a"), "1")?;
            Ok(())
        })
        .unwrap();
        assert_eq!(fs::read_to_string(to.join("a")).unwrap(), "1");
        // A failed replacement leaves the old one alone.
        let failed = replace_with(&to, |tmp| {
            fs::create_dir_all(tmp)?;
            bail!("network went away")
        });
        assert!(failed.is_err());
        assert_eq!(fs::read_to_string(to.join("a")).unwrap(), "1");
        assert!(!dir.path().join("models").join("clap.new").exists());
    }

    #[test]
    fn hex_is_lowercase() {
        assert_eq!(hex(&[0x0f, 0xa0]), "0fa0");
        assert_eq!(
            hex(&Sha256::digest(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
