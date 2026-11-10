//! `dhunn profile`: what dhunn has learned about a listener.

use std::path::Path;

use anyhow::{Result, bail};
use dhunn_index::SoundSpace;
use dhunn_profile::FEATURES;
use dhunn_store::{Store, User};

use crate::listen::{know, load, name, resolve_user};

pub fn profile(db: &Path, user: Option<&str>) -> Result<()> {
    let (store, space) = load(db)?;
    let Some(user) = resolve_user(&store, user)? else {
        bail!("which user? pass --user <name> (see `dhunn user list`)");
    };
    for line in profile_lines(&store, &space, &user)? {
        println!("{line}");
    }
    Ok(())
}

/// What dhunn has learned about `user`, as lines of text.
pub(crate) fn profile_lines(store: &Store, space: &SoundSpace, user: &User) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let Some(k) = know(store, space, Some(user))? else {
        out.push(format!(
            "{} has no plays of library tracks yet: import a history (`dhunn import`) or just listen.",
            user.name
        ));
        return Ok(out);
    };
    let p = &k.profile;
    out.push(format!(
        "{} · adventurousness {:.2} · {} plays of library tracks ({:.0}% listened, the rest skipped)",
        user.name,
        user.adventurousness,
        p.observations,
        p.listen_rate * 100.0
    ));
    out.push(String::new());
    out.push("Taste regions (recent plays count more; half-life 90 days):".into());
    for (i, r) in p.regions.iter().enumerate() {
        let tracks: Vec<String> = r.top.iter().take(3).map(|&t| name(store, t)).collect();
        out.push(format!(
            "  {}. {:>3.0}%  {}",
            i + 1,
            r.share * 100.0,
            tracks.join(" · ")
        ));
    }
    if !p.disliked.is_empty() {
        out.push(String::new());
        out.push(format!(
            "Never played again (disliked): {} tracks",
            p.disliked.len()
        ));
    }
    out.push(String::new());
    match &k.model {
        Some(m) => {
            out.push(format!(
                "Skip model: trained on {} plays, AUC {:.2} on them (held-out numbers: `dhunn eval`)",
                m.samples, m.train_auc
            ));
            let mut w: Vec<(&str, f32)> = FEATURES.iter().copied().zip(m.weights).collect();
            w.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
            out.push("  (+ = more likely you keep listening, − = more likely you skip)".into());
            for (f, x) in w.into_iter().filter(|(_, x)| x.abs() >= 0.05) {
                out.push(format!("  {f:<20} {x:+.2}"));
            }
        }
        None => out.push("Skip model: not enough plays with both listens and skips yet.".into()),
    }
    Ok(out)
}
