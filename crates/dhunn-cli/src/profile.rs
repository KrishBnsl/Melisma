//! `dhunn profile`: what dhunn has learned about a listener.

use std::path::Path;

use anyhow::{Result, bail};
use dhunn_profile::FEATURES;

use crate::listen::{know, load, name, resolve_user};

pub fn profile(db: &Path, user: Option<&str>) -> Result<()> {
    let (store, space) = load(db)?;
    let Some(user) = resolve_user(&store, user)? else {
        bail!("which user? pass --user <name> (see `dhunn user list`)");
    };
    let Some(k) = know(&store, &space, Some(&user))? else {
        println!(
            "{} has no plays of library tracks yet: import a history (`dhunn import`) or listen with `dhunn play`.",
            user.name
        );
        return Ok(());
    };
    let p = &k.profile;
    println!(
        "{} · adventurousness {:.2} · {} plays of library tracks ({:.0}% listened, the rest skipped)",
        user.name,
        user.adventurousness,
        p.observations,
        p.listen_rate * 100.0
    );
    println!("\nTaste regions (recent plays count more; half-life 90 days):");
    for (i, r) in p.regions.iter().enumerate() {
        let tracks: Vec<String> = r.top.iter().take(3).map(|&t| name(&store, t)).collect();
        println!(
            "  {}. {:>3.0}%  {}",
            i + 1,
            r.share * 100.0,
            tracks.join(" · ")
        );
    }
    if !p.disliked.is_empty() {
        println!(
            "\nNever played again (disliked): {} tracks",
            p.disliked.len()
        );
    }
    match &k.model {
        Some(m) => {
            println!(
                "\nSkip model: trained on {} plays, AUC {:.2} on them (held-out numbers: `dhunn eval`)",
                m.samples, m.train_auc
            );
            let mut w: Vec<(&str, f32)> = FEATURES.iter().copied().zip(m.weights).collect();
            w.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
            println!("  (+ = more likely you keep listening, − = more likely you skip)");
            for (f, x) in w.into_iter().filter(|(_, x)| x.abs() >= 0.05) {
                println!("  {f:<20} {x:+.2}");
            }
        }
        None => println!("\nSkip model: not enough plays with both listens and skips yet."),
    }
    Ok(())
}
