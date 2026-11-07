//! `dhunn eval`: how well recommendations work, compared with the last run.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use dhunn_eval::{
    ListenerConfig, ReplayConfig, Report, params_list, render, replay, simulated_listeners,
    summarize,
};
use dhunn_steer::{SessionConfig, SteerParams};

use crate::listen::{load, resolve_user, song_keys};

pub struct EvalOptions<'a> {
    pub user: Option<&'a str>,
    pub holdout_days: u32,
    pub listeners: usize,
    pub sessions: usize,
    pub len: usize,
    /// `name=value` steering overrides.
    pub params: &'a [String],
    pub label: Option<&'a str>,
    pub save: bool,
    pub json: bool,
}

fn parse_params(items: &[String]) -> Result<SteerParams> {
    let mut p = SteerParams::default();
    for item in items {
        let Some((k, v)) = item.split_once('=') else {
            bail!("--param takes name=value, got {item:?}");
        };
        let v: f32 = v
            .trim()
            .parse()
            .with_context(|| format!("--param {k}: {v:?} is not a number"))?;
        p.set(k.trim(), v).map_err(anyhow::Error::msg)?;
    }
    Ok(p)
}

pub fn eval(db: &Path, opts: EvalOptions<'_>) -> Result<()> {
    let params = parse_params(opts.params)?;
    let (mut store, space) = load(db)?;
    let user = resolve_user(&store, opts.user)?;
    let steer = SessionConfig {
        adventurousness: user.as_ref().map_or(0.5, |u| u.adventurousness),
        artist_of: store.primary_artists()?,
        song_of: song_keys(&store)?,
        params,
        ..Default::default()
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    let replay = match &user {
        Some(u) => replay(
            &store,
            &space,
            u.id,
            &ReplayConfig {
                holdout_days: opts.holdout_days,
                session_gap_s: 30 * 60,
                steer: steer.clone(),
            },
        )?,
        None => None,
    };
    let listeners = (opts.listeners > 0 && opts.sessions > 0)
        .then(|| {
            simulated_listeners(
                &store,
                &space,
                &ListenerConfig {
                    listeners: opts.listeners,
                    sessions: opts.sessions,
                    length: opts.len,
                    steer: steer.clone(),
                    ..Default::default()
                },
            )
        })
        .transpose()?;
    let logged = match &user {
        Some(u) => summarize(&store.logged_sessions(u.id)?, now),
        None => None,
    };
    let report = Report {
        created_at: now,
        label: opts.label.map(str::to_owned),
        library_tracks: space.len(),
        user: user.as_ref().map(|u| u.name.clone()),
        params: params_list(&params),
        replay,
        listeners,
        logged,
    };

    let previous: Option<Report> = store
        .last_eval_run()?
        .and_then(|r| serde_json::from_str(&r.report_json).ok());
    if opts.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", render(&report, previous.as_ref()));
        if let Some(p) = &previous {
            let when = dhunn_import::time::format_timestamp(p.created_at);
            println!(
                "\n(Brackets: change since the last run, {}{}.)",
                when.replace('T', " ").trim_end_matches('Z'),
                p.label
                    .as_deref()
                    .map(|l| format!(" \"{l}\""))
                    .unwrap_or_default()
            );
        }
    }
    if opts.save {
        store.save_eval_run(opts.label, &serde_json::to_string(&report)?)?;
    }
    Ok(())
}
