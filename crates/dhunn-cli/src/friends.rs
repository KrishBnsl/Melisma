//! Friends: whose taste dhunn may use, `dhunn friends`, and the pieces sessions need.

use std::path::Path;

use anyhow::{Result, bail};
use dhunn_core::{TrackId, normalize_name};
use dhunn_index::SoundSpace;
use dhunn_profile::{
    Friend, Obs, Profile, ProfileConfig, friend_signal, history_sessions, logged_sessions, overlap,
    shared_sessions,
};
use dhunn_store::{Store, User};

use crate::listen::{facts, load, name, now_ms, resolve_user};

/// A user's listening as sessions: imported history plus `dhunn play` sessions (without
/// `--private` ones when the data is for someone else's recommendations).
pub(crate) fn sessions_of(
    store: &Store,
    user: &User,
    include_private: bool,
) -> Result<Vec<Vec<Obs>>> {
    let mut sessions = history_sessions(&store.matched_plays(user.id)?, 30 * 60);
    let logged = store.logged_sessions(user.id)?;
    sessions.extend(if include_private {
        logged_sessions(&logged)
    } else {
        shared_sessions(&logged)
    });
    Ok(sessions)
}

pub(crate) fn profile_from(
    store: &Store,
    space: &SoundSpace,
    sessions: &[Vec<Obs>],
) -> Result<Profile> {
    Ok(Profile::build(
        space,
        sessions,
        &store.primary_artists()?,
        now_ms() / 1000,
        &ProfileConfig::default(),
    ))
}

/// Everyone else who shares their profile and has listened to something, with how much
/// their taste overlaps `mine`.
pub(crate) fn friends_of(
    store: &Store,
    space: &SoundSpace,
    me: &User,
    mine: Option<&Profile>,
) -> Result<Vec<Friend>> {
    let mut out = Vec::new();
    for u in store.users()? {
        if u.id == me.id || !u.share_profile {
            continue;
        }
        let sessions = sessions_of(store, &u, false)?;
        if sessions.is_empty() {
            continue;
        }
        let profile = profile_from(store, space, &sessions)?;
        if profile.regions.is_empty() {
            continue;
        }
        let o = mine.map_or(0.0, |m| overlap(m, &profile));
        out.push(Friend {
            name: u.name.clone(),
            profile,
            overlap: o,
        });
    }
    out.sort_by(|a, b| b.overlap.total_cmp(&a.overlap));
    Ok(out)
}

/// A friend by name, with a helpful error if they don't exist or don't share.
pub(crate) fn find_friend<'a>(
    store: &Store,
    friends: &'a [Friend],
    who: &str,
) -> Result<&'a Friend> {
    let want = normalize_name(who);
    if let Some(f) = friends.iter().find(|f| normalize_name(&f.name) == want) {
        return Ok(f);
    }
    match store.user_by_name(who)? {
        None => bail!("no user named {who:?} (see `dhunn user list`)"),
        Some(u) if !u.share_profile => {
            bail!(
                "{} doesn't share their taste (`dhunn user set {} --share yes` turns it on)",
                u.name,
                u.name
            )
        }
        Some(u) => bail!("{} hasn't listened to anything in this library yet", u.name),
    }
}

/// "Aarav plays this a lot" / "Aarav likes this".
pub(crate) fn friend_text(friends: &[Friend], who: u32, strength: f32) -> Option<String> {
    let f = friends.get(who as usize)?;
    Some(if strength >= 0.6 {
        format!("{} plays this a lot", f.name)
    } else {
        format!("{} likes this", f.name)
    })
}

/// `dhunn friends`: who shares their taste, how close it is to yours, and what they love that
/// you've never played.
pub fn friends(db: &Path, user: Option<&str>, n: usize) -> Result<()> {
    let (store, space) = load(db)?;
    let Some(me) = resolve_user(&store, user)? else {
        bail!("which user? pass --user <name> (see `dhunn user list`)");
    };
    let mine_sessions = sessions_of(&store, &me, true)?;
    let mine = (!mine_sessions.is_empty())
        .then(|| profile_from(&store, &space, &mine_sessions))
        .transpose()?;
    let friends = friends_of(&store, &space, &me, mine.as_ref())?;
    if friends.is_empty() {
        println!(
            "No friends sharing their taste yet. Add someone with `dhunn user add <name>` and import their history."
        );
        return Ok(());
    }
    println!("Taste overlap with {}:", me.name);
    for f in &friends {
        println!("  {:<16} {:>3.0}%", f.name, f.overlap * 100.0);
    }
    let known: std::collections::HashSet<TrackId> = mine
        .as_ref()
        .map(|p| p.tracks.keys().copied().collect())
        .unwrap_or_default();
    // Friend-sourced novelty: what friends love that I've never played, ranked by friend
    // signal and closeness to my taste.
    let mut picks: Vec<(TrackId, f32, u32)> = friend_signal(&friends)
        .into_iter()
        .filter(|(t, (s, _))| *s > 0.0 && !known.contains(t) && space.contains(*t))
        .map(|(t, (s, who))| {
            let close = mine
                .as_ref()
                .and_then(|p| space.vector(t).and_then(|v| p.taste_of(v)))
                .unwrap_or(0.0);
            (t, s * (1.0 + close.max(0.0)) / 2.0, who)
        })
        .collect();
    picks.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if picks.is_empty() {
        println!("\nYou've already played everything your friends love.");
        return Ok(());
    }
    println!("\nFrom your friends — loved by them, new to you:");
    for (t, _, who) in picks.into_iter().take(n) {
        println!(
            "  {}  [{}]  — {}",
            name(&store, t),
            facts(&store, t),
            friends[who as usize].name
        );
    }
    Ok(())
}
