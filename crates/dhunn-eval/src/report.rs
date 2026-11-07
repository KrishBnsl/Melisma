//! The combined `dhunn eval` report and its text rendering (with changes since the last run).

use dhunn_steer::SteerParams;
use serde::{Deserialize, Serialize};

use crate::friends::FriendReport;
use crate::listeners::ListenerReport;
use crate::logged::{Counts, LoggedReport};
use crate::replay::ReplayReport;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    /// Unix seconds.
    pub created_at: i64,
    pub label: Option<String>,
    pub library_tracks: usize,
    /// Share of "sounds like" from learned embeddings (0 = DSP only).
    #[serde(default)]
    pub clap_share: f32,
    pub user: Option<String>,
    pub params: Vec<(String, f32)>,
    pub replay: Option<ReplayReport>,
    pub listeners: Option<ListenerReport>,
    #[serde(default)]
    pub friends: Option<FriendReport>,
    pub logged: Option<LoggedReport>,
}

pub fn params_list(p: &SteerParams) -> Vec<(String, f32)> {
    vec![
        ("smoothness".into(), p.smoothness),
        ("onward".into(), p.onward),
        ("heard_memory".into(), p.heard_memory as f32),
        ("avoid_penalty".into(), p.avoid_penalty),
        ("finish_pull".into(), p.finish_pull),
        ("like_pull".into(), p.like_pull),
        ("taste_weight".into(), p.taste_weight),
        ("retention_weight".into(), p.retention_weight),
    ]
}

/// How a number is shown, and how its change is shown.
#[derive(Clone, Copy)]
enum Fmt {
    /// 0..1 shown as a percentage.
    Pct,
    /// Shown with this many decimals.
    Dec(usize),
}

fn cell(v: f64, prev: Option<f64>, f: Fmt) -> String {
    let (shown, delta) = match f {
        Fmt::Pct => (
            format!("{:.1}%", v * 100.0),
            prev.map(|p| (v - p) * 100.0)
                .map(|d| (d, format!("{d:+.1}"), 0.05)),
        ),
        Fmt::Dec(n) => (
            format!("{v:.n$}"),
            prev.map(|p| v - p)
                .map(|d| (d, format!("{d:+.n$}"), 0.5 * 10f64.powi(-(n as i32)))),
        ),
    };
    match delta {
        Some((d, s, eps)) if d.abs() >= eps => format!("{shown} ({s})"),
        _ => shown,
    }
}

fn table(header: &[&str], rows: &[Vec<String>]) -> String {
    let mut widths: Vec<usize> = header.iter().map(|h| h.chars().count()).collect();
    for r in rows {
        for (w, c) in widths.iter_mut().zip(r) {
            *w = (*w).max(c.chars().count());
        }
    }
    let line = |cells: Vec<&str>| {
        let mut s = String::new();
        for (i, (c, w)) in cells.iter().zip(&widths).enumerate() {
            let pad = w - c.chars().count();
            if i == 0 {
                s.push_str(c);
                s.push_str(&" ".repeat(pad));
            } else {
                s.push_str("  ");
                s.push_str(&" ".repeat(pad));
                s.push_str(c);
            }
        }
        s.trim_end().to_owned() + "\n"
    };
    let mut out = line(header.to_vec());
    for r in rows {
        out.push_str(&line(r.iter().map(String::as_str).collect()));
    }
    out
}

fn counts_line(c: &Counts) -> String {
    format!(
        "{} sessions, {} plays · skipped {:.0}% (in the first 30 s: {:.0}%) · liked {:.0}% · disliked {} · further {} · back {}",
        c.sessions,
        c.plays,
        c.skip_rate() * 100.0,
        c.early_skip_rate() * 100.0,
        c.like_rate() * 100.0,
        c.dislikes,
        c.further,
        c.back
    )
}

/// Renders `r` as text; numbers that changed since `prev` show the change in brackets.
pub fn render(r: &Report, prev: Option<&Report>) -> String {
    let mut out = String::new();
    let params: Vec<String> = r
        .params
        .iter()
        .map(|(k, v)| {
            let changed = prev
                .and_then(|p| p.params.iter().find(|(pk, _)| pk == k))
                .is_some_and(|(_, pv)| (pv - v).abs() > 1e-6);
            format!("{k} {v}{}", if changed { " (changed)" } else { "" })
        })
        .collect();
    out.push_str(&format!(
        "Library: {} tracks · sound space: {} · steering: {}\n",
        r.library_tracks,
        if r.clap_share > 0.0 {
            format!("{:.0}% CLAP", r.clap_share * 100.0)
        } else {
            "DSP only".to_owned()
        },
        params.join(", ")
    ));

    match &r.replay {
        Some(rep) => {
            let pr = prev.and_then(|p| p.replay.as_ref());
            out.push_str(&format!(
                "\nNext-track prediction on {}'s history\n  last {} days held out: {} plays in {} sessions, {} predictions; the models know the {} listens before that\n\n",
                r.user.as_deref().unwrap_or("the user"),
                rep.holdout_days,
                rep.test_plays,
                rep.test_sessions,
                rep.steps,
                rep.train_listens
            ));
            let rows: Vec<Vec<String>> = rep
                .rows
                .iter()
                .map(|row| {
                    let p = pr.and_then(|p| p.rows.iter().find(|x| x.name == row.name));
                    vec![
                        row.name.clone(),
                        cell(row.hit10, p.map(|p| p.hit10), Fmt::Pct),
                        cell(row.hit50, p.map(|p| p.hit50), Fmt::Pct),
                        cell(row.mrr, p.map(|p| p.mrr), Fmt::Dec(3)),
                    ]
                })
                .collect();
            out.push_str(&table(&["model", "hit@10", "hit@50", "MRR"], &rows));
            if !rep.skip_auc.is_empty() {
                let auc: Vec<String> = rep
                    .skip_auc
                    .iter()
                    .map(|(n, a)| {
                        let p = pr
                            .and_then(|p| p.skip_auc.iter().find(|x| &x.0 == n))
                            .map(|x| x.1);
                        format!("{n} {}", cell(*a, p, Fmt::Dec(2)))
                    })
                    .collect();                out.push_str(&format!(
                    "\nTelling listens from skips (AUC, 0.5 = chance): {}\n",
                    auc.join(" · ")
                ));
            }
            if let Some(m) = &rep.skip_model {
                let mut w: Vec<(&str, f32)> = dhunn_profile::FEATURES
                    .iter()
                    .copied()
                    .zip(m.weights)
                    .collect();
                w.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
                let top: Vec<String> = w
                    .iter()
                    .take(3)
                    .map(|(f, x)| format!("{f} {x:+.2}"))
                    .collect();
                out.push_str(&format!(
                    "Skip model: trained on {} plays before the hold-out; strongest signals: {}\n",
                    m.samples,
                    top.join(", ")
                ));
            }
        }
        None => out.push_str(
            "\nNext-track prediction: skipped (no user with enough matched history; see `dhunn import` or `dhunn dev sample-history`)\n",
        ),
    }

    if let Some(l) = &r.listeners {
        let pl = prev.and_then(|p| p.listeners.as_ref());
        out.push_str(&format!(
            "\nSimulated listeners (taste defined in {} features): {} listeners × {} sessions × {} tracks, each starting from one of their favourites\n\n",
            l.judge.name().to_uppercase(), l.listeners, l.sessions, l.length
        ));
        let rows: Vec<Vec<String>> = l
            .rows
            .iter()
            .map(|row| {
                let p = pl.and_then(|p| p.rows.iter().find(|x| x.name == row.name));
                vec![
                    row.name.clone(),
                    cell(row.taste, p.map(|p| p.taste), Fmt::Dec(1)),
                    cell(row.skip, p.map(|p| p.skip), Fmt::Pct),
                    cell(row.like, p.map(|p| p.like), Fmt::Pct),
                    cell(row.smooth, p.map(|p| p.smooth), Fmt::Dec(2)),
                    cell(row.worst_hop, p.map(|p| p.worst_hop), Fmt::Dec(2)),
                    cell(row.travel, p.map(|p| p.travel), Fmt::Dec(2)),
                    cell(row.variety, p.map(|p| p.variety), Fmt::Pct),
                    cell(row.same_artist, p.map(|p| p.same_artist), Fmt::Pct),
                ]
            })
            .collect();
        out.push_str(&table(
            &[
                "picker",
                "taste",
                "skipped",
                "liked",
                "smooth",
                "worst hop",
                "travel",
                "variety",
                "same artist",
            ],
            &rows,
        ));
        out.push_str(
            "\n  taste = mean taste percentile of picks (50 = average track) · smooth / worst hop = sound\n  similarity between consecutive tracks · travel = 1 − similarity of the last pick to the start\n",
        );
        if !l.modes.is_empty() {
            out.push_str("\nModes (adventurousness 0.5, with the listener's profile)\n\n");
            let opt = |v: Option<f64>, p: Option<Option<f64>>, f: Fmt| {
                v.map_or("–".to_owned(), |v| cell(v, p.flatten(), f))
            };
            let rows: Vec<Vec<String>> = l
                .modes
                .iter()
                .map(|row| {
                    let p = pl.and_then(|p| p.modes.iter().find(|x| x.name == row.name));
                    vec![
                        row.name.clone(),
                        cell(row.taste, p.map(|p| p.taste), Fmt::Dec(1)),
                        cell(row.skip, p.map(|p| p.skip), Fmt::Pct),
                        cell(row.new_to_you, p.map(|p| p.new_to_you), Fmt::Pct),
                        cell(row.smooth, p.map(|p| p.smooth), Fmt::Dec(2)),
                        cell(row.worst_hop, p.map(|p| p.worst_hop), Fmt::Dec(2)),
                        cell(row.travel, p.map(|p| p.travel), Fmt::Dec(2)),
                        opt(row.reached, p.map(|p| p.reached), Fmt::Pct),
                        opt(row.picks_to_reach, p.map(|p| p.picks_to_reach), Fmt::Dec(1)),
                    ]
                })
                .collect();
            out.push_str(&table(
                &[
                    "mode",
                    "taste",
                    "skipped",
                    "new to you",
                    "smooth",
                    "worst hop",
                    "travel",
                    "reached",
                    "picks to get there",
                ],
                &rows,
            ));
        }
    }

    if let Some(f) = &r.friends {
        let pf = prev.and_then(|p| p.friends.as_ref());
        out.push_str(&format!(
            "\nFriend groups: {} groups of three × {} sessions × {} tracks — A shares a home track with B, not with C (measured taste overlap {:.0}% and {:.0}%)\n\n",
            f.groups,
            f.sessions,
            f.length,
            f.overlap_b * 100.0,
            f.overlap_c * 100.0
        ));
        let rows: Vec<Vec<String>> = f
            .rows
            .iter()
            .map(|row| {
                let p = pf.and_then(|p| p.rows.iter().find(|x| x.name == row.name));
                vec![
                    row.name.clone(),
                    cell(row.taste, p.map(|p| p.taste), Fmt::Dec(1)),
                    cell(row.skip, p.map(|p| p.skip), Fmt::Pct),
                    cell(row.new_to_you, p.map(|p| p.new_to_you), Fmt::Pct),
                    cell(
                        row.friends_favourites,
                        p.map(|p| p.friends_favourites),
                        Fmt::Pct,
                    ),
                    cell(row.taste_b, p.map(|p| p.taste_b), Fmt::Dec(1)),
                    cell(row.both, p.map(|p| p.both), Fmt::Dec(1)),
                    cell(row.b_at_end, p.map(|p| p.b_at_end), Fmt::Dec(1)),
                ]
            })
            .collect();
        out.push_str(&table(
            &[
                "session",
                "A's taste",
                "skipped",
                "new to A",
                "friends' favourites",
                "B's taste",
                "suits both",
                "B's taste at end",
            ],
            &rows,
        ));
        out.push_str("\n  suits both = mean of min(A's, B's taste percentile) per pick\n");
    }

    match &r.logged {
        Some(g) => {
            out.push_str("\nYour dhunn sessions\n");
            out.push_str(&format!("  all time:    {}\n", counts_line(&g.all)));
            out.push_str(&format!("  last 7 days: {}\n", counts_line(&g.last_7_days)));
        }
        None => out.push_str("\nYour dhunn sessions: none yet (`dhunn play`)\n"),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::RankRow;

    fn report(hit10: f64) -> Report {
        Report {
            created_at: 0,
            label: None,
            library_tracks: 10,
            clap_share: 0.0,
            user: Some("krish".into()),
            params: params_list(&SteerParams::default()),
            replay: Some(ReplayReport {
                holdout_days: 14,
                train_listens: 100,
                test_plays: 20,
                test_sessions: 3,
                steps: 15,
                rows: vec![RankRow {
                    name: "steer".into(),
                    hit10,
                    hit50: 0.5,
                    mrr: 0.1,
                }],
                skip_auc: vec![("your taste".into(), 0.7)],
                skip_model: None,
            }),
            listeners: None,
            friends: None,
            logged: None,
        }
    }

    #[test]
    fn shows_changes_since_last_run() {
        let a = report(0.20);
        let b = report(0.25);
        let text = render(&b, Some(&a));
        assert!(text.contains("25.0% (+5.0)"), "{text}");
        // Unchanged values show no bracket.
        assert!(text.contains("50.0%  "), "{text}");
        assert!(!render(&b, None).contains("(+"));
        // Round-trips through JSON (saved runs are compared later).
        let json = serde_json::to_string(&b).unwrap();
        assert_eq!(serde_json::from_str::<Report>(&json).unwrap(), b);
    }
}
