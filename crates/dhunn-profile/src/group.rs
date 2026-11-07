//! Friends: how much two people's tastes overlap, what friends love, where a friend's taste
//! goes beyond yours, and where a group's tastes meet.
//!
//! Small-group collaborative filtering: with ≤ 20 people there is no need for matrix
//! factorization — a friend's signal for a track is simply their (time-decayed) affinity for
//! it, weighted by trust: 0.5 for any friend (these are people you chose) plus up to 0.5 more
//! the more their taste overlaps yours.

use std::collections::HashMap;

use dhunn_core::TrackId;
use dhunn_index::{dot, normalize};

use crate::Profile;

/// A friend who shares their profile.
pub struct Friend {
    pub name: String,
    pub profile: Profile,
    /// Overlap of their taste with the listener's (0..1).
    pub overlap: f32,
}

/// How much of `a`'s taste lies close to some part of `b`'s: Σ over `a`'s regions of
/// share × closeness to `b`'s nearest region (similarity, floored at 0).
fn coverage(a: &Profile, b: &Profile) -> f32 {
    a.regions
        .iter()
        .map(|ra| {
            ra.share
                * b.regions
                    .iter()
                    .map(|rb| dot(&ra.centroid, &rb.centroid))
                    .fold(0.0f32, f32::max)
        })
        .sum()
}

/// How much two tastes overlap (0..1): the coverage of each by the other, averaged. 1 = the
/// same regions, 0 = nothing in common.
pub fn overlap(a: &Profile, b: &Profile) -> f32 {
    (coverage(a, b) + coverage(b, a)) / 2.0
}

/// How much a friend's love for a track counts: 0.5 for any friend, up to 1 for one whose
/// taste is the same as yours.
pub fn trust(overlap: f32) -> f32 {
    0.5 + 0.5 * overlap.clamp(0.0, 1.0)
}

/// A friend's affinity for each track they've played, scaled so their heavy rotation is ≈ 1
/// and tracks they skip or dislike are negative (down to −1).
fn affinity(p: &Profile) -> HashMap<TrackId, f32> {
    let mut pos: Vec<f32> = p.weight.values().copied().filter(|w| *w > 0.0).collect();
    if pos.is_empty() {
        return HashMap::new();
    }
    pos.sort_by(f32::total_cmp);
    // The 90th percentile of positive weights counts as "on repeat".
    let scale = pos[(pos.len() * 9 / 10).min(pos.len() - 1)].max(1e-6);
    p.weight
        .iter()
        .map(|(&t, &w)| (t, (w / scale).clamp(-1.0, 1.0)))
        .collect()
}

/// Per track: the friends' signal (Σ trust · affinity, clamped to −1..1) and the index of
/// the friend contributing most.
pub fn friend_signal(friends: &[Friend]) -> HashMap<TrackId, (f32, u32)> {
    let mut out: HashMap<TrackId, (f32, u32, f32)> = HashMap::new();
    for (i, f) in friends.iter().enumerate() {
        for (t, a) in affinity(&f.profile) {
            let c = trust(f.overlap) * a;
            let e = out.entry(t).or_insert((0.0, i as u32, f32::MIN));
            e.0 += c;
            if c > e.2 {
                e.1 = i as u32;
                e.2 = c;
            }
        }
    }
    out.into_iter()
        .map(|(t, (s, i, _))| (t, (s.clamp(-1.0, 1.0), i)))
        .collect()
}

/// The part of a friend's taste the listener has least of: the friend's region that scores
/// highest on share × (1 − closeness to the listener's nearest region).
pub fn beyond_me(me: &Profile, friend: &Profile) -> Option<Vec<f32>> {
    friend
        .regions
        .iter()
        .map(|r| {
            let near = me
                .regions
                .iter()
                .map(|m| dot(&m.centroid, &r.centroid))
                .fold(0.0f32, f32::max);
            (r, r.share * (1.0 - near.max(0.0)))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(r, _)| r.centroid.clone())
}

/// Where everyone's tastes meet: among all the group's region centres (and midpoints of
/// pairs), the point whose *weakest* listener likes it most. `None` if someone has no
/// profile yet.
pub fn meeting_point(group: &[&Profile]) -> Option<Vec<f32>> {
    if group.is_empty() || group.iter().any(|p| p.regions.is_empty()) {
        return None;
    }
    let mut cands: Vec<Vec<f32>> = Vec::new();
    let all: Vec<&Vec<f32>> = group
        .iter()
        .flat_map(|p| p.regions.iter().map(|r| &r.centroid))
        .collect();
    for (i, a) in all.iter().enumerate() {
        cands.push((*a).clone());
        for b in &all[i + 1..] {
            let mut m: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
            normalize(&mut m);
            cands.push(m);
        }
    }
    cands.into_iter().max_by(|a, b| {
        let worst = |v: &[f32]| {
            group
                .iter()
                .map(|p| p.taste_of(v).unwrap_or(-1.0))
                .fold(f32::MAX, f32::min)
        };
        worst(a).total_cmp(&worst(b))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{obs, ring};
    use crate::{Obs, ProfileConfig};

    fn listener(tracks: &[u64]) -> Profile {
        let s = ring();
        let sess: Vec<Obs> = (0..3)
            .flat_map(|_| tracks.iter().map(|&t| obs(t, true)))
            .collect();
        Profile::build(&s, &[sess], &HashMap::new(), 0, &ProfileConfig::default())
    }

    #[test]
    fn overlap_signal_and_destinations() {
        let s = ring();
        // Me: around 5 and 35. Aarav: around 35 and 50. Bea: around 20.
        let me = listener(&[3, 4, 5, 6, 7, 33, 34, 35, 36, 37]);
        let aarav = listener(&[33, 34, 35, 36, 37, 48, 49, 50, 51, 52]);
        let bea = listener(&[18, 19, 20, 21, 22]);
        let oa = overlap(&me, &aarav);
        let ob = overlap(&me, &bea);
        assert!(oa > 0.2 && oa > 2.0 * ob, "aarav {oa}, bea {ob}");
        assert!((overlap(&me, &me) - 1.0).abs() < 1e-3);
        assert!((overlap(&me, &aarav) - overlap(&aarav, &me)).abs() < 1e-6);

        let friends = vec![
            Friend {
                name: "aarav".into(),
                profile: aarav,
                overlap: oa,
            },
            Friend {
                name: "bea".into(),
                profile: bea,
                overlap: ob,
            },
        ];
        let sig = friend_signal(&friends);
        let (s50, who) = sig[&TrackId(50)];
        assert!(s50 > 0.0 && who == 0);
        assert!(
            sig[&TrackId(50)].0 > sig[&TrackId(20)].0,
            "a closer friend counts more"
        );
        assert!(!sig.contains_key(&TrackId(10)));

        // What Aarav has that I don't: the region around 50, not the shared one at 35.
        let beyond = beyond_me(&me, &friends[0].profile).unwrap();
        assert!(dot(&beyond, s.vector(TrackId(50)).unwrap()) > 0.95);

        // Me and Aarav meet around 35.
        let meet = meeting_point(&[&me, &friends[0].profile]).unwrap();
        assert!(dot(&meet, s.vector(TrackId(35)).unwrap()) > 0.95);
        assert!(meeting_point(&[&me, &Profile::default()]).is_none());
    }
}
