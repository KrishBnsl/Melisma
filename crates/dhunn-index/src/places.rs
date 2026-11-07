//! A map of the library: tracks grouped into **places** (clusters in the sound space), each
//! linked to its nearest places. Steering uses it to pick somewhere to go (surprise me,
//! something new) and to plan a route there in small steps (destinations).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use dhunn_core::TrackId;
use rayon::prelude::*;

use crate::{SoundSpace, dot, normalize};

/// Neighbours per place in the route graph.
const LINKS: usize = 6;
/// At most this many tracks are used to fit the places (all are then assigned).
const FIT_SAMPLE: usize = 20_000;

#[derive(Debug, Clone)]
pub struct Place {
    /// Unit vector.
    pub centroid: Vec<f32>,
    pub members: Vec<TrackId>,
    /// Mean similarity of members to the centroid: how coherent the place sounds.
    pub coherence: f32,
}

#[derive(Debug, Clone)]
pub struct Places {
    pub places: Vec<Place>,
    of_track: HashMap<TrackId, usize>,
    /// Per place: `(neighbour, similarity)`, symmetric.
    links: Vec<Vec<(usize, f32)>>,
}

#[derive(PartialEq)]
struct Node(f32, usize);
impl Eq for Node {}
impl PartialOrd for Node {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Node {
    // Min-heap on cost.
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0).then(o.1.cmp(&self.1))
    }
}

fn nearest_centre(centres: &[Vec<f32>], v: &[f32]) -> usize {
    (0..centres.len())
        .max_by(|&a, &b| {
            dot(&centres[a], v)
                .total_cmp(&dot(&centres[b], v))
                .then(b.cmp(&a))
        })
        .expect("at least one centre")
}

impl Places {
    /// About one place per 8 tracks (4–300 places), fitted by spherical k-means with a
    /// deterministic farthest-point start.
    pub fn build(space: &SoundSpace) -> Self {
        let ids = space.ids();
        let n = ids.len();
        if n == 0 {
            return Self {
                places: Vec::new(),
                of_track: HashMap::new(),
                links: Vec::new(),
            };
        }
        let k = ((n as f64 / 8.0).round() as usize).clamp(4, 300).min(n);
        let step = n.div_ceil(FIT_SAMPLE).max(1);
        let sample: Vec<&[f32]> = ids
            .iter()
            .step_by(step)
            .filter_map(|&t| space.vector(t))
            .collect();

        let mut centres: Vec<Vec<f32>> = vec![sample[0].to_vec()];
        let mut best: Vec<f32> = sample.iter().map(|v| dot(&centres[0], v)).collect();
        while centres.len() < k {
            let far = best
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .expect("sample not empty");
            let c = sample[far].to_vec();
            best.iter_mut()
                .zip(&sample)
                .for_each(|(b, v)| *b = b.max(dot(&c, v)));
            centres.push(c);
        }
        let dims = space.dims();
        for _ in 0..15 {
            let assign: Vec<usize> = sample
                .par_iter()
                .map(|v| nearest_centre(&centres, v))
                .collect();
            let mut sums = vec![vec![0f32; dims]; k];
            for (v, &a) in sample.iter().zip(&assign) {
                sums[a].iter_mut().zip(*v).for_each(|(s, x)| *s += x);
            }
            for (c, mut s) in centres.iter_mut().zip(sums) {
                if s.iter().any(|x| *x != 0.0) {
                    normalize(&mut s);
                    *c = s;
                }
            }
        }

        // Assign every track; drop places nobody landed in.
        let assign: Vec<usize> = ids
            .par_iter()
            .map(|&t| nearest_centre(&centres, space.vector(t).expect("id from space")))
            .collect();
        let mut members: Vec<Vec<TrackId>> = vec![Vec::new(); k];
        for (&t, &a) in ids.iter().zip(&assign) {
            members[a].push(t);
        }
        let mut places: Vec<Place> = Vec::new();
        for (c, m) in centres.into_iter().zip(members) {
            if m.is_empty() {
                continue;
            }
            let coherence = m
                .iter()
                .map(|&t| dot(&c, space.vector(t).expect("member")))
                .sum::<f32>()
                / m.len() as f32;
            places.push(Place {
                centroid: c,
                members: m,
                coherence,
            });
        }
        let of_track = places
            .iter()
            .enumerate()
            .flat_map(|(i, p)| p.members.iter().map(move |&t| (t, i)))
            .collect();

        let mut links: Vec<Vec<(usize, f32)>> = vec![Vec::new(); places.len()];
        for i in 0..places.len() {
            let mut near: Vec<(usize, f32)> = (0..places.len())
                .filter(|&j| j != i)
                .map(|j| (j, dot(&places[i].centroid, &places[j].centroid)))
                .collect();
            near.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            for &(j, s) in near.iter().take(LINKS) {
                if !links[i].iter().any(|l| l.0 == j) {
                    links[i].push((j, s));
                }
                if !links[j].iter().any(|l| l.0 == i) {
                    links[j].push((i, s));
                }
            }
        }
        Self {
            places,
            of_track,
            links,
        }
    }

    pub fn len(&self) -> usize {
        self.places.len()
    }

    pub fn is_empty(&self) -> bool {
        self.places.is_empty()
    }

    pub fn place_of(&self, t: TrackId) -> Option<usize> {
        self.of_track.get(&t).copied()
    }

    /// The place whose centre is closest to a unit vector.
    pub fn nearest(&self, v: &[f32]) -> Option<usize> {
        (!self.places.is_empty()).then(|| {
            let c: Vec<Vec<f32>> = self.places.iter().map(|p| p.centroid.clone()).collect();
            nearest_centre(&c, v)
        })
    }

    /// The cheapest route of places from `from` to `to` (both included), where a hop
    /// between places costs (1 − similarity)²: many small steps beat one big jump.
    /// `None` if `to` can't be reached.
    pub fn route(&self, from: usize, to: usize) -> Option<Vec<usize>> {
        let n = self.places.len();
        if from >= n || to >= n {
            return None;
        }
        let mut cost = vec![f32::INFINITY; n];
        let mut prev = vec![usize::MAX; n];
        let mut heap = BinaryHeap::new();
        cost[from] = 0.0;
        heap.push(Node(0.0, from));
        while let Some(Node(c, i)) = heap.pop() {
            if i == to {
                break;
            }
            if c > cost[i] {
                continue;
            }
            for &(j, s) in &self.links[i] {
                let nc = c + (1.0 - s).powi(2);
                if nc < cost[j] {
                    cost[j] = nc;
                    prev[j] = i;
                    heap.push(Node(nc, j));
                }
            }
        }
        if !cost[to].is_finite() {
            return None;
        }
        let mut path = vec![to];
        while *path.last().expect("not empty") != from {
            path.push(prev[*path.last().expect("not empty")]);
        }
        path.reverse();
        Some(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AspectWeights;

    /// 120 tracks on a circle: a route between opposite sides must go round.
    fn ring() -> SoundSpace {
        let names: Vec<String> = ["mfcc00_mean", "log2_tempo"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let rows = (0..120)
            .map(|i| {
                let a = i as f32 / 120.0 * std::f32::consts::TAU;
                (TrackId(i + 1), vec![a.cos(), a.sin()])
            })
            .collect();
        SoundSpace::build(&names, rows, &AspectWeights::default())
    }

    #[test]
    fn places_cover_the_library_and_routes_take_small_steps() {
        let s = ring();
        let p = Places::build(&s);
        assert_eq!(p.len(), 15);
        let total: usize = p.places.iter().map(|x| x.members.len()).sum();
        assert_eq!(total, 120);
        assert!(p.places.iter().all(|x| x.coherence > 0.9));
        let from = p.place_of(TrackId(1)).unwrap();
        let to = p.place_of(TrackId(61)).unwrap(); // opposite side
        let route = p.route(from, to).unwrap();
        assert!(route.len() >= 5, "{route:?}");
        for w in route.windows(2) {
            let sim = dot(&p.places[w[0]].centroid, &p.places[w[1]].centroid);
            assert!(sim > 0.8, "hop {w:?} too big: {sim}");
        }
        assert_eq!(p.route(from, from).unwrap(), vec![from]);
        assert_eq!(p.nearest(s.vector(TrackId(61)).unwrap()), Some(to));
    }
}
