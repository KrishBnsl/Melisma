//! Steering modes that go somewhere: where to go, and the route of places to get there.

use std::collections::HashSet;

use dhunn_core::TrackId;
use dhunn_index::dot;
use dhunn_index::places::Places;

/// Journeys are paced to arrive in about this many picks.
const ARRIVE_WITHIN: usize = 12;

/// What a session is doing.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Mode {
    /// Drift gradually from the seeds (the default).
    #[default]
    Drift,
    /// Stay close to the sound you play most.
    Usual,
    /// One big hop to a coherent part of the library you haven't explored, then settle in.
    Surprise,
    /// Travel, in small steps, to the nearest part of the library you've never played.
    New,
    /// Travel, in small steps, to a destination (a track or an artist's sound).
    Toward {
        /// Unit vector in the sound space.
        target: Vec<f32>,
        label: String,
    },
}

impl Mode {
    pub fn name(&self) -> &'static str {
        match self {
            Mode::Drift => "drift",
            Mode::Usual => "usual",
            Mode::Surprise => "surprise",
            Mode::New => "new",
            Mode::Toward { .. } => "toward",
        }
    }

    /// Whether the mode travels to a place chosen at the start.
    pub fn travels(&self) -> bool {
        matches!(self, Mode::Surprise | Mode::New | Mode::Toward { .. })
    }
}

/// Where a journey is going, for explanations.
#[derive(Debug, Clone, PartialEq)]
pub enum Goal {
    /// A part of the library the listener hasn't explored.
    Surprise,
    /// Tracks the listener has never played.
    New,
    Destination(String),
}

/// A pick's place on the journey: stop `step` of `of` (stop 0 = where the session started;
/// `step == of` = arrived).
#[derive(Debug, Clone, PartialEq)]
pub struct Leg {
    pub goal: Goal,
    pub step: usize,
    pub of: usize,
    /// This pick is the big hop of a surprise.
    pub hop: bool,
    /// The first pick at this stop (with `step == of`: the pick that arrived).
    pub first_here: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct Journey {
    pub goal: Goal,
    /// Place ids, starting place first.
    pub stops: Vec<usize>,
    pub at: usize,
    /// Picks made at the current stop.
    pub picks_here: usize,
    /// Picks per stop: paced so the journey arrives in about `ARRIVE_WITHIN` picks.
    pub per_stop: usize,
    /// The exact target (a destination track/artist), used once arrived.
    pub target: Option<Vec<f32>>,
    pub hop_pending: bool,
}

impl Journey {
    pub fn arrived(&self) -> bool {
        self.at + 1 >= self.stops.len()
    }

    pub fn leg(&self, hop: bool) -> Leg {
        Leg {
            goal: self.goal.clone(),
            step: self.at,
            of: self.stops.len() - 1,
            hop,
            first_here: self.picks_here == 0,
        }
    }
}

fn unknown_share(places: &Places, p: usize, known: &HashSet<TrackId>) -> f32 {
    let m = &places.places[p].members;
    m.iter().filter(|t| !known.contains(t)).count() as f32 / m.len().max(1) as f32
}

/// Chooses a journey for `mode` from where the session is (`anchor`). `taste` are the
/// listener's taste centroids, `known` the tracks they have ever played, `pick` a random
/// number in 0..1. `None` if the mode doesn't travel or nowhere fits.
pub(crate) fn plan(
    mode: &Mode,
    places: &Places,
    anchor: &[f32],
    taste: &[Vec<f32>],
    known: &HashSet<TrackId>,
    route: bool,
    pick: f32,
) -> Option<Journey> {
    let here = places.nearest(anchor)?;
    let taste_of = |v: &[f32]| taste.iter().map(|c| dot(c, v)).fold(0.0f32, f32::max);
    let (goal, to, target) = match mode {
        Mode::Surprise => {
            // Coherent, unexplored and far from where the session is — but somewhere the
            // listener's taste reaches, so the surprise is a pleasant one.
            let mut cands: Vec<(usize, f32)> = (0..places.len())
                .filter(|&p| p != here && places.places[p].members.len() >= 3)
                .map(|p| {
                    let c = &places.places[p].centroid;
                    let score = places.places[p].coherence + unknown_share(places, p, known)
                        - dot(c, anchor).max(0.0)
                        + 0.5 * taste_of(c);
                    (p, score)
                })
                .collect();
            cands.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let top = cands.len().min(3);
            if top == 0 {
                return None;
            }
            let p = cands[((pick * top as f32) as usize).min(top - 1)].0;
            // One big hop: no route.
            return Some(Journey {
                goal: Goal::Surprise,
                stops: vec![here, p],
                at: 0,
                picks_here: 0,
                per_stop: 1,
                target: None,
                hop_pending: true,
            });
        }
        Mode::New => {
            // The nearest place where most tracks are new to the listener.
            let p = (0..places.len())
                .filter(|&p| p != here && unknown_share(places, p, known) >= 0.5)
                .max_by(|&a, &b| {
                    let s = |p: usize| {
                        unknown_share(places, p, known)
                            * (1.0 + dot(&places.places[p].centroid, anchor))
                    };
                    s(a).total_cmp(&s(b)).then(b.cmp(&a))
                })?;
            (Goal::New, p, None)
        }
        Mode::Toward { target, label } => {
            let p = places.nearest(target)?;
            (Goal::Destination(label.clone()), p, Some(target.clone()))
        }
        Mode::Drift | Mode::Usual => return None,
    };
    let stops = if route {
        places.route(here, to).unwrap_or(vec![here, to])
    } else {
        vec![here, to]
    };
    // Pace long routes to arrive in time; short ones still move on after two picks.
    let per_stop = (ARRIVE_WITHIN / stops.len().saturating_sub(1).max(1)).clamp(1, 2);
    Some(Journey {
        goal,
        // Head out on the first pick: the starting place is where the session already is.
        at: usize::from(stops.len() > 1),
        stops,
        picks_here: 0,
        per_stop,
        target,
        hop_pending: false,
    })
}
