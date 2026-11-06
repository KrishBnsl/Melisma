//! Evaluation tooling: simulated listeners now; offline metrics in step 5.

pub mod rng;
pub mod simulate;

pub use simulate::{GroundTruth, SimConfig, SimError, SimStats, simulate_spotify_history};
