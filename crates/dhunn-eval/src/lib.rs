//! Evaluation: simulated listeners, next-track prediction on listening history, simulated
//! sessions with feedback, and a summary of real `dhunn play` sessions.

pub mod listeners;
pub mod logged;
pub mod replay;
pub mod report;
pub mod rng;
pub mod simulate;
pub mod taste;

pub use listeners::{ListenerConfig, ListenerReport, SessionRow, simulated_listeners};
pub use logged::{LoggedReport, summarize};
pub use replay::{RankRow, ReplayConfig, ReplayReport, replay};
pub use report::{Report, params_list, render};
pub use simulate::{GroundTruth, SimConfig, SimError, SimStats, simulate_spotify_history};
pub use taste::TasteSpace;
