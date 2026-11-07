//! Learned embeddings per track (e.g. CLAP), keyed by model name and version.

use std::collections::{HashMap, HashSet};

use dhunn_core::TrackId;
use rusqlite::params;

use crate::{Result, Store, blob_to_vec, vec_to_blob};

impl Store {
    /// Stores (or replaces) a track's embedding for `model`.
    pub fn save_embedding(
        &mut self,
        track: TrackId,
        model: &str,
        version: u32,
        vec: &[f32],
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings (track_id, model, version, vec) VALUES (?1, ?2, ?3, ?4)",
            params![track.0 as i64, model, version, vec_to_blob(vec)],
        )?;
        Ok(())
    }

    /// Tracks that already have a current embedding for `model`.
    pub fn tracks_with_embedding(&self, model: &str, version: u32) -> Result<HashSet<TrackId>> {
        let mut stmt = self
            .conn
            .prepare("SELECT track_id FROM embeddings WHERE model = ?1 AND version = ?2")?;
        let rows = stmt.query_map(params![model, version], |r| {
            Ok(TrackId(r.get::<_, i64>(0)? as u64))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// All current embeddings for `model`.
    pub fn embeddings(&self, model: &str, version: u32) -> Result<HashMap<TrackId, Vec<f32>>> {
        let mut stmt = self
            .conn
            .prepare("SELECT track_id, vec FROM embeddings WHERE model = ?1 AND version = ?2")?;
        let rows = stmt.query_map(params![model, version], |r| {
            Ok((
                TrackId(r.get::<_, i64>(0)? as u64),
                blob_to_vec(&r.get::<_, Vec<u8>>(1)?),
            ))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Each track's embedding from the newest version stored for `model`.
    pub fn latest_embeddings(&self, model: &str) -> Result<HashMap<TrackId, Vec<f32>>> {
        let mut stmt = self.conn.prepare(
            "SELECT track_id, vec FROM embeddings WHERE model = ?1
             AND version = (SELECT MAX(version) FROM embeddings WHERE model = ?1)",
        )?;
        let rows = stmt.query_map(params![model], |r| {
            Ok((
                TrackId(r.get::<_, i64>(0)? as u64),
                blob_to_vec(&r.get::<_, Vec<u8>>(1)?),
            ))
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// All track ids, for tools that walk the whole library.
    pub fn track_ids(&self) -> Result<Vec<TrackId>> {
        let mut stmt = self.conn.prepare("SELECT id FROM tracks ORDER BY id")?;
        let rows = stmt.query_map([], |r| Ok(TrackId(r.get::<_, i64>(0)? as u64)))?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FileStamp, NewAnalysis};
    use dhunn_core::{FeatureVec, TrackMeta};

    #[test]
    fn embeddings_round_trip_by_model_and_version() {
        let mut s = Store::open_in_memory().unwrap();
        let a = NewAnalysis {
            content_hash: "h".into(),
            meta: TrackMeta {
                title: "t".into(),
                album: None,
                year: None,
                duration_s: 1.0,
                mb_recording_id: None,
                artists: vec![],
            },
            features: FeatureVec {
                kind: "dsp".into(),
                version: 1,
                names: vec!["a".into()],
                values: vec![0.0],
            },
            summary_json: None,
        };
        let t = s
            .save_analysis(
                "/x",
                FileStamp {
                    size: 1,
                    mtime_ns: 1,
                },
                &a,
            )
            .unwrap();
        s.save_embedding(t, "clap", 1, &[0.5, -0.5]).unwrap();
        assert_eq!(s.embeddings("clap", 1).unwrap()[&t], vec![0.5, -0.5]);
        assert!(s.embeddings("clap", 2).unwrap().is_empty());
        s.save_embedding(t, "clap", 2, &[1.0]).unwrap();
        assert!(s.tracks_with_embedding("clap", 1).unwrap().is_empty());
        assert!(s.tracks_with_embedding("clap", 2).unwrap().contains(&t));
        assert_eq!(s.track_ids().unwrap(), vec![t]);
        assert_eq!(s.latest_embeddings("clap").unwrap()[&t], vec![1.0]);
    }
}
