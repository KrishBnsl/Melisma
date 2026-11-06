//! Scans a music folder and brings the store up to date.
//!
//! For each audio file: skip it if its size/mtime are unchanged and its track already has
//! current features; otherwise decode it and hash the decoded audio. Known audio (a renamed,
//! moved, re-tagged or duplicate file) is linked to its existing track without re-analysis;
//! new audio is analyzed (DSP features + tags) and saved. Files that fail are recorded and
//! not retried until they change. Files that disappeared from disk are forgotten.
//!
//! Analysis runs on a bounded worker pool; all database writes happen on the calling thread.

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant, UNIX_EPOCH},
};

use dhunn_core::{ANALYSIS_SAMPLE_RATE, TrackId, TrackMeta};
use dhunn_dsp::{AnalyzeOptions, DSP_VERSION};
use dhunn_store::{FileStamp, NewAnalysis, Store, StoreError};
use rayon::prelude::*;
use xxhash_rust::xxh3::Xxh3;

/// File extensions treated as audio (everything symphonia can decode).
pub const AUDIO_EXTENSIONS: &[&str] = &[
    "mp3", "flac", "wav", "wave", "m4a", "mp4", "aac", "ogg", "oga", "aif", "aiff", "caf",
];

const FEATURE_KIND: &str = "dsp";

#[derive(Debug, thiserror::Error)]
pub enum AnalysisError {
    #[error("cannot open music folder {path}: {source}")]
    Root {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("could not start worker threads: {0}")]
    Threads(String),
}

#[derive(Debug, Clone)]
pub struct ScanOptions {
    /// Re-analyze every file even if it looks unchanged.
    pub force: bool,
    /// Tracks analyzed in parallel (each needs a few hundred MB for a long track).
    pub jobs: usize,
    pub dsp: AnalyzeOptions,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            force: false,
            jobs: default_jobs(),
            dsp: AnalyzeOptions::default(),
        }
    }
}

/// Half the cores, between 1 and 6: analysis is also parallel inside each track.
pub fn default_jobs() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .div_ceil(2)
        .clamp(1, 6)
}

/// Progress notifications, delivered on the calling thread.
#[derive(Debug)]
pub enum ScanEvent<'a> {
    /// Discovery finished: `to_process` of `found` files need decoding.
    Planned { found: usize, to_process: usize },
    File {
        done: usize,
        total: usize,
        path: &'a str,
        outcome: FileOutcome<'a>,
    },
}

#[derive(Debug)]
pub enum FileOutcome<'a> {
    Analyzed {
        meta: &'a TrackMeta,
        seconds: f64,
    },
    /// Audio already known (moved, re-tagged or duplicate file): linked, not re-analyzed.
    Reused,
    Failed {
        error: &'a str,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScanReport {
    pub found: usize,
    pub unchanged: usize,
    /// Previously failed and unchanged since, so not retried.
    pub skipped_failed: usize,
    pub analyzed: usize,
    pub reused: usize,
    pub failed: usize,
    /// Files that disappeared from disk since the last scan.
    pub removed: usize,
    /// Directory entries that could not be read (permissions, broken links, non-UTF-8 names).
    pub unreadable: usize,
    pub elapsed: Duration,
}

struct Job {
    path: String,
    stamp: FileStamp,
}

enum Outcome {
    Analyzed {
        analysis: Box<NewAnalysis>,
        seconds: f64,
    },
    Known {
        hash: String,
        meta: TrackMeta,
    },
    Failed(String),
}

fn is_audio(path: &Path) -> bool {
    let hidden = path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with("._"));
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    !hidden && ext.is_some_and(|e| AUDIO_EXTENSIONS.contains(&e.as_str()))
}

fn stamp_of(meta: &std::fs::Metadata) -> FileStamp {
    let mtime_ns = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as i64)
        .unwrap_or(0);
    FileStamp {
        size: meta.len(),
        mtime_ns,
    }
}

/// Hash of the decoded audio: identifies the recording regardless of file name or tags.
pub fn content_hash(samples: &[f32], sample_rate: u32) -> String {
    let mut h = Xxh3::new();
    h.update(&sample_rate.to_le_bytes());
    for s in samples {
        h.update(&s.to_le_bytes());
    }
    format!("{:016x}", h.digest())
}

fn track_meta(path: &Path, duration_s: f64) -> TrackMeta {
    let tags = dhunn_meta::read_tags(path);
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    TrackMeta {
        title: tags.title.unwrap_or(stem),
        album: tags.album,
        year: tags.year,
        duration_s,
        mb_recording_id: tags.mb_recording_id,
        artists: tags.artists,
    }
}

fn process(
    path: &str,
    known: &HashSet<String>,
    in_flight: &Mutex<HashSet<String>>,
    dsp: &AnalyzeOptions,
) -> Outcome {
    let started = Instant::now();
    let p = Path::new(path);
    let audio = match dhunn_audio::decode_mono(p) {
        Ok(a) => a,
        Err(e) => return Outcome::Failed(e.to_string()),
    };
    let hash = content_hash(&audio.samples, audio.sample_rate);
    let meta = track_meta(p, audio.duration_secs());
    let first_copy = in_flight
        .lock()
        .map(|mut s| s.insert(hash.clone()))
        .unwrap_or(true);
    if known.contains(&hash) || !first_copy {
        return Outcome::Known { hash, meta };
    }
    let samples =
        match dhunn_audio::resample(&audio.samples, audio.sample_rate, ANALYSIS_SAMPLE_RATE) {
            Ok(s) => s,
            Err(e) => return Outcome::Failed(e.to_string()),
        };
    drop(audio);
    let summary = match dhunn_dsp::analyze(&samples, ANALYSIS_SAMPLE_RATE, dsp) {
        Ok((s, _)) => s,
        Err(e) => return Outcome::Failed(e.to_string()),
    };
    let analysis = NewAnalysis {
        content_hash: hash,
        meta,
        features: summary.feature_vec(),
        summary_json: serde_json::to_string(&summary).ok(),
    };
    Outcome::Analyzed {
        analysis: Box::new(analysis),
        seconds: started.elapsed().as_secs_f64(),
    }
}

/// Applies worker results to the store on the scanning thread and reports progress.
struct Sink<'s, F> {
    store: &'s mut Store,
    on_event: F,
    report: ScanReport,
    done: usize,
    total: usize,
    /// Copies of audio whose first copy was still being analyzed when they finished.
    deferred: Vec<(usize, String)>,
}

impl<F: FnMut(&ScanEvent<'_>)> Sink<'_, F> {
    fn emit(&mut self, path: &str, outcome: FileOutcome<'_>) {
        self.done += 1;
        (self.on_event)(&ScanEvent::File {
            done: self.done,
            total: self.total,
            path,
            outcome,
        });
    }

    fn fail(&mut self, job: &Job, error: &str) -> Result<(), AnalysisError> {
        self.store.record_failure(&job.path, job.stamp, error)?;
        self.report.failed += 1;
        self.emit(&job.path, FileOutcome::Failed { error });
        Ok(())
    }

    fn handle(
        &mut self,
        i: usize,
        job: &Job,
        prev: Option<TrackId>,
        outcome: Outcome,
    ) -> Result<(), AnalysisError> {
        match outcome {
            Outcome::Analyzed { analysis, seconds } => {
                self.store.save_analysis(&job.path, job.stamp, &analysis)?;
                self.report.analyzed += 1;
                self.emit(
                    &job.path,
                    FileOutcome::Analyzed {
                        meta: &analysis.meta,
                        seconds,
                    },
                );
            }
            Outcome::Known { hash, meta } => {
                match self.store.link_file(&job.path, job.stamp, &hash)? {
                    Some(id) => {
                        if prev == Some(id) {
                            self.store.update_meta(id, &meta)?; // same file, new tags
                        }
                        self.report.reused += 1;
                        self.emit(&job.path, FileOutcome::Reused);
                    }
                    None => self.deferred.push((i, hash)),
                }
            }
            Outcome::Failed(error) => self.fail(job, &error)?,
        }
        Ok(())
    }

    fn finish_deferred(&mut self, jobs: &[Job]) -> Result<(), AnalysisError> {
        for (i, hash) in std::mem::take(&mut self.deferred) {
            let job = &jobs[i];
            if self.store.link_file(&job.path, job.stamp, &hash)?.is_some() {
                self.report.reused += 1;
                self.emit(&job.path, FileOutcome::Reused);
            } else {
                self.fail(job, "identical to another file that failed to analyze")?;
            }
        }
        Ok(())
    }
}

/// Scans `root` recursively and updates `store`. See the module docs for the rules.
pub fn scan(
    store: &mut Store,
    root: &Path,
    opts: &ScanOptions,
    on_event: impl FnMut(&ScanEvent<'_>),
) -> Result<ScanReport, AnalysisError> {
    let started = Instant::now();
    let root = root.canonicalize().map_err(|source| AnalysisError::Root {
        path: root.to_path_buf(),
        source,
    })?;
    let root_str = root.to_string_lossy().into_owned();
    let mut report = ScanReport::default();

    let known_files = store.files_under(&root_str)?;
    let failures = store.failures_under(&root_str)?;
    let current = store.track_ids_with_features(FEATURE_KIND, DSP_VERSION)?;
    let known_hashes = if opts.force {
        HashSet::new()
    } else {
        store.hashes_with_features(FEATURE_KIND, DSP_VERSION)?
    };

    // Discover.
    let mut seen = HashSet::new();
    let mut jobs = Vec::new();
    for entry in walkdir::WalkDir::new(&root).follow_links(true) {
        let Ok(entry) = entry else {
            report.unreadable += 1;
            continue;
        };
        if !entry.file_type().is_file() || !is_audio(entry.path()) {
            continue;
        }
        let (Some(path), Ok(meta)) = (entry.path().to_str(), entry.metadata()) else {
            report.unreadable += 1;
            continue;
        };
        report.found += 1;
        let stamp = stamp_of(&meta);
        seen.insert(path.to_owned());
        if !opts.force {
            if known_files
                .get(path)
                .is_some_and(|k| k.stamp == stamp && current.contains(&k.track_id))
            {
                report.unchanged += 1;
                continue;
            }
            if failures.get(path) == Some(&stamp) {
                report.skipped_failed += 1;
                continue;
            }
        }
        jobs.push(Job {
            path: path.to_owned(),
            stamp,
        });
    }
    let mut on_event = on_event;
    on_event(&ScanEvent::Planned {
        found: report.found,
        to_process: jobs.len(),
    });

    // Analyze in parallel; write results here as they arrive.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(opts.jobs.max(1))
        .thread_name(|i| format!("dhunn-analyze-{i}"))
        .build()
        .map_err(|e| AnalysisError::Threads(e.to_string()))?;
    let in_flight = Mutex::new(HashSet::new());
    let cancel = AtomicBool::new(false);
    let (tx, rx) = mpsc::channel::<(usize, Outcome)>();
    let mut sink = Sink {
        store,
        on_event,
        report,
        done: 0,
        total: jobs.len(),
        deferred: Vec::new(),
    };

    std::thread::scope(|scope| -> Result<(), AnalysisError> {
        let (jobs, known_hashes, in_flight, cancel) = (&jobs, &known_hashes, &in_flight, &cancel);
        scope.spawn(move || {
            pool.install(|| {
                jobs.par_iter().enumerate().for_each(|(i, job)| {
                    if cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let outcome = process(&job.path, known_hashes, in_flight, &opts.dsp);
                    let _ = tx.send((i, outcome));
                });
            });
        });
        for (i, outcome) in rx {
            let prev = known_files.get(&jobs[i].path).map(|k| k.track_id);
            if let Err(e) = sink.handle(i, &jobs[i], prev, outcome) {
                cancel.store(true, Ordering::Relaxed);
                return Err(e);
            }
        }
        sink.finish_deferred(jobs)
    })?;
    let Sink {
        store, mut report, ..
    } = sink;

    // Forget files that no longer exist (only those actually gone, not merely unreadable).
    let gone: Vec<String> = known_files
        .keys()
        .chain(failures.keys())
        .filter(|p| !seen.contains(*p) && !Path::new(p).exists())
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let gone_files = gone.iter().filter(|p| known_files.contains_key(*p)).count();
    store.remove_paths(&gone)?;
    report.removed = gone_files;

    report.elapsed = started.elapsed();
    Ok(report)
}

/// Groups failures by error message, most common first (for summaries).
pub fn summarize_errors(errors: &[(String, String)]) -> Vec<(String, usize)> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (_, e) in errors {
        *counts.entry(e.as_str()).or_default() += 1;
    }
    let mut v: Vec<(String, usize)> = counts.into_iter().map(|(e, n)| (e.to_owned(), n)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn write_tone(path: &Path, freq: f32, secs: f32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 22050,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for i in 0..(secs * 22050.0) as usize {
            let t = i as f32 / 22050.0;
            let beat = (t % 0.5) * 2.0;
            let s =
                0.3 * (TAU * freq * t).sin() + 0.2 * (TAU * 55.0 * t).sin() * (-beat * 8.0).exp();
            w.write_sample((s * 32767.0) as i16).unwrap();
        }
        w.finalize().unwrap();
    }

    fn scan_quiet(store: &mut Store, root: &Path, opts: &ScanOptions) -> ScanReport {
        scan(store, root, opts, |_| {}).unwrap()
    }

    fn opts() -> ScanOptions {
        ScanOptions {
            jobs: 2,
            ..Default::default()
        }
    }

    #[test]
    fn full_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        std::fs::create_dir(root.join("album")).unwrap();
        write_tone(&root.join("album/one.wav"), 220.0, 3.0);
        write_tone(&root.join("album/two.wav"), 330.0, 3.0);
        write_tone(&root.join("three.WAV"), 440.0, 3.0);
        std::fs::write(root.join("notes.txt"), "not audio").unwrap();
        std::fs::write(root.join("._one.wav"), "macOS metadata").unwrap();
        let mut store = Store::open_in_memory().unwrap();

        // First scan analyzes everything.
        let r = scan_quiet(&mut store, root, &opts());
        assert_eq!((r.found, r.analyzed, r.failed), (3, 3, 0), "{r:?}");
        let st = store.stats().unwrap();
        assert_eq!((st.tracks, st.files), (3, 3));
        assert_eq!(st.features, vec![("dsp".to_string(), DSP_VERSION, 3)]);
        // Untagged files are titled by file name.
        let titles: HashSet<String> = (1..=3)
            .map(|i| store.track(TrackId(i)).unwrap().unwrap().title)
            .collect();
        assert!(titles.contains("one") && titles.contains("three"));

        // Second scan: nothing to do.
        let r = scan_quiet(&mut store, root, &opts());
        assert_eq!((r.unchanged, r.analyzed, r.reused), (3, 0, 0), "{r:?}");

        // A copy and a move reuse the existing analysis.
        std::fs::copy(root.join("album/one.wav"), root.join("copy-of-one.wav")).unwrap();
        std::fs::rename(root.join("album/two.wav"), root.join("moved-two.wav")).unwrap();
        let r = scan_quiet(&mut store, root, &opts());
        assert_eq!(
            (r.analyzed, r.reused, r.removed, r.unchanged),
            (0, 2, 1, 2),
            "{r:?}"
        );
        let st = store.stats().unwrap();
        assert_eq!((st.tracks, st.files), (3, 4));

        // A corrupt file fails once and is then skipped until it changes.
        std::fs::write(root.join("broken.mp3"), vec![0u8; 5000]).unwrap();
        let r = scan_quiet(&mut store, root, &opts());
        assert_eq!(r.failed, 1, "{r:?}");
        let r = scan_quiet(&mut store, root, &opts());
        assert_eq!((r.failed, r.skipped_failed), (0, 1), "{r:?}");
        std::fs::remove_file(root.join("broken.mp3")).unwrap();
        scan_quiet(&mut store, root, &opts());
        assert_eq!(store.stats().unwrap().failures, 0);

        // --force re-analyzes every distinct recording once.
        let r = scan_quiet(
            &mut store,
            root,
            &ScanOptions {
                force: true,
                ..opts()
            },
        );
        assert_eq!(r.analyzed + r.reused, 4, "{r:?}");
        assert_eq!(r.analyzed, 3, "{r:?}");
    }

    #[test]
    fn duplicates_in_one_scan_are_analyzed_once() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["a.wav", "b.wav", "c.wav"] {
            write_tone(&dir.path().join(name), 262.0, 2.0);
        }
        let mut store = Store::open_in_memory().unwrap();
        let mut events = 0;
        let r = scan(&mut store, dir.path(), &opts(), |e| {
            if matches!(e, ScanEvent::File { .. }) {
                events += 1;
            }
        })
        .unwrap();
        assert_eq!((r.analyzed, r.reused), (1, 2), "{r:?}");
        assert_eq!(events, 3);
        assert_eq!(store.stats().unwrap().tracks, 1);
    }

    #[test]
    fn hash_ignores_container_but_not_audio() {
        let a = content_hash(&[0.1, 0.2], 44100);
        assert_eq!(a, content_hash(&[0.1, 0.2], 44100));
        assert_ne!(a, content_hash(&[0.1, 0.3], 44100));
        assert_ne!(a, content_hash(&[0.1, 0.2], 48000));
    }

    #[test]
    fn missing_root_is_an_error() {
        let mut store = Store::open_in_memory().unwrap();
        assert!(matches!(
            scan(&mut store, Path::new("/no/such/dir"), &opts(), |_| {}),
            Err(AnalysisError::Root { .. })
        ));
    }
}
