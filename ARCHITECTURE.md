# dhunn — Milestone 1 Architecture

Status: draft v2 · 2026-11-06 · Scope: Engine + CLI demo (see REQUIREMENTS.md §13)
v2: re-scoped to a private app for a small friend group — pooled library (≈10k–500k tracks), multi-user, history import, friend-based discovery, external (no-audio) discovery items.

## 1. Overview

```
 audio files ──► [analysis] ──► features ──► [store] ◄──► [index]
                    │                          │  ▲          │
          (DSP + ONNX embeddings)    (SQLite + vector blobs) (exact/HNSW + artist graph)
                                               │  │
 Spotify/Apple exports ──► [import] ──► plays ─┤  └── [meta] ◄── MusicBrainz / ListenBrainz
                                               │        (artist graph + external discovery items)
 user input ──► [profile] ◄── events ── [events log]
 (user, seeds, vibe,  │  (per user + friends)  ▲
  mood, mode,         ▼                        │
  adventurousness)  [steer] ──► candidates ──► [score] ──► [explain] ──► session
                                                                         │
                          [eval] ◄── imported history + logged sessions (held-out)
```

All runtime code is Rust. Python lives only in `ml/` for model export/training.

## 2. Workspace layout

```
dhunn/
├── Cargo.toml                 # workspace
├── crates/
│   ├── dhunn-core/            # shared types: TrackId, FeatureVec, Event, errors, config
│   ├── dhunn-audio/           # decode + resample + framing (symphonia, rubato)
│   ├── dhunn-dsp/             # spectral, MFCC, chroma, HPSS, onset, tempo, loudness
│   ├── dhunn-embed/           # ONNX inference (ort): audio embedding, CLAP text/audio
│   ├── dhunn-analysis/        # library scan: change detection, dedupe by audio hash, parallel audio→dsp(→embed)
│   ├── dhunn-store/           # SQLite (rusqlite) for metadata, features, events
│   ├── dhunn-meta/            # tag reading + artist-credit parsing (lofty) ✅; later MusicBrainz matching + artist graph, ListenBrainz similar-artists / discovery items
│   ├── dhunn-import/          # Spotify (extended + basic) / Apple parsers, history → library matching ✅
│   ├── dhunn-index/           # sound space: aspect-weighted vectors, exact kNN, aspect closeness ✅; later HNSW + graph
│   ├── dhunn-profile/         # long-term taste + session state
│   ├── dhunn-steer/           # sessions: next pick + feedback (basic drift model ✅); later modes, trajectory planning
│   ├── dhunn-score/           # candidate scoring + blending weights
│   ├── dhunn-explain/         # human-readable reasons
│   ├── dhunn-eval/            # simulated listeners ✅; offline metrics, A/B session export
│   └── dhunn-cli/             # `dhunn` binary
├── ml/                        # Python: model export to ONNX, later fine-tuning
├── models/                    # .onnx files (git-ignored, fetched by script)
└── docs/
```

Rationale: small crates with one job each — `dhunn-core` has no heavy deps, so the future `server`, Android (uniffi) and desktop bindings depend only on what they need. On Android, `dhunn-analysis` can be feature-gated to DSP-only if ONNX models are too heavy.

## 3. Analysis pipeline

**Decode** (`dhunn-audio`): symphonia → mono f32 → resample to 22.05 kHz (DSP) and the model's native rate (e.g. 48 kHz for CLAP, 24 kHz for MERT) with rubato. For embeddings, sample clips at the model's native window length (e.g. ~10 s for CLAP) from intro/middle/late sections and average them; DSP summaries use the full track.

**DSP** (`dhunn-dsp`, realfft) — *implemented*: frame 2048 / hop 512, periodic Hann, centered zero-padded frames. Per-frame features follow librosa 1.0's definitions.

| Group | Features | Summary per track |
|---|---|---|
| Spectral | centroid, bandwidth, rolloff (85%), flatness, ZCR, 7-band octave contrast | mean, std |
| Timbre | 20 MFCC (128-band Slaney mel, log, ortho DCT) + Δ (9-frame) | mean, std (→ 80 dims) |
| Harmony | STFT chroma (`chroma_stft`, tuning fixed at 0), Krumhansl–Kessler key + mode, key clarity | mean chroma, tonic-relative chroma (key-invariant), key, clarity |
| Percussion | median-filter HPSS (kernel 31, soft masks) → harmonic ratio; onset strength; low-band (<150 Hz) percussive energy ("thump") share and dB flux | ratios, mean/std, p90 |
| Rhythm | tempo (8 s autocorrelation tempogram + 120 BPM log-normal prior), pulse clarity | scalars |
| Dynamics | frame RMS → dBFS, dynamic range (p95 − p10 of non-silent frames) | mean/std, scalar |
| Pitch | YIN on the re-synthesized harmonic part: median f0, range (p10–p90 semitones), voiced fraction, periodicity | scalars |

Output: `DspSummary` (named, human-readable — used for explanations) and its flat `FeatureVec` (134 dims, `dsp` v1, fixed order). Tracks longer than 10 min are analyzed on a centered 10-min excerpt (`--full` overrides). Speed: ~0.7 s for a 4-min track on one machine (HPSS is the largest stage).

**Validation:** `uv run ml/validate_dsp.py <files…>` compares 125 summary values against librosa 1.0 (pinned). On 10 native 22.05 kHz recordings (pop, orchestral, piano, drums, solo vocals, click track) all 125 match within tolerance, most to <0.01%. Inputs at other rates differ slightly because librosa resamples with soxr and dhunn with rubato (verified clean: imaging < −110 dB); this is consistent across the library, so it doesn't affect similarity. Known algorithm behavior, not bugs: tempo can land on double/half the felt tempo (same as librosa), and key detection is ambiguous on short or static passages.

Not yet: tuning estimation for chroma, true LUFS (K-weighted) loudness, beat tracking / regularity.

**Embeddings** (`dhunn-embed`, `ort`):
- `audio_emb`: general-purpose music embedding (MERT-class or similar, chosen in `ml/` for license + size).
- `clap_audio`: CLAP audio embedding (shares space with CLAP text → enables vibe text search).
- Derived heads (small linear/MLP, trained later): valence, arousal, danceability. M1 can use CLAP zero-shot prompts ("happy", "sad", "energetic"…) as a placeholder.

**Parallelism:** rayon over tracks; embedding batched. Resumable — each track keyed by content hash (xxhash of decoded PCM sample) + analysis version, so re-runs skip done work.

## 4. Storage (`dhunn-store`)

SQLite (single file, WAL mode, numbered migrations via `PRAGMA user_version`). At group scale (≤500k tracks, ≤20 users) this is enough for the shared server too.

**Implemented** (migration 1):

```
tracks(id, content_hash UNIQUE, title, album, year, duration_s, mb_recording_id, added_at)
track_files(path PK, track_id, size, mtime_ns)       -- many files → one track (copies, moves)
artists(id, name, name_key UNIQUE, mb_artist_id)      -- name_key = normalized name
track_artists(track_id, artist_id, role, position)    -- role: primary | featured
feature_schemas(kind, version, names JSON)            -- dimension names, stored once per layout
features(track_id, kind, version, vec BLOB f32 LE, summary JSON, computed_at)
scan_failures(path PK, size, mtime_ns, error, failed_at)
```

A **track** is a recording identified by `content_hash` (xxh3 of its decoded samples + sample rate), so renamed, moved, re-tagged and duplicated files share one track and one analysis. Tracks survive their files being deleted (listening history refers to tracks).

**Implemented** (migration 2):

```
users(id, name UNIQUE NOCASE, adventurousness 0..1, share_profile, created_at)
history_plays(id, user_id, ts, source, raw_artist, raw_title, raw_album, ms_played,
              skipped NULL, reason_end, shuffle, source_uri,
              track_id NULL, artist_id NULL, match_kind)   -- exact | fuzzy | artist | none
              UNIQUE (user_id, source, ts, raw_artist, raw_title)  -- re-import is a no-op
```

**Implemented** (migration 3):

```
sessions(id, user_id, started_at, ended_at, mode, params JSON)
events(id, session_id, user_id, ts_ms, track_id, kind, position_ms, payload JSON, schema_version)
       -- kind: play | complete | skip | like | dislike | further | back | pause | resume | quit | error
```

**Planned** (later migrations, with the steps that need them):

```
tracks.added_by -> users
artist_links(a, b, kind, weight, source)     -- collab, member_of, influence, same_label, lb_similar
embeddings(track_id, model, version, vec BLOB)
external_items(id, mb_recording_id, mb_artist_id, title, artist, source, link)  -- no audio
```

## 4b. Library scan (`dhunn-analysis`) — *implemented*

`dhunn analyze <dir>` walks the folder (following symlinks, skipping `._*` files) and, per audio file:
1. **Unchanged** (same size + mtime, track has current `dsp` features) → skip without reading.
2. **Failed before** and unchanged since → skip (retried once the file changes, or with `--force`).
3. Otherwise decode and hash. **Known audio** → link the file to the existing track (a re-tagged file also refreshes the track's tags). **New audio** → resample, DSP-analyze, read tags (title falls back to the file name; artist credits parsed for "feat."/"ft."), save in one transaction.
4. Files gone from disk are forgotten (only if the path really no longer exists — an unreadable folder is never treated as deleted).

Decoding/analysis runs on a bounded worker pool (default: half the cores, max 6); all writes happen on one thread. Copies of the same audio within one scan are analyzed once. Measured on the FMP corpus (553 files, mostly 10 s–4 min): 65 s for a first scan with 4 workers, ~1 GB peak RAM; rescan 0.0 s.

## 4a. History import (`dhunn-import`) — *implemented*

- **Spotify extended streaming history** (`Streaming_History_Audio_*.json`): `ts` (UTC, end of play), `ms_played`, track/artist/album, `spotify_track_uri`, `reason_end`, `skipped`, `shuffle`. Podcasts/audiobooks ignored; network fields (`ip_addr`) never read.
- **Spotify basic account data** (`StreamingHistory_music_*.json`): `endTime`, `artistName`, `trackName`, `msPlayed` — no skip flag. Arrives in days rather than weeks.
- **Apple Music** (`Apple Music Play Activity.csv`, `Apple Music - Play History Daily Tracks.csv`): read by header name, `PLAY_END` events only; daily aggregates expanded into individual plays. *Built from the documented columns, not yet checked against a real export.*
- **Matching** (`matcher.rs`), per distinct (artist, title): titles lose version notes ("- Remastered 2009", "(feat. X)", "[Mono]") and punctuation but keep "live"/"acoustic"/"remix"; artists lose a leading "The", "&" → "and"; every credited artist is tried. Then: **exact** → **fuzzy** (Jaro–Winkler ≥ 0.92 on title within the artist, ≥ 0.90 on artist within the title) → untagged library tracks by exact title (≥ 12 chars, so "Intro" never matches) → **artist only** → **none**. Artist-only and unmatched plays are retried after every `dhunn analyze`, so they get picked up as the library grows.
- **Listen vs skip:** the export's `skipped` flag when present, otherwise < 30 s = skip (so a full play of a 20 s track is a listen).
- **Untagged files** named `Artist - Title` (optionally with a track number) get artist/title from the file name at scan time.
- **Uses (next steps):** seed long-term taste (listens, skips, recency); artist affinity for unowned music; "loved but not in library" suggestions; held-out sessions for eval.

**Simulated listener** (`dhunn-eval::simulate`, `dhunn dev sample-history`): a hidden taste of N "home" tracks in z-scored DSP space; library picks weighted by affinity × similarity to the previous track (sessions drift), 12% random picks (shuffle/radio), 25% plays of popular songs not in the library, 30% of library names re-spelled the streaming-service way, skip probability rising as affinity falls. Writes Spotify-format JSON plus `dhunn_ground_truth.json` (home tracks + every track's affinity) for evaluation. Deterministic per seed. On the FMP library (120 days): 2,154 plays, 78% matched, and the most-listened tracks are the hidden home tracks.

## 5. Index (`dhunn-index`)

**Implemented (DSP only):** the 134 DSP dimensions are z-scored (clipped to ±4) and grouped into seven **aspects** — timbre (MFCC), brightness (centroid/rolloff/flatness/contrast/ZCR), harmony (key-relative chroma, mode, clarity), rhythm (tempo, pulse, onsets), punch (HPSS balance, thump), dynamics, pitch. Each aspect is scaled by `weight / √dims`, so it counts by its weight however many dimensions it has (timbre 80, rhythm 4); default weights: timbre, brightness, rhythm 1.0 · punch 0.8 · harmony 0.7 · dynamics, pitch 0.5. Rows are L2-normalized (dot = cosine). Search is exact and parallel; the space is rebuilt from the store at startup (milliseconds at FMP scale; persisting it is a later optimization).

**Explanations** use per-aspect *closeness* `1 − d / typical d` (typical = mean distance between random pairs, sampled at build time), so an aspect is only cited when two tracks are unusually close on it — plain cosine on a 3-dim aspect looked "99% similar" for most pairs.

**Planned:**
- **Combined sound space:** per-track vector = concat of normalized blocks [dsp (z-scored, PCA→32), audio_emb (→64), clap (→64)] with per-block weights. Weights live in config so eval can tune them.
- **Search:** `VectorIndex` trait with `Exact` (brute force, SIMD, parallel with rayon; the default — 500k × 160 f32 ≈ 320 MB, roughly 10–20 ms per query single-threaded and a few ms in parallel, inside the 50 ms budget) and `Hnsw` (e.g. `hnsw_rs` or `usearch`) as an optional impl for the top end or low-RAM devices.
- **Artist graph:** petgraph from `artist_links` (MusicBrainz + ListenBrainz similar-artists) + derived "sounds-like" edges (artist centroid kNN). Used for stepping stones and explanations.
- **External items:** no audio → not in the sound space. Placed via the artist graph: an item's position is approximated by the centroid of its nearest owned artists, and it is only ever surfaced in the separate "Discover" list, never mixed into a playable session.
- **Region labels:** k-means clusters over the sound space, labelled by dominant tags/genres + top CLAP descriptors → names like "warm downtempo / neo-soul" for explanations and destinations.

## 6. Profile (`dhunn-profile`)

- **Per user.** Every profile and session is keyed by `user_id`.
- **Long-term taste:** weighted set of sound-space centroids (mixture, k≈3–8) learned from positive events and imported history with time decay; plus a "frontier" (regions touched but not settled); plus artist affinities (covers unowned artists from imported history).
- **Adventurousness** a ∈ [0, 1] per user → default σ and `w_novel` (e.g. σ = σ_min + a·(σ_max − σ_min)).
- **Friend profiles:** taste mixtures of other users who opted in (`share_profile`). Overlap between two mixtures defines "bridge" regions.
- **Session state:** last N tracks, current position (EMA of recent track vectors), velocity (direction of drift), recent reactions, exploration speed σ.
- Updates are pure functions of `(state, event) → state` — easy to test and replay from the event log.

## 7. Steering (`dhunn-steer`)

**Implemented — basic drift model** (step 4; the planner below extends it in step 6). A session keeps an **anchor** (starts at the seeds' centroid), an **exploration** level `e` (from adventurousness: `0.1 + 0.5·a`) and recent **avoid** points. Each pick scores the ~60–200 tracks nearest the anchor:

```
score = (1 − e)·sim(anchor) + 1.0·sim(previous track) − e·sim(recently heard centroid) − 0.5·Σ w·sim(avoid)
```

then samples among the best 3–8. Continuity with the previous track plus repulsion from what was heard makes sessions move *forward* along a path instead of circling or hopping across the anchor. Feedback: **finish** pulls the anchor 20% toward the track; **like** 40%; an **early skip** adds an avoid point (weight falls with how late the skip was) and eases exploration down; **dislike** bans the track and adds a strong avoid point; **further** raises `e` by 0.2 and extrapolates the anchor past the current track in the direction of travel; **back** lowers `e` and restores the previous anchor. Never in a session: seeds, a second version of the same song (same normalized artist + title — WAV/MP3 copies, remasters), more than 2 tracks in a row by one artist; across sessions, tracks played in the last 36 h are skipped while alternatives exist.

**Planned (step 6):**

Each mode produces a **target** and a **step budget**:

| Mode | Target | Step budget |
|---|---|---|
| Drift (default) | point slightly beyond current position along frontier/novelty direction | small, σ adaptive |
| Surprise me | sampled distant-but-coherent region (high-quality cluster not yet visited) | one big hop, then small |
| Something new | nearest region with unheard artists | medium |
| Vibe | CLAP text vector / mood controls / seed centroid / context prior → mapped into sound space | medium until reached |
| Destination | named region/artist/genre | path planned in advance |
| Toward friend | nearest centroid of the friend's taste mixture not already in mine | path planned in advance, like Destination |
| Bridge (group) | overlap region between two or more users' mixtures | small; stay near the overlap |

**Path planning (destination):** A* over a kNN graph of cluster centroids (and the artist graph as alternate edges) with edge cost = distance, constrained so each hop ≤ σ. Each waypoint yields a few tracks before advancing.

**Adaptation:** skip early → shrink σ and pull back toward last liked point; "go further" → grow σ; "too far" → step back one waypoint, shrink σ.

## 8. Scoring (`dhunn-score`)

For candidates = kNN around the next step point (∪ graph neighbours):

```
score = w_target · sim(track, step_point)
      + w_taste  · max_sim(track, taste_centroids)
      + w_smooth · sim(track, last_track)
      + w_novel  · novelty(track, user history)
      + w_friend · friend_signal(track, user) -- friends' plays/likes weighted by taste overlap with me
      + w_hist   · history_affinity(track, user) -- imported history: artist/track affinity
      - w_repeat · recency_penalty
      - w_artist · same_artist_streak_penalty
```

`friend_signal` is small-group collaborative filtering: no matrix factorization needed at ≤20 users — a weighted sum over friends of (their affinity for the track × overlap between their taste and mine). Weights come from mode presets + the user's adventurousness + a blend schedule (behavioral/friend weight grows with data). Final pick: sample from top-k with temperature (deterministic with seed for eval).

## 9. Explanations (`dhunn-explain`)

Each pick returns structured `Reason`s → rendered text:
- top contributing score term,
- 1–2 named scalars with largest similarity to previous track (e.g. "similar tempo 84 BPM", "warm low end"),
- steering context ("moving toward: neo-soul, step 3/6"),
- graph link if used ("collaborated with X"),
- friend context if used ("Aarav has this on repeat; close to your late-night stuff") — only from friends who share their profile.

## 10. Events & logging

**Implemented:** `dhunn play` logs play, complete, skip (with position), like, dislike, further, back, pause/resume, quit and error to `events`, each written immediately. These are the real feedback for tuning; profile building (step 6) will read them alongside imported history.

Originally planned: every CLI action emits an `Event` (play, skip{at_ms}, complete, replay, like, dislike, never, more_like, further, back, too_far, vibe_set) to the store. Signals only a real player can produce (volume changes, seek) get event kinds reserved now and are emitted once the apps exist. Profile/session state can be rebuilt by replaying events. Versioned schema from day one.

## 11. Evaluation (`dhunn-eval`)

- **Real data first:** imported Spotify history (with `skipped`, `ms_played`, `reason_end`) restricted to tracks matched in the library gives real sessions; split by time (train on older, hold out the most recent weeks).
- **Offline:** given those, event logs or public datasets (e.g. sessions over FMA tracks), compute hit-rate@k / NDCG@k for next-track, skip-prediction AUC, transition smoothness, novelty, diversity, long-tail coverage, repetition rate, steering reach (target reached? max hop ≤ σ?).
- **Simulated users:** synthetic listeners with hidden taste regions + skip models (✅ implemented, see §4a), to regression-test steering before real users exist; their ground truth gives "did we recommend what they like" metrics.
- **Listening tests:** export pairs of sessions (variant A vs B) as blind playlists; record preferences.
- `dhunn eval --config variants.toml` prints a comparison table; every algorithm change runs it.

## 12. CLI (`dhunn-cli`, clap)

```
dhunn user add|list|set <name> [--adventure 0.4]   # ✅
dhunn [--db <file>] …                  # library DB; default: platform data dir, or $DHUNN_DB
dhunn analyze <dir> [--force] [-j N] [-q] [--added-by <user>]  # scan + analyze (✅ except --added-by)
dhunn stats                            # library totals + recent failures (✅)
dhunn import spotify|apple <path> --user <name>   # ✅ history export → history_plays + matching
dhunn history --user <name>          # ✅ match rates, top artists/tracks
dhunn dev sample-history --out <dir> [--days N --seed S --regions K]  # ✅ simulated listener
dhunn similar <song words> [-n 10]    # ✅ nearest tracks + which aspects they share
dhunn session [--seed <song words>]... [--user] [--len N] [--adventure A] [--rng S]  # ✅ print a planned session
dhunn play [--seed <song words>]... [--user] [--adventure A]  # ✅ listen: space pause · n next · l like · d dislike · f further · b back · ± volume · q quit
dhunn index                           # (not needed yet: the space is built at startup) clusters, artist graph
dhunn mb-sync                         # optional MusicBrainz/ListenBrainz matching (rate-limited)
dhunn session --user <name>
             --seed <track>... | --vibe "rainy night drive" | --mood e=0.3,v=-0.2
             --mode drift|surprise|new|dest|friend|bridge --to "jazz" | --friend <name>
             [--adventure 0.7] --len 20 [--interactive]
dhunn discover --user <name>          # unowned suggestions (ListenBrainz + history gaps)
dhunn explain <track>
dhunn eval [--config variants.toml]
dhunn inspect <track>                 # dump features
```
`--interactive` plays nothing in M1 (or optionally via `rodio`) but accepts keys: s=skip, l=like, f=further, b=back, t=too far — all logged.

## 13. Key dependencies

symphonia, rubato, realfft, ndarray, rayon, ort (ONNX Runtime), rusqlite, lofty, petgraph, hnsw_rs/usearch, linfa (k-means, PCA) or hand-rolled, serde, serde_json, csv (history imports), strsim (fuzzy matching), xxhash-rust (content hash), ureq (MusicBrainz/ListenBrainz HTTP), clap, tracing, rodio (optional).

## 14. Build order

1. ✅ `core`, `audio`, `dsp` + `inspect` command — validated against librosa (`ml/validate_dsp.py`).
2. ✅ `store`, `meta` (tags), `analysis`, `analyze` + `stats` commands.
3. ✅ `import` (Spotify extended + basic, Apple best-effort) + track matching + simulated-listener sample data.
4. ✅ `index` (exact, aspect-weighted) + basic steering session + **`dhunn play`** (rodio playback, keyboard controls, event logging) — pulled forward so the app can be used daily and real feedback accumulates.
5. `eval` on imported history + simulated users — baseline numbers.
6. `profile` (per user, adventurousness) + `steer` (drift, destination) + `explain`.
7. `embed` (CLAP + audio model) → vibe text, surprise/new modes; re-run eval.
8. Friend modes (`friend`, `bridge`, `friend_signal`) once two or more users have imported history.
9. `meta` (MusicBrainz + ListenBrainz graph), `discover`, interactive mode + event logging. HNSW only if exact search gets slow.

## 15. Decisions to revisit
- Exact embedding models (license, size, Android feasibility).
- Fixed vs learned block weights in the sound space (learn once we have logs).
- How the pooled library is synced to the shared server and streamed to clients (post-M1).
- Within-group privacy defaults for shared profiles.
