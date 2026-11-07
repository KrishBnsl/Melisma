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
│   ├── dhunn-embed/           # ONNX inference (ort): CLAP audio + text embeddings ✅
│   ├── dhunn-analysis/        # library scan: change detection, dedupe by audio hash, parallel audio→dsp(→embed)
│   ├── dhunn-store/           # SQLite (rusqlite) for metadata, features, events
│   ├── dhunn-meta/            # tag reading + artist-credit parsing (lofty) ✅; later MusicBrainz matching + artist graph, ListenBrainz similar-artists / discovery items
│   ├── dhunn-import/          # Spotify (extended + basic) / Apple parsers, history → library matching ✅
│   ├── dhunn-index/           # sound space: aspect-weighted vectors, exact kNN, aspect closeness, library map of places + routes ✅; later HNSW + artist graph
│   ├── dhunn-profile/         # long-term taste + session state
│   ├── dhunn-profile/         # per-user taste regions, dislikes, learned skip model ✅
│   ├── dhunn-steer/           # sessions: next pick + feedback (basic drift model ✅); later modes, trajectory planning
│   ├── dhunn-score/           # candidate scoring + blending weights
│   ├── dhunn-explain/         # human-readable reasons
│   ├── dhunn-eval/            # simulated listeners, history replay, simulated sessions, logged-session summary ✅; later A/B session export
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

**Embeddings** (`dhunn-embed`, `ort`) — **implemented (step 7): CLAP.**
- **Model:** LAION CLAP trained on music (HTSAT-base audio encoder + RoBERTa text encoder, 512-d shared space), from LAION's original checkpoint `music_audioset_epoch_15_esc_90.14.pt` (CC0), converted in `ml/export_clap.py`. The Hugging Face conversion `laion/larger_clap_music` is unusable: its projection layers are untrained (zero biases, init-scale weights, logit scales ≈ 1) and its encoders don't match the original, so every text scored ≈ 0.01 against every track. The export loads the original strictly (only HTSAT's classification heads and rebuilt buffers may be left over), checks that the original model's own mel filters equal the front end baked into the graph (|Δ| 4e-9), and that ONNX equals transformers (cosine 1.000000, audio and text).
- **Front end inside the graph:** `audio.onnx` takes raw mono 48 kHz audio; reflect padding, a windowed DFT as one strided convolution, the Slaney mel filters and the dB conversion are constants in the graph. Rust only decodes, resamples (rubato) and cuts clips.
- **Per track:** three 10 s clips at 15/45/75 % of the track (shorter tracks repeat-padded, as in training), embedded in one batch and averaged. ~2.3 tracks/s on an M-series CPU (479 tracks in 205 s), so ~70 min per 10k tracks, once. `dhunn embed` is separate from `analyze` (models stay optional; e.g. DSP-only on Android) and resumable.
- **Text:** RoBERTa BPE via the pure-Rust `tokenizers` crate (the tokenizer's own padding/truncation switched off — it padded batches with an all-ones mask and changed embeddings by up to 13%); ≤ 77 tokens.
- **Runtime:** ONNX Runtime CPU. `ort`'s default "most efficient device" policy picks CoreML on macOS, which can't run HTSAT, so sessions prefer CPU explicitly. One ORT environment per process.
- Later: a general audio embedding (MERT-class, licence permitting) if CLAP proves too coarse; mood heads (valence/arousal) trained on CLAP.

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
       -- mode: drift | smoke (automated audio check; ignored by eval)
```

**Implemented** (migration 4):

```
eval_runs(id, created_at, label, report JSON)   -- every `dhunn eval` run, so the next one shows changes
```

**Implemented** (migration 5):

```
embeddings(track_id, model, version, vec BLOB, PRIMARY KEY (track_id, model))   -- CLAP vectors (512 × f32)
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
- **Combined sound space — implemented (step 7):** per-track vector = [√(1−s)·DSP unit vector, √s·CLAP], CLAP centred on the library mean first (raw CLAP vectors share a large common direction). Similarity = (1−s)·DSP cosine + s·CLAP cosine. `s` = 0.25 by default (`DEFAULT_CLAP_SHARE`, chosen in eval, §11); DSP only when fewer than half the tracks have embeddings. Aspect explanations still use the DSP part. Text queries match the raw CLAP vectors (`clap_matches`).
- **Search:** `VectorIndex` trait with `Exact` (brute force, SIMD, parallel with rayon; the default — 500k × 160 f32 ≈ 320 MB, roughly 10–20 ms per query single-threaded and a few ms in parallel, inside the 50 ms budget) and `Hnsw` (e.g. `hnsw_rs` or `usearch`) as an optional impl for the top end or low-RAM devices.
- **Artist graph:** petgraph from `artist_links` (MusicBrainz + ListenBrainz similar-artists) + derived "sounds-like" edges (artist centroid kNN). Used for stepping stones and explanations.
- **External items:** no audio → not in the sound space. Placed via the artist graph: an item's position is approximated by the centroid of its nearest owned artists, and it is only ever surfaced in the separate "Discover" list, never mixed into a playable session.
- **Region labels:** k-means clusters over the sound space, labelled by dominant tags/genres + top CLAP descriptors → names like "warm downtempo / neo-soul" for explanations and destinations.

## 6. Profile (`dhunn-profile`)

**Implemented (step 6a).** Rebuilt from the user's imported history plus `dhunn play` sessions every time it's needed (milliseconds at this scale; nothing stored, always current):
- **Track weights:** each play adds `decay × signal`, decay = half-life 90 days, signal = +1 listened / −0.3 skipped (< 30 s) / +2 like / −3 dislike.
- **Taste regions:** weighted spherical k-means over positively weighted tracks, k = √(n/4) clamped to 1–6, deterministic start. `taste_of(v)` = closeness to the nearest region (best at finding what you'll play next); `overall_of(v)` = closeness to the share-weighted centre (best at telling listens from skips).
- **Dislikes** with no later like → excluded from every session.
- **Skip model:** logistic regression on 9 features — previous-track similarity, taste, overall taste, track and artist listen rate (smoothed log-odds, leave-one-out while training), track plays, share of recent skips, similarity to a recent skip, position in session. Trains in milliseconds; needs ≥ 30 plays with both listens and skips. Steering uses it *without familiarity or taste*: track/artist rates are capped at your average (it still avoids what you tend to skip, but never favours a track for being familiar) and the taste features are neutralized (taste is steering's own term). A GRU replaces it only when there are thousands of logged sessions and eval shows it wins.
- `dhunn profile` shows regions (with their top tracks), dislikes and the model's weights.

Originally planned:

- **Per user.** Every profile and session is keyed by `user_id`.
- **Long-term taste:** weighted set of sound-space centroids (mixture, k≈3–8) learned from positive events and imported history with time decay; plus a "frontier" (regions touched but not settled); plus artist affinities (covers unowned artists from imported history).
- **Adventurousness** a ∈ [0, 1] per user → default σ and `w_novel` (e.g. σ = σ_min + a·(σ_max − σ_min)).
- **Friend profiles — implemented (step 8, `dhunn-profile::group`):** built for every other user who shares their profile (`dhunn user set <name> --share yes|no`), from their history and their non-`--private` sessions. **Overlap** = how much of each taste lies close to some region of the other's (share × nearest-region similarity, averaged both ways; 0..1). **Trust** = 0.5 + 0.5 × overlap: any friend counts, closer ones more (with the first formula — products of region shares — two real people scored 2–5% and friends could never influence a pick).
- **Session state:** last N tracks, current position (EMA of recent track vectors), velocity (direction of drift), recent reactions, exploration speed σ.
- Updates are pure functions of `(state, event) → state` — easy to test and replay from the event log.

## 7. Steering (`dhunn-steer`)

**Implemented — basic drift model** (step 4; the planner below extends it in step 6). A session keeps an **anchor** (starts at the seeds' centroid), an **exploration** level `e` (from adventurousness: `0.1 + 0.5·a`) and recent **avoid** points. Each pick scores the ~60–200 tracks nearest the anchor:

```
score = (1 − e)·sim(anchor) + 1.0·sim(previous track) − e·sim(recently heard centroid) − 0.5·Σ w·sim(avoid)
```

then samples among the best 3–8. Continuity with the previous track plus repulsion from what was heard makes sessions move *forward* along a path instead of circling or hopping across the anchor. Feedback: **finish** pulls the anchor 20% toward the track; **like** 40%; an **early skip** adds an avoid point (weight falls with how late the skip was) and eases exploration down; **dislike** bans the track and adds a strong avoid point; **further** raises `e` by 0.2 and extrapolates the anchor past the current track in the direction of travel; **back** lowers `e` and restores the previous anchor. Never in a session: seeds, a second version of the same song (same normalized artist + title — WAV/MP3 copies, remasters), more than 2 tracks in a row by one artist; across sessions, tracks played in the last 36 h are skipped while alternatives exist.

**Implemented (step 6a) — profile in the score:**

```
score += taste_weight · (1 − e)² · taste_of(track)   + retention_weight · (1 − e) · (P(listen) − 0.5)
```

with `taste_weight` 0.5 and `retention_weight` 1.0. The profile's pull fades fast as exploration rises, so adventurousness (and "further") still lets a session leave home. Without the fade, or with taste also counted inside the skip model, every adventurousness level gave the same sessions (variety −17 points at a=0.8).

**Implemented (step 6b) — modes.** A **library map** (`dhunn-index::places`) groups tracks into places (spherical k-means, ~1 per 8 tracks, 4–300, fitted on ≤ 20k sampled tracks) linked to their 6 nearest places. Modes that travel plan a **journey** of places on the first pick:

| Mode | Where it goes | How |
|---|---|---|
| drift (default) | onward from the seeds | as above |
| usual | stays with what you play most | exploration 0.05, double taste weight |
| surprise (`--mode surprise`, or `s` while playing) | a coherent place far from here, mostly unplayed, that your taste still reaches (score = coherence + unplayed share − closeness to here + ½·taste) — one of the best 3 at random | one big hop (continuity ignored for that pick), then settles (exploration halved) |
| something new | the nearest place where ≥ 50% of tracks are new to you | route of small steps; +0.3 score for unplayed tracks |
| toward (`--to <song or artist>`) | a track, or the centre of an artist's tracks | route of small steps; settles on arrival |

**Routes:** Dijkstra over the place graph with hop cost (1 − similarity)², so many small steps beat one big jump. A journey leaves its starting place on the first pick and spends at most 2 picks per stop (paced to arrive within ~12). **Adaptation:** a skip on the way means "not this part of the way" — move on sooner (lingering at stops the listener skipped stalled journeys in eval: 79% arrived vs 89% after the change); "further" jumps to the next stop; "back" returns to the previous one. Explanations name the journey: "on the way to AC/DC · stop 3 of 5", "arrived near …", "surprise: a jump to a part of your library you haven't explored".

**Vibe — implemented (step 7):** `--vibe "rainy night drive"` embeds the text with CLAP, takes the 10 tracks whose CLAP vectors match it best, and travels to their (match-weighted) centre like any destination — a route of small steps from where you are, not a jump. `dhunn vibe "<text>"` lists the matches. On FMP: "heavy distorted rock guitars" → Nirvana, AC/DC, Apocalyptica, Metallica; "solo piano, calm" → Chopin Op. 28/4, Schumann Träumerei; "a crowd clapping" → the applause recordings; "drums only" fails (FMP has hardly any).

**Friends — implemented (step 8):**
- **Friend signal** in every session (unless `--solo`): `+ 0.3 × Σ_friends trust × affinity` (affinity = their time-decayed weight for the track, scaled so their 90th percentile = 1; skips and dislikes negative). It doesn't fade with exploration — friends are a good place to explore. Picks credit the friend: "Aarav plays this a lot" / "Aarav likes this".
- **Toward a friend** (`--friend <name>`): destination = the friend's region scoring highest on share × (1 − closeness to my nearest region) — the part of their taste I don't have yet — reached by a planned route.
- **Listening together** (`--with <name>`, repeatable): the taste term uses whoever a track suits *least*; with no other mode, the session starts by travelling to the meeting point (among all members' region centres and pairwise midpoints, the one whose weakest listener likes it most).
- **`dhunn friends`:** overlaps, and friend-sourced novelty — tracks friends love that I've never played, ranked by friend signal × closeness to my taste.
- **Privacy:** only users with `share_profile` are ever used; `--private` sessions count for the listener's own profile, never for friends.

**Still planned:** artist-graph edges in routes (step 9).

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

`friend_signal` (implemented, §7) is small-group collaborative filtering: no matrix factorization needed at ≤20 users — a weighted sum over friends of (their affinity for the track × trust, from overlap between their taste and mine). Weights come from mode presets + the user's adventurousness + a blend schedule (behavioral/friend weight grows with data). Final pick: sample from top-k with temperature (deterministic with seed for eval).

## 9. Explanations (`dhunn-explain`)

Each pick returns structured `Reason`s → rendered text:
- top contributing score term,
- 1–2 named scalars with largest similarity to previous track (e.g. "similar tempo 84 BPM", "warm low end"),
- steering context ("moving toward: neo-soul, step 3/6"),
- graph link if used ("collaborated with X"),
- friend context if used ("Aarav plays this a lot" ✅) — only from friends who share their profile.

## 10. Events & logging

**Implemented:** `dhunn play` logs play, complete, skip (with position), like, dislike, further, back, pause/resume, quit and error to `events`, each written immediately. These are the real feedback for tuning; profile building (step 6) will read them alongside imported history.

Originally planned: every CLI action emits an `Event` (play, skip{at_ms}, complete, replay, like, dislike, never, more_like, further, back, too_far, vibe_set) to the store. Signals only a real player can produce (volume changes, seek) get event kinds reserved now and are emitted once the apps exist. Profile/session state can be rebuilt by replaying events. Versioned schema from day one.

## 11. Evaluation (`dhunn-eval`) — *implemented (step 5)*

`dhunn eval` runs three checks in about 2 s on a 500-track library, prints them, saves the report to `eval_runs` and shows every number's change since the previous run in brackets. `--param name=value` tries a steering constant without changing code (`SteerParams`: smoothness, onward, heard_memory, avoid_penalty, finish_pull, like_pull); `--label` names a run.

1. **Next-track prediction on history** (`replay`). The user's matched plays are split by time: the last `--holdout-days` (14) are test, the rest train. Test plays are cut into sessions (pause > 30 min). At each step every model ranks the whole library given the session so far, and we record where the next *listened* track landed (hit@10, hit@50, MRR; skipped tracks aren't targets but still feed the session, as in reality). Models: random (analytic), popular (train listens), your taste (centroid of train listens), previous track (nearest by sound), steer (`Session::ranked` with the real plays observed and finished/skipped fed back). Plus skip AUC: how well taste similarity and previous-track similarity separate listens from skips.
2. **Simulated listeners** (`listeners`). Listeners with hidden tastes (`taste.rs`: home tracks in plain z-scored DSP space, *not* the recommender's weighted space) start sessions from one of their top-5% tracks; every pick is "played" to them and they skip / finish / like / dislike by affinity, which is fed back. Pickers: random, nearest chain (always the closest unplayed track to the last one), steer at adventurousness 0.2/0.5/0.8. Metrics: taste percentile of picks, skip and like rate, smooth (mean consecutive similarity), worst hop, travel (1 − similarity of the last pick to the start), per-listener variety, same-artist pairs. 16 listeners × 6 sessions × 25 tracks by default; run-to-run noise is about ±1–2 taste points, so smaller differences mean nothing.
3. **Your dhunn sessions** (`logged`): plays, skip rate (and skips in the first 30 s), likes, dislikes, further/back, all time and last 7 days — the number that matters once the app is used daily.

**Baseline** (FMP library, 479 tracks, sample history of 2,154 plays, default parameters):

| next track (113 predictions) | hit@10 | hit@50 | MRR |
|---|---|---|---|
| random | 2.1% | 10.5% | 0.011 |
| popular | 65.5% | 83.2% | 0.287 |
| your taste | 38.1% | 71.7% | 0.116 |
| previous track | 42.5% | 63.7% | 0.186 |
| steer | 42.5% | 52.2% | 0.187 |

Skip AUC: your taste 0.84, previous track 0.74.

| simulated sessions | taste | skipped | liked | smooth | worst hop | travel | variety |
|---|---|---|---|---|---|---|---|
| random | 51.0 | 46.0% | 2.7% | 0.01 | −0.50 | 0.98 | 86.7% |
| nearest chain | 79.4 | 16.4% | 12.8% | 0.83 | 0.61 | 0.69 | 47.1% |
| steer a=0.2 | 82.7 | 13.8% | 12.0% | 0.79 | 0.52 | 0.65 | 58.5% |
| steer a=0.5 | 82.3 | 14.6% | 12.4% | 0.79 | 0.48 | 0.68 | 60.4% |
| steer a=0.8 | 78.2 | 17.5% | 11.2% | 0.77 | 0.41 | 0.72 | 67.0% |

What it says:
- Steering's feedback works: it beats always-play-the-nearest on taste (+3), skips (about 2 points fewer) and variety (+11 to +20 points), at a small cost in smoothness.
- **Adventurousness barely changes how far a session travels** (0.65 → 0.72) while costing taste. `heard_memory=40` makes it work (a=0.8 travels 0.85) for similar cost — the first tuning candidate.
- **Steering ignores long-term taste**: on history, "popular" and "your taste" beat it at hit@50 because the session only knows its seeds and what it played. Mixing the profile in is step 6.
- Caveats: the history is simulated (its listener picks by previous-track similarity and replays favourites, which flatters those two models); the simulated listeners' taste comes from the same DSP features the recommender uses. Real history and logged sessions are the arbiters. Same-artist pairs can't be measured on FMP (9 of 479 tracks have artist tags).

**After step 6a (profile + skip model, run "profile + skip model"):**

| next track | hit@10 | hit@50 | MRR |
|---|---|---|---|
| your taste (regions) | 57.5% | 79.6% | 0.236 |
| skip model | 55.8% | 76.1% | 0.250 |
| steer, no profile | 42.5% | 52.2% | 0.187 |
| steer | 43.4% | 53.1% | 0.193 |

Skip AUC: skip model 0.81, taste regions 0.76, previous track 0.74 (the single centroid of step 5 scored 0.84 — hence the "overall taste" feature).

| simulated sessions (listeners now have 60 days of past history) | taste | skipped | liked | travel | variety |
|---|---|---|---|---|---|
| steer a=0.2 | 85.5 | 10.8% | 14.6% | 0.63 | 54.2% |
| steer a=0.5 | 85.0 | 11.5% | 14.3% | 0.68 | 56.9% |
| steer a=0.8 | 81.9 | 14.5% | 13.0% | 0.78 | 61.4% |
| steer a=0.5, no profile | 82.3 | 14.6% | 12.4% | 0.68 | 60.4% |

- Knowing you helps: +3 taste points, 3 points fewer skips, more likes; adventurousness is now a working dial (travel 0.63 → 0.78, variety 54% → 61%).
- It costs ~3–4 points of variety at the same adventurousness — the price of pulling toward home.
- On history, steering gains little (+0.9 hit@10): the replay rewards replaying favourites, which steering deliberately doesn't do. "Your taste" and the skip model alone predict far better — they'd make a good "your usual" mode, not a discovery engine.

**After step 6b — modes** (adventurousness 0.5, with profile; destinations are tracks far from the start that the listener feels middling about; "reached" = a pick among the destination's 10 nearest tracks within 25 picks):

| mode | taste | skipped | new to you | worst hop | reached |
|---|---|---|---|---|---|
| usual | 87.9 | 9.2% | 20% | 0.54 | – |
| drift | 84.2 | 12.8% | 28% | 0.50 | – |
| something new | 70.0 | 25.1% | 80% | 0.28 | – |
| surprise | 65.8 | 29.8% | 83% | −0.19 (the hop) | – |
| toward, planned route | 70.2 | 28.9% | 45% | 0.25 | 93% in 8.4 picks |
| toward, straight jump | 69.8 | 29.7% | 44% | −0.27 | 100% in 2 picks |

The planned route gets there in ~8 tracks without a jarring jump, at the same skip rate. Surprise first aimed *away* from the listener's taste (42% skipped); aiming for unexplored places their taste still reaches brought that to 30%.

**Step 7 — how much CLAP.** Simulated listeners' tastes are defined in DSP features by default, which would grade a DSP-only recommender by its own yardstick. `--judge clap` defines them in CLAP space instead; a blend that holds up under both is the robust choice. Steering, a=0.5, with profile:

| CLAP share | taste (DSP judge) | taste (CLAP judge) | skipped (DSP / CLAP judge) | history replay hit@10 / MRR |
|---|---|---|---|---|
| 0 | 84.2 | 82.9 | 13.0% / 17.6% | 43.4% / 0.193 |
| **0.25** | 83.8 | 85.9 | 13.6% / 13.0% | 43.4% / 0.194 |
| 0.5 | 82.7 | 86.7 | 14.3% / 13.1% | 42.5% / 0.207 |
| 0.75 | 81.3 | 88.0 | 16.1% / 11.7% | 39.8% / 0.204 |
| 1 | 79.3 | 88.2 | 16.5% / 11.1% | 32.7% / 0.177 |

0.25 has the best worst case (taste 83.8, skips 13.6%): DSP-like listeners lose almost nothing, CLAP-like listeners skip a quarter less. The history replay is simulated from DSP features too, so it leans the same way. Real listening (logged sessions) should settle it; `--clap-share` makes that a one-flag comparison.

**Step 8 — friend groups** (8 groups of three; A shares one of two home tracks with B, none with C; 60 days of simulated history each; sessions start from A's favourites; adventurousness 0.5, 25% CLAP):

| session | A's taste | skipped | friends' favourites | B's taste | suits both | B's taste at end |
|---|---|---|---|---|---|---|
| solo | 81.2 | 15.1% | 34.9% | 67.7 | 64.0 | 66.2 |
| with friend signal | 80.5 | 14.6% | 51.6% | 73.3 | 67.8 | 70.8 |
| toward friend B | 71.8 | 26.4% | 66.0% | 85.5 | 67.9 | 80.0 |
| solo, B listening too | 80.2 | 37.9% | 38.1% | 70.7 | 65.6 | 68.5 |
| bridge A + B | 77.6 | 32.1% | 44.0% | 79.1 | 74.1 | 77.4 |

("Skipped" with two listeners = either one skips; "suits both" = mean of min(A's, B's percentile).) The friend signal makes half of all picks friends' favourites at no cost to A; "toward B" ends deep in B's taste (80) at the price of more skips on the way, as any journey out of one's taste does; the bridge suits both better (74 vs 66) and cuts joint skips. Measured overlap: A–B 65%, A–C 38% (random tastes aren't unrelated in sound). Eval now takes ~35 s.

Planned: listening tests (export pairs of sessions from variants A and B as blind playlists; record preferences); steering-reach metrics once destinations exist.

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
dhunn session [--seed <song words>]... [--user] [--adventure A] [--mode drift|usual|surprise|new] [--to <song or artist> | --vibe "<description>"] [--len N] [--rng S]  # ✅ print a planned session
dhunn play    (same steering options, plus --friend <name> · --with <name> · --solo · --private; user from --user or $DHUNN_USER)  # ✅ listen: space pause · n next · l like · d dislike · f further · b back · s surprise me · ± volume · q quit
dhunn embed [--models DIR] [--force] [-q]   # ✅ CLAP embeddings for analyzed tracks (resumable)
dhunn vibe "<description>" [-n 10]          # ✅ tracks that best match a description
dhunn friends [--user] [-n 10]              # ✅ taste overlap with friends + what they love that you've never played
dhunn user set <name> [--adventure A] [--share yes|no]   # ✅
dhunn profile [--user]               # ✅ taste regions, dislikes, skip-model weights
dhunn eval [--clap-share S] [--judge dsp|clap] [--user] [--holdout-days 14] [--listeners 16 --sessions 6 --len 25] [--param name=value]... [--label L] [--no-save] [--json]  # ✅ compared with the last run
dhunn index                           # (not needed yet: the space is built at startup) clusters, artist graph
dhunn mb-sync                         # optional MusicBrainz/ListenBrainz matching (rate-limited)
dhunn session --user <name>
             --seed <track>... | --vibe "rainy night drive" | --mood e=0.3,v=-0.2
             --mode drift|surprise|new|dest|friend|bridge --to "jazz" | --friend <name>
             [--adventure 0.7] --len 20 [--interactive]
dhunn discover --user <name>          # unowned suggestions (ListenBrainz + history gaps)
dhunn explain <track>
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
5. ✅ `eval`: next-track replay on history, simulated listeners with feedback, logged-session summary; runs saved and compared — baseline numbers in §11.
6. ✅ (6a) `profile`: taste regions, dislikes, learned skip model, both mixed into steering; `dhunn profile`. ✅ (6b) modes: usual, surprise me, something new, toward a song/artist along a planned route; journey explanations; mode metrics in eval.
7. ✅ `embed`: CLAP (music) exported to ONNX, `dhunn embed`, blended into the sound space at 25%, `--vibe` and `dhunn vibe`; eval under DSP- and CLAP-defined listeners.
8. ✅ Friends: overlap + trust, friend signal in every session, `--friend`, `--with` (bridge), `dhunn friends`, sharing switch, `--private`, `DHUNN_USER`; friend-group eval.
9. `meta` (MusicBrainz + ListenBrainz graph), `discover`, interactive mode + event logging. HNSW only if exact search gets slow.

## 15. Decisions to revisit
- Exact embedding models (license, size, Android feasibility).
- Fixed vs learned block weights in the sound space (learn once we have logs).
- How the pooled library is synced to the shared server and streamed to clients (post-M1).
- Within-group privacy defaults for shared profiles.
