# dhunn — Recommendation Engine Requirements

Status: draft v2 · 2026-11-06 · v2: re-scoped from public streaming service to a private app for a small group of friends.

## 1. Product thesis

dhunn is a private music app for **me and a small group of friends** (≈2–20 people), built because Spotify/Apple Music recommendations got stale. It is not trying to beat them at generic "predict what you'll like" at their scale. It wins for *this group* on:

- **New, exciting, still relevant** — calibrated novelty: every pick is new-ish but close enough to taste to land. Each person controls how adventurous it is.
- **Friends as the discovery source** — "people like you" means people you actually trust, not millions of strangers. Steer toward a friend's taste, gradually.
- **Steerable** — the listener directs the journey ("go further", "go back", "take me toward jazz").
- **Explainable** — every pick says *why* and *where we're heading*.
- **Continuous** — sessions are planned trajectories with a speed limit, not independent picks; genre drift feels smooth and steady.
- **Fair to the long tail** — audio-first scoring gives a zero-play track an equal shot (no popularity bias).
- **Knows you from day one** — imports existing Spotify/Apple Music listening history.

## 2. Users & deployment

- **Users:** small, invite-only group (≈2–20). Not a public service. No catalog business, no public rehosting of commercial music.
- **Shared server:** one self-hosted box (home server or cheap VPS) holding the group's **pooled library** (≈10k–500k tracks) — Navidrome/Jellyfin-style, with a smarter engine.
- **Clients:** Android + desktop, each with the Rust core on-device.

| Mode | Library | Analysis | Recommendations |
|---|---|---|---|
| **Online (group server)** | Pooled group library on the shared server | Server-side, batch, on add | Whole pooled library + friend signals + external discovery |
| **Local / offline** | User's own files on device | On-device, Rust, background, once per track | Own library; optional "Discover" section (from group library / external suggestions) when the library is thin |

## 3. Signals the engine uses

### 3.1 Audio (core)
- **Low-level DSP:** spectrogram / mel-spectrogram, MFCC, chromagram (key/harmony), spectral centroid/rolloff/contrast (brightness/treble vs. bass), harmonic/percussive separation, onset strength & low-end energy ("thump"), loudness/dynamics, pitch.
- **Rhythm:** tempo, beat strength, beat regularity, danceability-like measures.
- **Learned embeddings:** general audio-similarity embedding (e.g. MERT-class) and joint text–audio embedding (CLAP-class) for mood and free-text vibe search.
- **Derived:** mood (valence/arousal), energy, timbre descriptors — each human-readable for explanations.

### 3.2 Non-audio (cheap context)
- Artist relationships (MusicBrainz: collaborations, members, influences, labels).
- Similar artists / recordings from **ListenBrainz** open data.
- Lyrics language and, where available, lyric themes.
- Release year / era.
- Genre/tag metadata where trustworthy (soft signal, never sole driver).

### 3.3 Behavioral
- **Imported history:** Spotify extended streaming history and Apple Music play activity (GDPR data exports); optionally Last.fm / ListenBrainz scrobbles.
- **Implicit:** skips (and skip time), full listens, replays, volume changes, time of day.
- **Explicit:** like/dislike, "never play", "more like this".
- **Steering feedback:** "too far", "go further", "go back" — adjusts exploration speed.
- **Friends:** co-listening and taste overlap within the group (small-group collaborative signals).

## 4. Friends features
- **Toward a friend:** steer gradually from my taste into a friend's taste region.
- **Bridges:** tracks that sit where two people's tastes overlap — good for shared/group sessions.
- **Friend-sourced novelty:** things a friend loves that I've never heard, ranked by how close they are to my taste.
- **Privacy within the group:** each person opts in to sharing their taste profile/history with the group; "private listening" sessions are never shared.

## 5. Where new music comes from
A pooled library alone recycles itself, so:
1. **Group library growth** — anything a friend adds is immediately discovery material for everyone.
2. **External suggestions (ListenBrainz/MusicBrainz)** — artists/recordings nobody in the group owns, surfaced as "Discover" items with link-outs. They have no audio, so they are placed via the artist graph and ranked by proximity to owned music.
3. **Later: legally downloadable audio** (Bandcamp purchases, Free Music Archive, Jamendo) — analyzed like any other track.
4. **Imported history gaps** — tracks you loved on Spotify/Apple that aren't in the library become "add this" suggestions.

## 6. Data flywheel (from day one)
- Log every session event: track played, position, skip/complete, steering actions, mode, vibe input, context, user. Versioned schema; local-mode logs stay on device unless the user opts in to syncing.
- **Blending strategy:** audio + metadata + imported history carry cold start; behavioral and friend-based weight grows with data per user and per track. Blend weights are explicit and tunable.

## 7. Session memory
- Model the current session as a trajectory, not just a static profile: recent tracks, recent reactions, current direction and velocity in the feature space.
- Long-term profile (stable taste) + short-term session state (current context) are combined at scoring time.
- Avoid repetition within and across recent sessions.

## 8. Exploration / steering
- **Adventurousness (per user):** a single control that sets default exploration speed and novelty weight.
- **Slow drift (default):** gradual movement toward adjacent-but-new regions, bounded step size per track.
- **Surprise me:** larger, still-coherent jump, then settle.
- **Something new:** bias toward unheard artists/genres near the user's frontier.
- **Today's vibe:** target region set by free-text prompt, mood/energy controls, seed tracks, and/or context (time, activity).
- **Destination journeys:** path from current region to a target (genre, artist, mood, or a friend's taste), via sonically and artist-adjacent stepping stones.
- Exploration speed adapts to steering feedback and skip behavior.

## 9. Explainability (user-facing)
- Each recommendation carries a short reason built from interpretable features and graph links, e.g. *"similar warm bass and slow tempo — moving you toward neo-soul"* or *"Aarav has this on repeat; close to your late-night stuff."*
- Session view can show the journey direction.

## 10. Cold start
- **Import Spotify/Apple history** (primary path — taste known on first run).
- Local library scan builds an initial profile automatically.
- Pick 3–5 favorite tracks/artists.
- Vibe quiz: quick A/B audio clips to place the user in the sound map.
- Fallback: diverse picks from the group library, learn quickly from skips.

## 11. Evaluation (before any tuning)
- **Real data from day one:** imported Spotify extended history includes skips, play duration and end reason — usable as held-out sessions for offline evaluation.
- **Offline metrics:** skip prediction, next-track ranking (hit rate / NDCG on held-out sessions), transition smoothness (feature distance between consecutive tracks), novelty (unheard artists/genres share), diversity, long-tail coverage, repetition rate.
- **Steering metrics:** does the session reach the target region; is per-step distance within bounds.
- **Listening tests:** blind A/B comparisons of generated sessions among the group.
- Reproducible evaluation harness runnable from the CLI; every algorithm change reports these numbers.

## 12. Technical requirements
- **Language:** Rust for engine core (analysis, indexing, scoring, steering, serving).
- **ML:** train/fine-tune in Python (PyTorch), export to ONNX, inference in Rust.
- **Platforms (on-device core):** Android, desktop (macOS/Windows/Linux).
- **Scale:** ≈10k–500k tracks in the pooled library → exact search is the default; approximate nearest-neighbor (HNSW) optional at the top end.
- **Hosting:** one self-hosted machine (home server or VPS); everything self-hosted, no managed cloud services required.
- **Performance targets (initial):** next-track recommendation < 50 ms on device; local analysis runs in the background without hurting playback.

## 13. Milestone 1 — Engine + CLI demo
1. Analyze a folder of audio files → feature store (DSP features + embeddings).
2. Import Spotify/Apple listening history for one or more users and match it to library tracks.
3. Build index (vectors + artist/metadata graph from MusicBrainz where matchable).
4. Generate steered sessions from the CLI per user: seed tracks, vibe text, mood controls, adventurousness, mode (drift / surprise / new / destination / toward-friend).
5. Explanations printed per track.
6. Session event logging (interactive feedback from the CLI).
7. Evaluation harness producing the metrics in §11, using imported history as real data.

## 14. Open questions
- Which pretrained models to adopt first (license, size, on-device feasibility on Android).
- How the pooled library gets synced/uploaded to the shared server, and how files are streamed to clients.
- Within-group privacy defaults (what friends can see of each other's listening).
- Lyrics data source and licensing.
