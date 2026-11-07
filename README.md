# Melisma

A music recommendation engine for you and a small group of friends, written in Rust.

Most recommenders work out what you'll like from what millions of strangers play. Melisma
works from **how the music sounds**: timbre, brightness, harmony, rhythm, low-end punch,
dynamics and pitch, plus a learned audio–text embedding (CLAP). It doesn't just play more of
the same. It **steers**: sessions drift gradually into new sound, or travel step by step
toward a song, an artist, a described vibe or a friend's taste. Every pick says why it was
chosen.

> **Status:** Milestone 1 (the engine plus a command-line player) is mostly done. There's no
> app UI or server yet. The engine and CLI are still named `dhunn`.

## What it does

- **Listens to your library.** It analyzes each track: a mel spectrogram, MFCCs, chroma and
  key, harmonic/percussive balance, onsets and tempo, pitch (YIN), and more. The features are
  validated against librosa. It also embeds each track with a music-trained CLAP model run
  through ONNX Runtime.
- **Learns your taste** from an imported Spotify or Apple Music history and from what you
  finish, skip, like and dislike while listening. Each person gets taste regions, a learned
  skip model, and a memory of tracks they never want to hear again.
- **Steers sessions:**
  - `drift`: gradual, continuous movement into new sound;
  - `usual`: stay close to what you play most;
  - `surprise`: one jump to a part of your library you haven't explored, then it settles there;
  - `new`: travel to music you've never played;
  - `--to "<song or artist>"`: a planned route of small steps;
  - `--vibe "rainy night drive"`: travel toward a described sound.
- **Reacts as you listen.** Skips steer away; *further* and *back* change how far it wanders.
- **Knows your friends** (if they opt in). Picks lean toward what friends love, with credit:
  "Aarav plays this a lot". `--friend <name>` travels into the part of a friend's taste you
  don't share yet. `--with <name>` is for listening together and plays what suits everyone.
- **Measures itself.** `dhunn eval` replays your history, runs simulated listeners and friend
  groups, and summarizes your real sessions. Each run is saved and compared with the last.

## Quick start

Requirements: Rust 1.92 or newer, and [uv](https://docs.astral.sh/uv/) for the one-time model
export.

```bash
cargo build --release
alias dhunn=./target/release/dhunn

# 1. Analyze a folder of audio files (mp3, flac, wav, m4a/aac, ogg, aiff). Resumable.
dhunn analyze ~/Music/library

# 2. Optional but recommended: learned embeddings (≈ 2 tracks/s; needs the model, see below).
uv run ml/export_clap.py --out models/clap
dhunn embed

# 3. Tell it who you are, and import your listening history.
dhunn user add krish --adventure 0.6
dhunn import spotify ~/Downloads/my_spotify_data --user krish
export DHUNN_USER=krish

# 4. Listen.
dhunn play
dhunn play --vibe "warm, slow and melancholic"
dhunn play --to "radiohead"
```

While playing: `space` pause · `n` next · `l` like · `d` dislike · `f` further · `b` back ·
`s` surprise me · `+`/`-` volume · `q` quit.

No music files handy? A copy of the
[FMP notebooks](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C0/C0.html) audio and
`dhunn dev sample-history --out <dir>` (a simulated listener) are enough to try everything.

## Commands

| Command | What it does |
|---|---|
| `analyze <dir>` | Scan and analyze new or changed tracks (by audio content, so renames and duplicates are free) |
| `embed` | CLAP embeddings for analyzed tracks |
| `import spotify\|apple <path> --user <name>` | Import a streaming history export and match it to your library |
| `play` / `session` | Listen to, or print, a steered session (`--mode`, `--to`, `--vibe`, `--friend`, `--with`, `--adventure`, `--solo`, `--private`) |
| `similar "<song>"` | The closest-sounding tracks, and which aspects they share |
| `vibe "<description>"` | The tracks that best match a description |
| `profile` | Your taste regions and what the skip model has learned |
| `friends` | Taste overlap with friends, and what they love that you've never played |
| `user add\|list\|set` | People, adventurousness, and whether their taste is shared (`--share yes\|no`) |
| `eval` | Recommendation quality, compared with the previous run (`--param`, `--clap-share`, `--judge`) |
| `stats`, `history`, `inspect` | Library totals, imported history, one file's features |

## How it works

```
audio ─► decode + resample ─► DSP features (134) ─┐
                         └──► CLAP embedding (512) ┴─► sound space ─► places (library map)
history + sessions ─► profile (taste regions, skip model) ─┐            │
friends' profiles ─► friend signal ────────────────────────┴─► steering session ─► pick + reason
```

- **Sound space:** z-scored DSP features grouped into seven perceptual aspects with equal
  footing, blended 75 / 25 with centred CLAP embeddings. Similarity is a dot product.
- **Steering:** each pick balances closeness to where the session is going, continuity with
  the last track, moving away from what was just heard, your long-term taste, the skip model
  and the friend signal. The taste pull fades as exploration rises, so adventurousness really
  changes where sessions go. Journeys follow routes over a graph of "places" (track clusters)
  that prefer many small steps over one big jump.
- **Everything is local:** SQLite storage, ONNX Runtime on the CPU, no network calls at runtime.

[ARCHITECTURE.md](ARCHITECTURE.md) has the full design, the evaluation results behind every
default, and the build order. [REQUIREMENTS.md](REQUIREMENTS.md) has the product goals.

## Repository layout

```
crates/
  dhunn-core      shared types          dhunn-index    sound space, library map
  dhunn-audio     decode + resample     dhunn-profile  taste, skip model, friends
  dhunn-dsp       audio features        dhunn-steer    sessions, modes, journeys
  dhunn-embed     CLAP via ONNX         dhunn-eval     simulation + evaluation
  dhunn-store     SQLite                dhunn-import   Spotify / Apple history
  dhunn-meta      tags                  dhunn-analysis library scanning
  dhunn-cli       the `dhunn` command
ml/               Python: CLAP export, DSP validation against librosa
models/           exported models (git-ignored; see models/README.md)
```

```bash
cargo test --workspace     # ~115 tests; the CLAP test is skipped if no model is exported
```

## Music sources

Melisma plays and analyzes audio files you have. Use music you have the right to: DRM-free
purchases (Bandcamp, Qobuz, 7digital, iTunes), CDs you've ripped, or freely licensed catalogues
(Free Music Archive, Jamendo). Sharing a pooled library with friends counts as redistribution,
so keep that in mind for anything you didn't buy.

## Credits

- CLAP model: [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP), music checkpoint
  `music_audioset_epoch_15_esc_90.14.pt` (CC0).
- DSP features follow [librosa](https://librosa.org) and are validated against it.
- Test audio: [FMP notebooks](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C0/C0.html)
  (Meinard Müller).

No licence has been chosen yet, so all rights are reserved for now.
