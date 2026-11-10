# Models

Exported model files live here (git-ignored; they are large and reproducible).

## CLAP (learned audio + text embeddings)

```bash
uv run ml/export_clap.py --out models/clap
```

Downloads LAION's original music CLAP checkpoint (`lukewys/laion_clap`,
`music_audioset_epoch_15_esc_90.14.pt`, 2.35 GB, CC0) and the `laion/larger_clap_music`
config/tokenizer, converts the weights, and writes `audio.onnx` (~286 MB), `text.onnx`
(~502 MB), `tokenizer.json` and `clap.json`. Every export is checked against the reference
implementation. (The Hugging Face weights of `laion/larger_clap_music` are not used: their
projection layers are untrained.)

dhunn looks for the model in `$DHUNN_MODELS/clap`, then `<data dir>/dhunn/models/clap`
(macOS: `~/Library/Application Support/dhunn/models/clap`, Linux: `~/.local/share/dhunn/models/clap`,
Windows: `%APPDATA%\dhunn\models\clap`), then `./models/clap` (only from this folder).
Put an export there so `dhunn embed` and vibe search work from anywhere:

```bash
dhunn model install models/clap --link   # or without --link to copy it
dhunn model                              # where it is
```

`dhunn model download` fetches this same export, ready-made, and checks every file against the
SHA-256 sums in `crates/dhunn-cli/src/model.rs`.
