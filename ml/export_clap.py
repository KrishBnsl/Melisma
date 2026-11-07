# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "huggingface_hub",
#   "torch>=2.6",
#   "transformers>=4.45,<5",
#   "numpy",
#   "onnx",
#   "onnxruntime",
#   "soundfile",
#   "soxr",
# ]
# ///
"""Export LAION CLAP (music) to ONNX for dhunn.

Weights come from LAION's original training checkpoint (lukewys/laion_clap,
music_audioset_epoch_15_esc_90.14.pt, CC0), converted here to transformers' ClapModel. The
Hugging Face conversion `laion/larger_clap_music` is NOT usable: its projection layers are
untrained (zero biases, init-scale weights) and its encoders don't match the original, so
every text scores the same against every track. The architecture/config still comes from
that repo (same model family), and the conversion is checked strictly.

    uv run ml/export_clap.py [--out models/clap] [--check audio1.wav audio2.wav ...]

Writes to the output folder:
  audio.onnx      raw mono 48 kHz audio [batch, 480000] -> unit embedding [batch, 512].
                  The log-mel front end (window, STFT, Slaney mel filters, dB) is baked into
                  the graph using the feature extractor's own arrays, so Rust only has to
                  decode, resample to 48 kHz and cut/repeat-pad 10 s clips.
  text.onnx       input_ids, attention_mask [batch, tokens] (int64) -> unit embedding [batch, 512].
  tokenizer.json  the RoBERTa tokenizer (for the Rust `tokenizers` crate).
  clap.json       manifest: model id, dims, sample rate, clip length, version.

Every export is checked against the transformers pipeline (cosine >= 0.9999 on the same
clips); --check also prints zero-shot text matches for the given files as a sanity check.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import ClapModel, ClapProcessor
from transformers.audio_utils import window_function

MODEL = "laion/larger_clap_music"  # config, tokenizer and feature extractor
ORIGINAL = ("lukewys/laion_clap", "music_audioset_epoch_15_esc_90.14.pt")  # weights
VERSION = 1  # bump when the export changes in a way that changes embeddings
SR = 48_000
CLIP = 480_000  # 10 s


RENAMES = {
    "text_branch": "text_model",
    "audio_branch": "audio_model.audio_encoder",
    "attn": "attention.self",
    "self.proj": "output.dense",
    "attention.self_mask": "attn_mask",
    "mlp.fc1": "intermediate.dense",
    "mlp.fc2": "output.dense",
    "norm1": "layernorm_before",
    "norm2": "layernorm_after",
    "bn0": "batch_norm",
}
# Parts of the original that transformers does elsewhere (front end: feature extractor) or
# not at all (training-only MLP heads).
DROPPED = ("spectrogram_extractor", "logmel_extractor", "text_transform", "audio_transform")


def load_original():
    """LAION's checkpoint as a transformers state dict, plus its own mel filters."""
    import re
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(*ORIGINAL)
    # Safe loading: besides tensors the file holds an epoch number and a few numpy
    # scalars (training metrics); allow exactly those types, nothing executable.
    safe = [
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        type(np.dtype("float64")),
        type(np.dtype("int64")),
    ]
    with torch.serialization.safe_globals(safe):
        ck = torch.load(path, map_location="cpu", weights_only=True)
    src = {k.removeprefix("module."): v for k, v in ck["state_dict"].items()}
    mel = src.get("audio_branch.logmel_extractor.melW")
    out = {}
    for key, value in src.items():
        if any(d in key for d in DROPPED):
            continue
        for a, b in RENAMES.items():
            if a in key:
                key = key.replace(a, b)
        m = re.match(r".*_projection\.(\d+)\..*", key)
        if m:
            i = int(m.group(1))
            key = key.replace(f"_projection.{i}.", f"_projection.linear{1 if i == 0 else 2}.")
        if "audio" in key and "qkv" in key:
            d = value.size(0) // 3
            for j, part in enumerate(["query", "key", "value"]):
                out[key.replace("qkv", part)] = value[j * d : (j + 1) * d]
        else:
            out[key] = value
    return out, mel


class AudioEmbed(torch.nn.Module):
    """Waveform -> CLAP audio embedding, front end included."""

    def __init__(self, model: ClapModel, fe):
        super().__init__()
        self.model = model
        n_fft, hop = fe.fft_window_size, fe.hop_length
        self.hop = hop
        self.pad = n_fft // 2
        win = torch.tensor(window_function(n_fft, "hann"), dtype=torch.float64)
        k = torch.arange(n_fft, dtype=torch.float64)
        f = torch.arange(n_fft // 2 + 1, dtype=torch.float64)
        ang = 2 * math.pi * torch.outer(f, k) / n_fft  # [513, 1024]
        # STFT as one strided convolution: real and imaginary DFT rows, windowed.
        kernel = torch.cat([torch.cos(ang) * win, -torch.sin(ang) * win])  # [1026, 1024]
        self.register_buffer("kernel", kernel.float().unsqueeze(1))
        self.register_buffer("mel", torch.tensor(fe.mel_filters_slaney, dtype=torch.float32))  # [513, 64]
        self.bins = n_fft // 2 + 1

    def features(self, wav: torch.Tensor) -> torch.Tensor:
        x = F.pad(wav.unsqueeze(1), (self.pad, self.pad), mode="reflect")  # [B, 1, T+pad]
        spec = F.conv1d(x, self.kernel, stride=self.hop)  # [B, 1026, frames]
        re, im = spec[:, : self.bins], spec[:, self.bins :]
        power = (re * re + im * im).transpose(1, 2)  # [B, frames, 513]
        mel = torch.clamp(power @ self.mel, min=1e-10)
        return 10.0 * torch.log10(mel)  # [B, frames, 64]

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        feats = self.features(wav).unsqueeze(1)  # [B, 1, frames, 64]
        longer = torch.zeros(wav.shape[0], 1, dtype=torch.bool)
        e = self.model.get_audio_features(input_features=feats, is_longer=longer)
        return F.normalize(e, dim=-1)


class TextEmbed(torch.nn.Module):
    def __init__(self, model: ClapModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        e = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(e, dim=-1)


def load_clip(path: str) -> np.ndarray:
    """First 10 s of a file at 48 kHz mono, repeat-padded like the feature extractor."""
    import soundfile as sf
    import soxr

    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        y = soxr.resample(y, sr, SR).astype(np.float32)
    y = y[:CLIP]
    if len(y) < CLIP:
        y = np.tile(y, CLIP // len(y))
        y = np.pad(y, (0, CLIP - len(y)))
    return y


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return (a * b).sum(-1)


PROMPTS = [
    "solo piano",
    "orchestral classical music",
    "heavy metal with distorted guitars",
    "acoustic folk song with vocals",
    "electronic dance music",
    "a musical scale played slowly",
    "applause and crowd noise",
    "speech, someone talking",
    "calm, sad and slow",
    "happy and energetic",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/clap")
    ap.add_argument("--check", nargs="*", default=[])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    proc = ClapProcessor.from_pretrained(MODEL)
    model = ClapModel.from_pretrained(MODEL).eval()
    fe = proc.feature_extractor
    state, orig_mel = load_original()
    res = model.load_state_dict(state, strict=False)
    # Allowed differences: buffers transformers builds itself (position/token-type ids,
    # Swin attention masks) and HTSAT's AudioSet classification heads, which embeddings
    # don't use. Anything else means the conversion lost or misnamed weights.
    missing = [k for k in res.missing_keys if not k.endswith(("position_ids", "token_type_ids"))]
    unused = [
        k for k in res.unexpected_keys
        if not k.endswith("attn_mask") and ".tscam_conv." not in k and ".audio_encoder.head." not in k
    ]
    print(f"original weights: {len(state)} tensors loaded; missing {missing}; unused {unused}")
    assert not missing and not unused, "conversion incomplete"
    # The front end must be the one the model was trained with.
    if orig_mel is not None:
        dm = np.abs(orig_mel.numpy() - fe.mel_filters_slaney).max()
        print(f"mel filters vs original: max |Δ| = {dm:.2e}")
        assert dm < 1e-5, "mel filters differ from the original model's"
    print(f"logit scales: audio {model.logit_scale_a.exp().item():.1f}, text {model.logit_scale_t.exp().item():.1f}")
    audio, text = AudioEmbed(model, fe).eval(), TextEmbed(model).eval()

    # 1. The baked-in front end must match the feature extractor.
    rng = np.random.default_rng(0)
    clips = [rng.standard_normal(CLIP).astype(np.float32) * 0.1]
    clips += [load_clip(p) for p in args.check[:4]]
    batch = np.stack(clips)
    ref = fe(list(batch), sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        mine = audio.features(torch.from_numpy(batch))
        theirs = ref["input_features"][:, 0]
        err = (mine - theirs).abs().max().item()
        print(f"front end: max |Δ| = {err:.2e} dB over {theirs.shape}")
        assert mine.shape == theirs.shape and err < 1e-2, "log-mel front end differs"
        want = F.normalize(model.get_audio_features(**ref), dim=-1).numpy()
        got = audio(torch.from_numpy(batch)).numpy()
    print(f"audio (torch): min cosine vs transformers = {cosine(want, got).min():.6f}")

    # 2. Export.
    with torch.no_grad():
        torch.onnx.export(
            audio,
            (torch.from_numpy(batch[:2]),),
            out / "audio.onnx",
            input_names=["waveform"],
            output_names=["embedding"],
            dynamic_axes={"waveform": {0: "batch"}, "embedding": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        tok = proc.tokenizer(PROMPTS[:2], padding=True, return_tensors="pt")
        torch.onnx.export(
            text,
            (tok["input_ids"], tok["attention_mask"]),
            out / "text.onnx",
            input_names=["input_ids", "attention_mask"],
            output_names=["embedding"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "tokens"},
                "attention_mask": {0: "batch", 1: "tokens"},
                "embedding": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
    proc.tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))

    # 3. ONNX Runtime must agree with transformers.
    import onnxruntime as ort

    sa = ort.InferenceSession(str(out / "audio.onnx"))
    got = sa.run(None, {"waveform": batch})[0]
    ca = cosine(want, got).min()
    print(f"audio (onnx): min cosine vs transformers = {ca:.6f}")
    assert ca > 0.9999, "audio export differs"

    st = ort.InferenceSession(str(out / "text.onnx"))
    tok = proc.tokenizer(PROMPTS, padding=True, return_tensors="np")
    tg = st.run(None, {"input_ids": tok["input_ids"].astype(np.int64),
                       "attention_mask": tok["attention_mask"].astype(np.int64)})[0]
    with torch.no_grad():
        tt = proc.tokenizer(PROMPTS, padding=True, return_tensors="pt")
        tw = F.normalize(model.get_text_features(**tt), dim=-1).numpy()
    ct = cosine(tw, tg).min()
    print(f"text (onnx): min cosine vs transformers = {ct:.6f}")
    assert ct > 0.9999, "text export differs"

    # Reference values for the Rust tests: tokens and embedding of one prompt, and the
    # embedding of 10 s of a 440 Hz sine at half amplitude (easy to regenerate exactly).
    ref_tok = proc.tokenizer(["solo piano"], return_tensors="np")
    t = np.arange(CLIP, dtype=np.float64) / SR
    sine = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)[None]
    sine_emb = sa.run(None, {"waveform": sine})[0][0]
    manifest = {
        "model": f"LAION CLAP music ({ORIGINAL[1]}, converted)",
        "version": VERSION,
        "dims": int(got.shape[1]),
        "sample_rate": SR,
        "clip_samples": CLIP,
        "max_tokens": 77,
        "check": {
            "text": "solo piano",
            "input_ids": ref_tok["input_ids"][0].tolist(),
            "embedding_head": tg[0][:8].tolist(),
            "sine_440_head": sine_emb[:8].tolist(),
        },
    }
    (out / "clap.json").write_text(json.dumps(manifest, indent=1))
    for f in ["audio.onnx", "text.onnx", "tokenizer.json", "clap.json"]:
        print(f"  {out / f}: {(out / f).stat().st_size / 1e6:.1f} MB")

    # 4. Sanity: what does CLAP hear in the given files?
    if args.check:
        files = args.check
        emb = sa.run(None, {"waveform": np.stack([load_clip(p) for p in files])})[0]
        sims = emb @ tg.T
        for f, row in zip(files, sims):
            top = np.argsort(-row)[:3]
            print(f"{Path(f).name[:60]:<60} " + " · ".join(f"{PROMPTS[i]} {row[i]:.2f}" for i in top))


if __name__ == "__main__":
    main()
