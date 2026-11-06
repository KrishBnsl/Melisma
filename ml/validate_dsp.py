# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["librosa==1.0.*", "numpy", "soundfile"]
# ///
"""Compare dhunn-dsp features with librosa's reference implementations.

Usage (from the repo root, after `cargo build --release`):

    uv run ml/validate_dsp.py <audio> [<audio> ...] [--dhunn target/release/dhunn]

Use 22.05 kHz WAV files for an exact comparison; other rates and lossy formats add
resampler/decoder differences on top of the feature code. Exits non-zero on mismatch.
"""

import argparse
import json
import subprocess
import sys

import librosa
import numpy as np

SR = 22050

# (key prefix, relative tolerance, absolute tolerance). First match wins.
TOLERANCES = [
    ("timbre.mfcc_delta", 0.05, 0.02),
    ("timbre.mfcc", 0.02, 0.3),
    ("harmony.chroma_mean", 0.02, 0.01),
    ("spectral.contrast_db", 0.02, 0.3),
    ("spectral.flatness", 0.05, 1e-4),
    ("percussion.onset", 0.02, 0.01),
    ("rhythm.tempo_bpm", 0.005, 0.0),
    ("percussion.harmonic_ratio", 0.01, 0.005),
    ("dynamics.rms_db", 0.01, 0.1),
    ("validation.yin_raw_f0_median_nonsilent_hz", 0.01, 0.5),
    ("validation.harmonic_rms", 0.01, 1e-6),
    ("", 0.01, 1e-6),
]


def tolerance(key):
    for prefix, rtol, atol in TOLERANCES:
        if key.startswith(prefix):
            return rtol, atol
    raise AssertionError("unreachable")


def reference(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    S = np.abs(librosa.stft(y))  # n_fft=2048, hop=512, centered, zero padding
    ref = {}

    def stat(name, x):
        ref[f"{name}.mean"] = float(np.mean(x))
        ref[f"{name}.std"] = float(np.std(x))

    stat("spectral.centroid_hz", librosa.feature.spectral_centroid(S=S, sr=SR)[0])
    stat("spectral.bandwidth_hz", librosa.feature.spectral_bandwidth(S=S, sr=SR)[0])
    stat("spectral.rolloff_hz", librosa.feature.spectral_rolloff(S=S, sr=SR)[0])
    stat("spectral.flatness", librosa.feature.spectral_flatness(S=S)[0])
    stat("spectral.zcr", librosa.feature.zero_crossing_rate(y)[0])
    contrast = librosa.feature.spectral_contrast(S=S, sr=SR)
    for b in range(contrast.shape[0]):
        stat(f"spectral.contrast_db[{b}]", contrast[b])

    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    for k in range(20):
        stat(f"timbre.mfcc[{k}]", mfcc[k])
        stat(f"timbre.mfcc_delta[{k}]", delta[k])

    chroma = librosa.feature.chroma_stft(S=S**2, sr=SR, tuning=0.0)
    for c in range(12):
        ref[f"harmony.chroma_mean[{c}]"] = float(chroma[c].mean())

    onset = librosa.onset.onset_strength(y=y, sr=SR)
    stat("percussion.onset", onset)
    tempo_fn = getattr(librosa.feature, "tempo", None) or librosa.beat.tempo
    ref["rhythm.tempo_bpm"] = float(np.atleast_1d(tempo_fn(onset_envelope=onset, sr=SR))[0])

    H, P = librosa.decompose.hpss(S)
    eh, ep = float(np.sum(H**2)), float(np.sum(P**2))
    ref["percussion.harmonic_ratio"] = eh / (eh + ep) if eh + ep > 0 else 0.0

    rms = librosa.feature.rms(y=y)[0]
    ref["dynamics.rms_mean"] = float(rms.mean())
    db = 20 * np.log10(np.maximum(rms, 1e-5))
    stat("dynamics.rms_db", db)

    # YIN itself, on the original signal (dhunn runs it on the harmonic part, where
    # unvoiced frames are near-silent noise and their f0 is not meaningful to compare).
    f0 = librosa.yin(y, fmin=65.0, fmax=2093.0, sr=SR)
    n = min(len(f0), len(db))
    nonsilent = db[:n] > db.max() - 40
    ref["validation.yin_raw_f0_median_nonsilent_hz"] = float(np.median(f0[:n][nonsilent]))
    # HPSS masks + inverse STFT, as used to get the harmonic signal for pitch.
    ref["validation.harmonic_rms"] = float(np.sqrt(np.mean(librosa.effects.harmonic(y) ** 2)))
    return ref


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def ours(path, binary):
    cmd = [binary, "inspect", "--json", "--validation", path]
    data = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    flat = flatten(data["summary"])
    flat.update(flatten({"validation": data["diagnostics"].get("validation", {})}))
    return flat


def compare(path, binary, verbose):
    ref = reference(path)
    got = ours(path, binary)
    failures = []
    rows = []
    for key, r in ref.items():
        if key not in got:
            failures.append((key, r, None, "missing"))
            continue
        g = got[key]
        rtol, atol = tolerance(key)
        err = abs(g - r)
        ok = err <= atol or err <= rtol * abs(r)
        rel = err / abs(r) if r != 0 else float("inf") if err else 0.0
        rows.append((key, r, g, rel, ok))
        if not ok:
            failures.append((key, r, g, f"rel {rel:.2%}"))

    print(f"\n{path}")
    worst = sorted(rows, key=lambda x: -x[3] if np.isfinite(x[3]) else 0)
    shown = rows if verbose else worst[:8]
    print(f"  {'feature':<42} {'librosa':>12} {'dhunn':>12} {'rel err':>9}")
    for key, r, g, rel, ok in shown:
        print(f"  {key:<42} {r:>12.5f} {g:>12.5f} {rel:>8.3%} {'' if ok else '  FAIL'}")
    print(f"  {len(rows) - len(failures)}/{len(ref)} within tolerance")
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--dhunn", default="target/release/dhunn")
    ap.add_argument("-v", "--verbose", action="store_true", help="show every feature, not just the worst")
    args = ap.parse_args()
    all_failures = 0
    for path in args.audio:
        failures = compare(path, args.dhunn, args.verbose)
        for key, r, g, why in failures:
            print(f"  FAIL {key}: librosa={r} dhunn={g} ({why})")
        all_failures += len(failures)
    print(f"\n{'OK' if all_failures == 0 else f'{all_failures} mismatches'}")
    sys.exit(1 if all_failures else 0)


if __name__ == "__main__":
    main()
