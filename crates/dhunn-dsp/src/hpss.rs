//! Harmonic/percussive source separation by median filtering (`librosa.decompose.hpss`,
//! kernel 31, soft masks with power 2, margin 1).

use std::ops::Range;

use rayon::prelude::*;

use crate::stft::Matrix;

/// Index into `0..n` with scipy's `mode="reflect"` (`d c b a | a b c d | d c b a`).
fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    let period = 2 * n;
    let m = i.rem_euclid(period);
    (if m >= n { period - 1 - m } else { m }) as usize
}

/// Sliding median with an odd kernel and reflect boundaries; keeps a sorted window so each
/// step is O(kernel) instead of a full sort.
pub fn median_filter_reflect(x: &[f32], kernel: usize, out: &mut [f32]) {
    debug_assert!(kernel % 2 == 1);
    let n = x.len();
    if n == 0 {
        return;
    }
    let half = (kernel / 2) as isize;
    let ext = |i: isize| x[reflect(i, n)];
    let mut win: Vec<f32> = (-half..=half).map(ext).collect();
    win.sort_by(f32::total_cmp);
    out[0] = win[half as usize];
    for (t, o) in out.iter_mut().enumerate().take(n).skip(1) {
        let old = ext(t as isize - 1 - half);
        let pos = win.partition_point(|v| v.total_cmp(&old).is_lt());
        win.remove(pos);
        let new = ext(t as isize + half);
        let pos = win.partition_point(|v| v.total_cmp(&new).is_lt());
        win.insert(pos, new);
        *o = win[half as usize];
    }
}

/// librosa's `softmask(x, x_ref, power=2)`: x² / (x² + x_ref²), 0 where both vanish.
fn softmask(x: f32, x_ref: f32) -> f32 {
    let z = x.max(x_ref);
    if z < f32::MIN_POSITIVE {
        return 0.0;
    }
    let m = (x / z).powi(2);
    let r = (x_ref / z).powi(2);
    m / (m + r)
}

pub struct Hpss {
    /// Soft mask for the harmonic part (`frames × bins`), for re-synthesis.
    pub harmonic_mask: Matrix,
    /// Σ |H|² over all cells.
    pub harmonic_energy: f64,
    /// Σ |P|² over all cells.
    pub percussive_energy: f64,
    /// Per-frame Σ |P|² over `low_bins` (bass-drum region).
    pub low_percussive_energy: Vec<f64>,
}

pub fn hpss(mag: &Matrix, kernel: usize, low_bins: Range<usize>) -> Hpss {
    let (nf, nb) = (mag.rows(), mag.cols());

    // Harmonic: median along time, per frequency bin (contiguous after transposing).
    let mag_t = mag.transpose();
    let mut harm_t = Matrix::zeros(nb, nf);
    harm_t
        .data_mut()
        .par_chunks_mut(nf)
        .zip(mag_t.data().par_chunks(nf))
        .for_each(|(out, row)| median_filter_reflect(row, kernel, out));
    drop(mag_t);

    // Percussive: median along frequency, per frame.
    let mut perc = Matrix::zeros(nf, nb);
    perc.data_mut()
        .par_chunks_mut(nb)
        .zip(mag.data().par_chunks(nb))
        .for_each(|(out, row)| median_filter_reflect(row, kernel, out));

    let mut harmonic_mask = Matrix::zeros(nf, nb);
    let per_frame: Vec<(f64, f64, f64)> = harmonic_mask
        .data_mut()
        .par_chunks_mut(nb)
        .enumerate()
        .map(|(t, mrow)| {
            let (s, p_med) = (mag.row(t), perc.row(t));
            let (mut eh, mut ep, mut low) = (0f64, 0f64, 0f64);
            for f in 0..nb {
                let h_med = harm_t.data()[f * nf + t];
                let mh = softmask(h_med, p_med[f]);
                let mp = softmask(p_med[f], h_med);
                mrow[f] = mh;
                let h = (s[f] * mh) as f64;
                let p = (s[f] * mp) as f64;
                eh += h * h;
                ep += p * p;
                if low_bins.contains(&f) {
                    low += p * p;
                }
            }
            (eh, ep, low)
        })
        .collect();

    Hpss {
        harmonic_mask,
        harmonic_energy: per_frame.iter().map(|x| x.0).sum(),
        percussive_energy: per_frame.iter().map(|x| x.1).sum(),
        low_percussive_energy: per_frame.iter().map(|x| x.2).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_indices() {
        let n = 4;
        let got: Vec<usize> = (-4..8).map(|i| reflect(i, n)).collect();
        assert_eq!(got, vec![3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0]);
    }

    #[test]
    fn median_filter_matches_naive() {
        let x: Vec<f32> = (0..50)
            .map(|i| ((i * 37) % 11) as f32 - (i % 3) as f32)
            .collect();
        let k = 7;
        let mut fast = vec![0f32; x.len()];
        median_filter_reflect(&x, k, &mut fast);
        for (t, &got) in fast.iter().enumerate() {
            let mut w: Vec<f32> = (t as isize - 3..=t as isize + 3)
                .map(|i| x[reflect(i, x.len())])
                .collect();
            w.sort_by(f32::total_cmp);
            assert_eq!(got, w[3], "t={t}");
        }
    }

    #[test]
    fn steady_tone_is_harmonic_and_click_is_percussive() {
        let (nf, nb) = (100, 64);
        // Horizontal line (steady tone) at bin 10.
        let mut tone = Matrix::zeros(nf, nb);
        for t in 0..nf {
            tone.row_mut(t)[10] = 1.0;
        }
        let h = hpss(&tone, 31, 0..0);
        assert!(h.harmonic_energy > 0.95 * (h.harmonic_energy + h.percussive_energy));

        // Vertical line (click) at frame 50.
        let mut click = Matrix::zeros(nf, nb);
        click.row_mut(50).fill(1.0);
        let h = hpss(&click, 31, 0..5);
        assert!(h.percussive_energy > 0.95 * (h.harmonic_energy + h.percussive_energy));
        assert!(h.low_percussive_energy[50] > 4.0);
    }
}
