//! Short-time Fourier transform (librosa-compatible framing) and a small dense matrix type.

use std::sync::Arc;

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex, num_complex::Complex32};

/// Periodic Hann window (`scipy.signal.get_window("hann", n, fftbins=True)`).
pub fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos()) as f32)
        .collect()
}

/// Dense row-major `f32` matrix. Spectrogram-like data is stored frame-major
/// (one row per frame).
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "matrix data length mismatch");
        Self { rows, cols, data }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        &mut self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|&v| f(v)).collect(),
        }
    }

    pub fn transpose(&self) -> Matrix {
        let mut out = Matrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        out
    }

    /// Mean of each column (e.g. the average spectrum over all frames).
    pub fn column_means(&self) -> Vec<f32> {
        let mut acc = vec![0f64; self.cols];
        for r in 0..self.rows {
            for (a, &v) in acc.iter_mut().zip(self.row(r)) {
                *a += v as f64;
            }
        }
        let n = self.rows.max(1) as f64;
        acc.into_iter().map(|a| (a / n) as f32).collect()
    }
}

/// STFT with `center=True` and zero ("constant") padding, matching `librosa.stft` defaults.
pub struct Stft {
    n_fft: usize,
    hop: usize,
    window: Vec<f32>,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        Self {
            n_fft,
            hop,
            window: hann_periodic(n_fft),
            fwd: planner.plan_fft_forward(n_fft),
            inv: planner.plan_fft_inverse(n_fft),
        }
    }

    pub fn n_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    pub fn n_frames(&self, len: usize) -> usize {
        1 + len / self.hop
    }

    fn padded(&self, y: &[f32]) -> Vec<f32> {
        let pad = self.n_fft / 2;
        let mut p = vec![0.0; y.len() + 2 * pad];
        p[pad..pad + y.len()].copy_from_slice(y);
        p
    }

    /// Calls `f(frame_index, spectrum)` for every frame.
    pub fn for_each_frame(&self, y: &[f32], mut f: impl FnMut(usize, &mut [Complex32])) {
        let p = self.padded(y);
        let mut input = self.fwd.make_input_vec();
        let mut spec = self.fwd.make_output_vec();
        let mut scratch = self.fwd.make_scratch_vec();
        for t in 0..self.n_frames(y.len()) {
            let start = t * self.hop;
            for (i, v) in input.iter_mut().enumerate() {
                *v = p[start + i] * self.window[i];
            }
            self.fwd
                .process_with_scratch(&mut input, &mut spec, &mut scratch)
                .expect("buffer sizes come from the planner");
            f(t, &mut spec);
        }
    }

    /// Magnitude spectrogram, `frames × bins`.
    pub fn magnitude(&self, y: &[f32]) -> Matrix {
        let mut out = Matrix::zeros(self.n_frames(y.len()), self.n_bins());
        self.for_each_frame(y, |t, spec| {
            for (o, c) in out.row_mut(t).iter_mut().zip(spec.iter()) {
                *o = c.norm();
            }
        });
        out
    }

    /// Re-synthesizes `y` after multiplying each STFT cell by `mask` (`frames × bins`),
    /// using windowed overlap-add like `librosa.istft`. Output has the same length as `y`.
    pub fn istft_masked(&self, y: &[f32], mask: &Matrix) -> Vec<f32> {
        let n_frames = self.n_frames(y.len());
        assert_eq!(
            (mask.rows(), mask.cols()),
            (n_frames, self.n_bins()),
            "mask shape"
        );
        let total = self.n_fft + self.hop * (n_frames - 1);
        let mut out = vec![0f32; total];
        let mut wss = vec![0f32; total];
        let mut frame = self.inv.make_output_vec();
        let mut scratch = self.inv.make_scratch_vec();
        let scale = 1.0 / self.n_fft as f32;
        let last = self.n_bins() - 1;
        self.for_each_frame(y, |t, spec| {
            for (c, &m) in spec.iter_mut().zip(mask.row(t)) {
                *c *= m;
            }
            // A real signal's DC and Nyquist bins are real; drop rounding noise.
            spec[0].im = 0.0;
            spec[last].im = 0.0;
            self.inv
                .process_with_scratch(spec, &mut frame, &mut scratch)
                .expect("DC/Nyquist imaginary parts are zeroed");
            let start = t * self.hop;
            for i in 0..self.n_fft {
                let w = self.window[i];
                out[start + i] += frame[i] * scale * w;
                wss[start + i] += w * w;
            }
        });
        for (o, &w) in out.iter_mut().zip(&wss) {
            if w > f32::MIN_POSITIVE {
                *o /= w;
            }
        }
        let pad = self.n_fft / 2;
        let mut y_out = out[pad..].to_vec();
        y_out.resize(y.len(), 0.0);
        y_out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_matches_librosa() {
        let s = Stft::new(2048, 512);
        assert_eq!(s.n_frames(22050), 44); // 1 + 22050 // 512
        assert_eq!(s.n_bins(), 1025);
    }

    #[test]
    fn sine_peaks_at_its_bin() {
        let sr = 22050.0;
        let f = 1000.0;
        let y: Vec<f32> = (0..22050)
            .map(|i| (std::f32::consts::TAU * f * i as f32 / sr).sin())
            .collect();
        let s = Stft::new(2048, 512);
        let m = s.magnitude(&y);
        let row = m.row(20);
        let peak = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let expected = (f * 2048.0 / sr).round() as usize;
        assert!(
            peak.abs_diff(expected) <= 1,
            "peak bin {peak}, expected {expected}"
        );
    }

    #[test]
    fn unit_mask_reconstructs_signal() {
        let y: Vec<f32> = (0..10_000)
            .map(|i| ((i as f32) * 0.01).sin() * 0.3 + ((i * 7 % 13) as f32 - 6.0) * 0.01)
            .collect();
        let s = Stft::new(2048, 512);
        let ones = Matrix::from_vec(
            s.n_frames(y.len()),
            s.n_bins(),
            vec![1.0; s.n_frames(y.len()) * s.n_bins()],
        );
        let r = s.istft_masked(&y, &ones);
        let err = y
            .iter()
            .zip(&r)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(err < 1e-4, "max reconstruction error {err}");
    }

    #[test]
    fn transpose_roundtrip() {
        let m = Matrix::from_vec(2, 3, vec![1., 2., 3., 4., 5., 6.]);
        let t = m.transpose();
        assert_eq!(t.row(0), &[1., 4.]);
        assert_eq!(t.transpose(), m);
    }
}
