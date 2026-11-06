//! Summary statistics (numpy semantics: population std, linear-interpolated percentiles).

use serde::{Deserialize, Serialize};

/// Mean and (population) standard deviation of a per-frame feature.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Stat {
    pub mean: f32,
    pub std: f32,
}

impl Stat {
    pub fn of(x: &[f32]) -> Self {
        Self {
            mean: mean(x),
            std: std(x),
        }
    }
}

pub fn mean(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|&v| v as f64).sum::<f64>() / x.len() as f64) as f32
}

pub fn std(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let m = x.iter().map(|&v| v as f64).sum::<f64>() / x.len() as f64;
    let var = x.iter().map(|&v| (v as f64 - m).powi(2)).sum::<f64>() / x.len() as f64;
    var.sqrt() as f32
}

/// `np.percentile(x, q)` with the default linear interpolation. Returns 0 for empty input.
pub fn percentile(x: &[f32], q: f64) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let mut s = x.to_vec();
    s.sort_by(f32::total_cmp);
    let pos = (q / 100.0).clamp(0.0, 1.0) * (s.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    (s[lo] as f64 + (s[hi] as f64 - s[lo] as f64) * frac) as f32
}

pub fn median(x: &[f32]) -> f32 {
    percentile(x, 50.0)
}

/// Pearson correlation; 0 when either input is constant.
pub fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let ma = a[..n].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let mb = b[..n].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let (mut sab, mut saa, mut sbb) = (0f64, 0f64, 0f64);
    for i in 0..n {
        let da = a[i] as f64 - ma;
        let db = b[i] as f64 - mb;
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return 0.0;
    }
    (sab / (saa.sqrt() * sbb.sqrt())) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_numpy() {
        let x = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(mean(&x), 2.5);
        assert!((std(&x) - 1.118034).abs() < 1e-6);
        assert!((percentile(&x, 90.0) - 3.7).abs() < 1e-6);
        assert_eq!(median(&x), 2.5);
    }

    #[test]
    fn pearson_basic() {
        assert!((pearson(&[1., 2., 3.], &[2., 4., 6.]) - 1.0).abs() < 1e-6);
        assert!((pearson(&[1., 2., 3.], &[3., 2., 1.]) + 1.0).abs() < 1e-6);
        assert_eq!(pearson(&[1., 1., 1.], &[1., 2., 3.]), 0.0);
    }
}
