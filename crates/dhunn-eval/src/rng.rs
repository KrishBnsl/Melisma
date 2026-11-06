//! Small deterministic PRNG (SplitMix64): reproducible simulations without a dependency.

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[lo, hi)`.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f64()
    }

    /// Uniform integer in `[lo, hi]`.
    pub fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as i64
    }

    pub fn chance(&mut self, p: f64) -> bool {
        self.f64() < p
    }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.next_u64() as usize % items.len()]
    }

    /// Index drawn with probability proportional to `weights` (all ≥ 0, not all 0).
    pub fn weighted(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let mut x = self.f64() * total;
        for (i, &w) in weights.iter().enumerate() {
            x -= w;
            if x < 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        let (mut a, mut b) = (Rng::new(7), Rng::new(7));
        for _ in 0..1000 {
            let x = a.f64();
            assert_eq!(x, b.f64());
            assert!((0.0..1.0).contains(&x));
            let i = a.int(3, 5);
            b.int(3, 5);
            assert!((3..=5).contains(&i));
        }
    }

    #[test]
    fn weighted_follows_weights() {
        let mut r = Rng::new(1);
        let mut counts = [0; 3];
        for _ in 0..30_000 {
            counts[r.weighted(&[1.0, 0.0, 2.0])] += 1;
        }
        assert_eq!(counts[1], 0);
        assert!(
            (counts[2] as f64 / counts[0] as f64 - 2.0).abs() < 0.15,
            "{counts:?}"
        );
    }
}
