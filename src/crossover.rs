use rand::Rng;
use crate::individual::Individual;

/// Uniform crossover: each gene is taken from either parent with equal probability.
pub fn uniform_crossover(p1: &Individual, p2: &Individual, rng: &mut impl Rng) -> (Individual, Individual) {
    let genome1: Vec<f64> = p1.genome.iter().zip(p2.genome.iter())
        .map(|(&a, &b)| if rng.gen::<bool>() { a } else { b })
        .collect();
    let genome2: Vec<f64> = p1.genome.iter().zip(p2.genome.iter())
        .map(|(&a, &b)| if rng.gen::<bool>() { a } else { b })
        .collect();
    (Individual::new(genome1), Individual::new(genome2))
}

/// Single-point crossover.
pub fn single_point_crossover(p1: &Individual, p2: &Individual, rng: &mut impl Rng) -> (Individual, Individual) {
    let len = p1.genome.len().min(p2.genome.len());
    if len == 0 {
        return (p1.clone(), p2.clone());
    }
    let point = rng.gen_range(1..len);
    let mut g1 = p1.genome.clone();
    let mut g2 = p2.genome.clone();
    for i in point..len {
        g1[i] = p2.genome[i];
        g2[i] = p1.genome[i];
    }
    (Individual::new(g1), Individual::new(g2))
}

/// Two-point crossover.
pub fn two_point_crossover(p1: &Individual, p2: &Individual, rng: &mut impl Rng) -> (Individual, Individual) {
    let len = p1.genome.len().min(p2.genome.len());
    if len < 2 {
        return single_point_crossover(p1, p2, rng);
    }
    let mut pt1 = rng.gen_range(1..len);
    let mut pt2 = rng.gen_range(1..len);
    if pt1 > pt2 {
        std::mem::swap(&mut pt1, &mut pt2);
    }
    let mut g1 = p1.genome.clone();
    let mut g2 = p2.genome.clone();
    for i in pt1..=pt2.min(len - 1) {
        g1[i] = p2.genome[i];
        g2[i] = p1.genome[i];
    }
    (Individual::new(g1), Individual::new(g2))
}

/// Blend crossover (BLX-alpha). For each gene, the child value is sampled from
/// an extended interval [min - alpha*range, max + alpha*range].
pub fn blend_crossover(p1: &Individual, p2: &Individual, alpha: f64, rng: &mut impl Rng) -> (Individual, Individual) {
    let genome1: Vec<f64> = p1.genome.iter().zip(p2.genome.iter())
        .map(|(&a, &b)| {
            let lo = a.min(b);
            let hi = a.max(b);
            let range = hi - lo;
            rng.gen_range((lo - alpha * range)..=(hi + alpha * range))
        })
        .collect();
    let genome2: Vec<f64> = p1.genome.iter().zip(p2.genome.iter())
        .map(|(&a, &b)| {
            let lo = a.min(b);
            let hi = a.max(b);
            let range = hi - lo;
            rng.gen_range((lo - alpha * range)..=(hi + alpha * range))
        })
        .collect();
    (Individual::new(genome1), Individual::new(genome2))
}

/// Simulated binary crossover (SBX). Commonly used in NSGA-II.
pub fn simulated_binary_crossover(
    p1: &Individual,
    p2: &Individual,
    eta: f64,
    bounds: &[(f64, f64)],
    rng: &mut impl Rng,
) -> (Individual, Individual) {
    let n = p1.genome.len();
    let mut g1 = vec![0.0; n];
    let mut g2 = vec![0.0; n];
    for j in 0..n {
        let a = p1.genome[j];
        let b = p2.genome[j];
        if (a - b).abs() < 1e-14 {
            g1[j] = a;
            g2[j] = b;
            continue;
        }
        let (y1, y2) = if a < b { (a, b) } else { (b, a) };
        let (lo, hi) = bounds.get(j).copied().unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
        let rand_val = rng.gen::<f64>();
        let beta_q = if rand_val <= 0.5 {
            (2.0 * rand_val).powf(1.0 / (eta + 1.0))
        } else {
            (1.0 / (2.0 * (1.0 - rand_val))).powf(1.0 / (eta + 1.0))
        };
        g1[j] = (0.5 * ((y1 + y2) - beta_q * (y2 - y1))).max(lo).min(hi);
        g2[j] = (0.5 * ((y1 + y2) + beta_q * (y2 - y1))).max(lo).min(hi);
    }
    (Individual::new(g1), Individual::new(g2))
}

/// Crossover operator enum.
#[derive(Clone, Debug)]
pub enum CrossoverMethod {
    Uniform,
    SinglePoint,
    TwoPoint,
    Blend { alpha: f64 },
}

pub fn crossover(
    p1: &Individual,
    p2: &Individual,
    method: &CrossoverMethod,
    rng: &mut impl Rng,
) -> (Individual, Individual) {
    match method {
        CrossoverMethod::Uniform => uniform_crossover(p1, p2, rng),
        CrossoverMethod::SinglePoint => single_point_crossover(p1, p2, rng),
        CrossoverMethod::TwoPoint => two_point_crossover(p1, p2, rng),
        CrossoverMethod::Blend { alpha } => blend_crossover(p1, p2, *alpha, rng),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parents() -> (Individual, Individual) {
        (
            Individual::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            Individual::new(vec![10.0, 20.0, 30.0, 40.0, 50.0]),
        )
    }

    #[test]
    fn test_uniform_crossover_genes_from_parents() {
        let (p1, p2) = parents();
        let mut rng = rand::thread_rng();
        let (c1, _c2) = uniform_crossover(&p1, &p2, &mut rng);
        for i in 0..5 {
            assert!(
                (c1.genome[i] == p1.genome[i] || c1.genome[i] == p2.genome[i]),
                "gene must come from a parent"
            );
        }
    }

    #[test]
    fn test_single_point_produces_valid_children() {
        let (p1, p2) = parents();
        let mut rng = rand::thread_rng();
        let (c1, c2) = single_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.genome.len(), 5);
        assert_eq!(c2.genome.len(), 5);
    }

    #[test]
    fn test_two_point_produces_valid_children() {
        let (p1, p2) = parents();
        let mut rng = rand::thread_rng();
        let (c1, c2) = two_point_crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.genome.len(), 5);
        assert_eq!(c2.genome.len(), 5);
    }

    #[test]
    fn test_blend_crossover_within_bounds() {
        let (p1, p2) = parents();
        let mut rng = rand::thread_rng();
        let alpha = 0.5;
        for _ in 0..100 {
            let (c1, _) = blend_crossover(&p1, &p2, alpha, &mut rng);
            for i in 0..5 {
                let lo = p1.genome[i].min(p2.genome[i]);
                let hi = p1.genome[i].max(p2.genome[i]);
                let range = hi - lo;
                assert!(c1.genome[i] >= lo - alpha * range - 1e-10);
                assert!(c1.genome[i] <= hi + alpha * range + 1e-10);
            }
        }
    }

    #[test]
    fn test_crossover_enum_dispatch() {
        let (p1, p2) = parents();
        let mut rng = rand::thread_rng();
        for method in &[
            CrossoverMethod::Uniform,
            CrossoverMethod::SinglePoint,
            CrossoverMethod::TwoPoint,
            CrossoverMethod::Blend { alpha: 0.5 },
        ] {
            let (c1, c2) = crossover(&p1, &p2, method, &mut rng);
            assert_eq!(c1.genome.len(), 5);
            assert_eq!(c2.genome.len(), 5);
        }
    }
}
