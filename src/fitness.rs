use rand::Rng;

/// NK Landscape: a tunably rugged fitness landscape.
/// N = number of genes, K = number of epistatic interactions per gene.
pub struct NKLandscape {
    pub n: usize,
    pub k: usize,
    /// For each gene i, the K+1 genes that contribute to its fitness (including itself).
    pub dependency: Vec<Vec<usize>>,
    /// For each gene i, a lookup table mapping each possible (K+1)-bit pattern to a fitness contribution.
    pub contributions: Vec<Vec<f64>>,
}

impl NKLandscape {
    pub fn new(n: usize, k: usize, rng: &mut impl Rng) -> Self {
        let mut dependency = Vec::with_capacity(n);
        let mut contributions = Vec::with_capacity(n);

        for i in 0..n {
            // Gene i depends on itself and K random others
            let mut deps = vec![i];
            let mut candidates: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            // Shuffle and take K
            for j in (0..candidates.len()).rev() {
                let r = rng.gen_range(0..=j);
                candidates.swap(r, j);
            }
            deps.extend(candidates.into_iter().take(k));
            deps.sort();
            deps.dedup();
            dependency.push(deps);

            // 2^(K+1) possible patterns
            let num_patterns = 1 << (k + 1);
            let table: Vec<f64> = (0..num_patterns).map(|_| rng.gen::<f64>()).collect();
            contributions.push(table);
        }

        Self { n, k, dependency, contributions }
    }

    /// Evaluate fitness of a binary genome.
    pub fn fitness(&self, genome: &[u8]) -> f64 {
        let mut total = 0.0;
        for i in 0..self.n {
            let deps = &self.dependency[i];
            let mut idx = 0usize;
            for (bit_pos, &dep) in deps.iter().enumerate() {
                if dep < genome.len() && genome[dep] == 1 {
                    idx |= 1 << bit_pos;
                }
            }
            total += self.contributions[i][idx % self.contributions[i].len()];
        }
        total / self.n as f64
    }

    /// Ruggedness metric: average number of local optima normalized.
    pub fn ruggedness(&self) -> f64 {
        // Approximate: sample random points and count how many are local optima
        let mut rng = rand::thread_rng();
        let samples = 200;
        let mut local_optima = 0;
        for _ in 0..samples {
            let genome: Vec<u8> = (0..self.n).map(|_| rng.gen_range(0..2)).collect();
            let fit = self.fitness(&genome);
            let is_local_opt = (0..self.n).all(|i| {
                let mut neighbor = genome.clone();
                neighbor[i] = 1 - neighbor[i];
                self.fitness(&neighbor) <= fit
            });
            if is_local_opt {
                local_optima += 1;
            }
        }
        local_optima as f64 / samples as f64
    }
}

/// Classic benchmark fitness functions.
pub fn sphere(x: &[f64]) -> f64 {
    -x.iter().map(|xi| xi * xi).sum::<f64>()
}

pub fn rastrigin(x: &[f64]) -> f64 {
    -x.iter()
        .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos() + 10.0)
        .sum::<f64>()
}

pub fn rosenbrock(x: &[f64]) -> f64 {
    if x.len() < 2 { return 0.0; }
    -x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum::<f64>()
}

pub fn ackley(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_cos: f64 = x.iter().map(|xi| (2.0 * std::f64::consts::PI * xi).cos()).sum();
    -(-20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp() + 20.0 + std::f64::consts::E)
}

pub fn griewank(x: &[f64]) -> f64 {
    let sum_sq: f64 = x.iter().map(|xi| xi * xi / 4000.0).sum();
    let prod: f64 = x.iter().enumerate()
        .map(|(i, xi)| (xi / ((i + 1) as f64).sqrt()).cos())
        .product::<f64>();
    -(1.0 + sum_sq - prod)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nk_landscape_fitness_in_range() {
        let mut rng = rand::thread_rng();
        let landscape = NKLandscape::new(10, 3, &mut rng);
        let genome = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1];
        let fit = landscape.fitness(&genome);
        assert!(fit >= 0.0 && fit <= 1.0);
    }

    #[test]
    fn test_nk_landscape_ruggedness() {
        let mut rng = rand::thread_rng();
        let landscape = NKLandscape::new(8, 4, &mut rng);
        let r = landscape.ruggedness();
        assert!(r >= 0.0 && r <= 1.0);
    }

    #[test]
    fn test_sphere_optimal_at_zero() {
        let x = vec![0.0; 5];
        assert!(sphere(&x).abs() < 1e-10);
        let x = vec![1.0; 5];
        assert!(sphere(&x) < 0.0);
    }

    #[test]
    fn test_rastrigin_optimal_at_zero() {
        let x = vec![0.0; 5];
        assert!(rastrigin(&x).abs() < 1e-10);
    }

    #[test]
    fn test_rosenbrock_optimal_at_one() {
        let x = vec![1.0; 5];
        assert!(rosenbrock(&x).abs() < 1e-10);
    }

    #[test]
    fn test_ackley_optimal_at_zero() {
        let x = vec![0.0; 5];
        let f = ackley(&x);
        assert!(f.abs() < 1e-5);
    }

    #[test]
    fn test_griewank_optimal_at_zero() {
        let x = vec![0.0; 5];
        assert!(griewank(&x).abs() < 1e-5);
    }

    #[test]
    fn test_sphere_negative_for_nonzero() {
        let x = vec![1.0, 2.0, 3.0];
        assert!(sphere(&x) < 0.0);
    }

    #[test]
    fn test_rosenbrock_negative_for_nonoptimal() {
        let x = vec![0.0, 0.0];
        assert!(rosenbrock(&x) < 0.0);
    }

    #[test]
    fn test_ackley_negative_for_nonzero() {
        let x = vec![1.0, 1.0];
        assert!(ackley(&x) < 0.0);
    }
}
