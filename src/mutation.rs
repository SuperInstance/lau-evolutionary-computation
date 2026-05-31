use rand::Rng;
use crate::individual::Individual;

/// Gaussian mutation: add N(0, sigma) to each gene with probability `rate`.
pub fn gaussian_mutation(ind: &mut Individual, rate: f64, sigma: f64, rng: &mut impl Rng) {
        let normal = rand_distr::Normal::new(0.0, sigma).unwrap();
    for gene in ind.genome.iter_mut() {
        if rng.gen::<f64>() < rate {
            *gene += rng.sample(normal);
        }
    }
}

/// Uniform mutation: replace each gene with a random value in bounds with probability `rate`.
pub fn uniform_mutation(
    ind: &mut Individual,
    rate: f64,
    bounds: &[(f64, f64)],
    rng: &mut impl Rng,
) {
    for (i, gene) in ind.genome.iter_mut().enumerate() {
        if rng.gen::<f64>() < rate {
            if let Some((lo, hi)) = bounds.get(i) {
                *gene = rng.gen_range(*lo..*hi);
            }
        }
    }
}

/// Adaptive mutation: the mutation rate adapts based on fitness improvement.
/// If no improvement for `stagnation` generations, increase sigma; otherwise decrease.
pub struct AdaptiveMutation {
    pub base_rate: f64,
    pub sigma: f64,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub adapt_factor: f64,
    pub stagnation_count: usize,
    pub best_fitness: f64,
}

impl AdaptiveMutation {
    pub fn new(rate: f64, sigma: f64) -> Self {
        Self {
            base_rate: rate,
            sigma,
            sigma_min: sigma * 0.1,
            sigma_max: sigma * 10.0,
            adapt_factor: 1.2,
            stagnation_count: 0,
            best_fitness: f64::NEG_INFINITY,
        }
    }

    pub fn update(&mut self, current_best: f64) {
        if current_best > self.best_fitness {
            self.best_fitness = current_best;
            self.stagnation_count = 0;
            self.sigma = (self.sigma / self.adapt_factor).max(self.sigma_min);
        } else {
            self.stagnation_count += 1;
            if self.stagnation_count > 5 {
                self.sigma = (self.sigma * self.adapt_factor).min(self.sigma_max);
                self.stagnation_count = 0;
            }
        }
    }

    pub fn mutate(&self, ind: &mut Individual, rng: &mut impl Rng) {
        gaussian_mutation(ind, self.base_rate, self.sigma, rng);
    }
}

/// Mutation operator enum.
#[derive(Clone, Debug)]
pub enum MutationMethod {
    Gaussian { rate: f64, sigma: f64 },
    Uniform { rate: f64, bounds: Vec<(f64, f64)> },
}

pub fn mutate(ind: &mut Individual, method: &MutationMethod, rng: &mut impl Rng) {
    match method {
        MutationMethod::Gaussian { rate, sigma } => {
            gaussian_mutation(ind, *rate, *sigma, rng);
        }
        MutationMethod::Uniform { rate, bounds } => {
            uniform_mutation(ind, *rate, bounds, rng);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_mutation_changes_some_genes() {
        let mut ind = Individual::new(vec![0.0; 100]);
        let mut rng = rand::thread_rng();
        gaussian_mutation(&mut ind, 1.0, 1.0, &mut rng);
        let changed = ind.genome.iter().filter(|&&g| g != 0.0).count();
        assert!(changed > 0);
    }

    #[test]
    fn test_gaussian_mutation_rate_controls_changes() {
        let mut ind = Individual::new(vec![0.0; 1000]);
        let mut rng = rand::thread_rng();
        gaussian_mutation(&mut ind, 0.1, 1.0, &mut rng);
        let changed = ind.genome.iter().filter(|&&g| g != 0.0).count();
        // ~10% should change, allow 3-20%
        assert!(changed > 10 && changed < 350);
    }

    #[test]
    fn test_uniform_mutation_within_bounds() {
        let bounds = vec![(0.0, 10.0); 5];
        let mut ind = Individual::new(vec![5.0; 5]);
        let mut rng = rand::thread_rng();
        uniform_mutation(&mut ind, 1.0, &bounds, &mut rng);
        for &g in &ind.genome {
            assert!(g >= 0.0 && g <= 10.0);
        }
    }

    #[test]
    fn test_adaptive_mutation_increases_on_stagnation() {
        let mut am = AdaptiveMutation::new(0.1, 1.0);
        am.best_fitness = 10.0;
        // No improvement
        am.update(5.0);
        am.update(5.0);
        am.update(5.0);
        am.update(5.0);
        am.update(5.0);
        am.update(5.0); // triggers stagnation increase
        assert!(am.sigma > 1.0);
    }

    #[test]
    fn test_adaptive_mutation_decreases_on_improvement() {
        let mut am = AdaptiveMutation::new(0.1, 1.0);
        am.update(10.0);
        am.update(20.0);
        assert!(am.sigma < 1.0);
    }

    #[test]
    fn test_mutation_enum_dispatch() {
        let mut ind1 = Individual::new(vec![0.0; 5]);
        let mut ind2 = Individual::new(vec![0.0; 5]);
        let mut rng = rand::thread_rng();
        mutate(&mut ind1, &MutationMethod::Gaussian { rate: 1.0, sigma: 1.0 }, &mut rng);
        mutate(&mut ind2, &MutationMethod::Uniform { rate: 1.0, bounds: vec![(0.0, 1.0); 5] }, &mut rng);
        assert!(ind1.genome.iter().any(|&g| g != 0.0));
        assert!(ind2.genome.iter().any(|&g| g != 0.0));
    }
}
