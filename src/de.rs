use rand::Rng;
use crate::individual::{Individual, Population, evaluate_population};

/// Differential evolution strategy variant.
#[derive(Clone, Debug)]
pub enum DEStrategy {
    /// DE/rand/1/bin
    Rand1Bin { f: f64, cr: f64 },
    /// DE/best/1/bin
    Best1Bin { f: f64, cr: f64 },
    /// DE/rand/2/bin
    Rand2Bin { f: f64, cr: f64 },
}

/// Configuration for differential evolution.
#[derive(Clone, Debug)]
pub struct DEConfig {
    pub pop_size: usize,
    pub genome_len: usize,
    pub bounds: Vec<(f64, f64)>,
    pub max_generations: usize,
    pub strategy: DEStrategy,
}

impl Default for DEConfig {
    fn default() -> Self {
        Self {
            pop_size: 50,
            genome_len: 10,
            bounds: vec![(-5.0, 5.0); 10],
            max_generations: 200,
            strategy: DEStrategy::Rand1Bin { f: 0.8, cr: 0.9 },
        }
    }
}

/// Result of a DE run.
#[derive(Clone, Debug)]
pub struct DEResult {
    pub best_individual: Individual,
    pub best_fitness_history: Vec<f64>,
    pub generations: usize,
}

/// Run differential evolution.
pub fn run_de(config: &DEConfig, fitness_fn: &dyn Fn(&[f64]) -> f64) -> DEResult {
    let mut rng = rand::thread_rng();

    // Initialize
    let mut pop: Population = (0..config.pop_size)
        .map(|_| Individual::random(config.genome_len, &config.bounds, &mut rng))
        .collect();
    evaluate_population(&mut pop, fitness_fn);

    let mut best_fitness_history = Vec::new();
    let mut best_ever = pop.iter().max_by(|a, b| {
        a.fitness.partial_cmp(&b.fitness).unwrap()
    }).unwrap().clone();
    best_fitness_history.push(best_ever.fitness.unwrap());

    for _gen in 0..config.max_generations {
        let best_idx = pop.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.fitness.partial_cmp(&b.fitness).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let mut new_pop = Vec::with_capacity(config.pop_size);

        for i in 0..config.pop_size {
            let trial = match &config.strategy {
                DEStrategy::Rand1Bin { f, cr } => {
                    de_rand1_bin(&pop, i, *f, *cr, &config.bounds, &mut rng)
                }
                DEStrategy::Best1Bin { f, cr } => {
                    de_best1_bin(&pop, i, best_idx, *f, *cr, &config.bounds, &mut rng)
                }
                DEStrategy::Rand2Bin { f, cr } => {
                    de_rand2_bin(&pop, i, *f, *cr, &config.bounds, &mut rng)
                }
            };

            let trial_fitness = fitness_fn(&trial);
            let target_fitness = pop[i].fitness.unwrap_or(f64::NEG_INFINITY);

            if trial_fitness >= target_fitness {
                new_pop.push(Individual::new(trial).with_fitness(trial_fitness));
            } else {
                new_pop.push(pop[i].clone());
            }
        }

        pop = new_pop;
        evaluate_population(&mut pop, fitness_fn);

        let gen_best = pop.iter().max_by(|a, b| {
            a.fitness.partial_cmp(&b.fitness).unwrap()
        }).unwrap();
        if gen_best.fitness > best_ever.fitness {
            best_ever = gen_best.clone();
        }
        best_fitness_history.push(best_ever.fitness.unwrap());
    }

    DEResult {
        best_individual: best_ever,
        best_fitness_history,
        generations: config.max_generations,
    }
}

fn pick_distinct(pop_size: usize, exclude: usize, n: usize, rng: &mut impl Rng) -> Vec<usize> {
    let mut indices = Vec::with_capacity(n);
    while indices.len() < n {
        let idx = rng.gen_range(0..pop_size);
        if idx != exclude && !indices.contains(&idx) {
            indices.push(idx);
        }
    }
    indices
}

fn de_rand1_bin(
    pop: &Population, i: usize, f: f64, cr: f64,
    bounds: &[(f64, f64)], rng: &mut impl Rng,
) -> Vec<f64> {
    let idxs = pick_distinct(pop.len(), i, 3, rng);
    let dim = pop[0].genome.len();
    let j_rand = rng.gen_range(0..dim);
    (0..dim).map(|j| {
        if rng.gen::<f64>() < cr || j == j_rand {
            let val = pop[idxs[0]].genome[j] + f * (pop[idxs[1]].genome[j] - pop[idxs[2]].genome[j]);
            clamp(val, bounds.get(j))
        } else {
            pop[i].genome[j]
        }
    }).collect()
}

fn de_best1_bin(
    pop: &Population, i: usize, best: usize, f: f64, cr: f64,
    bounds: &[(f64, f64)], rng: &mut impl Rng,
) -> Vec<f64> {
    let idxs = pick_distinct(pop.len(), i, 2, rng);
    let dim = pop[0].genome.len();
    let j_rand = rng.gen_range(0..dim);
    (0..dim).map(|j| {
        if rng.gen::<f64>() < cr || j == j_rand {
            let val = pop[best].genome[j] + f * (pop[idxs[0]].genome[j] - pop[idxs[1]].genome[j]);
            clamp(val, bounds.get(j))
        } else {
            pop[i].genome[j]
        }
    }).collect()
}

fn de_rand2_bin(
    pop: &Population, i: usize, f: f64, cr: f64,
    bounds: &[(f64, f64)], rng: &mut impl Rng,
) -> Vec<f64> {
    let idxs = pick_distinct(pop.len(), i, 5, rng);
    let dim = pop[0].genome.len();
    let j_rand = rng.gen_range(0..dim);
    (0..dim).map(|j| {
        if rng.gen::<f64>() < cr || j == j_rand {
            let val = pop[idxs[0]].genome[j]
                + f * (pop[idxs[1]].genome[j] - pop[idxs[2]].genome[j])
                + f * (pop[idxs[3]].genome[j] - pop[idxs[4]].genome[j]);
            clamp(val, bounds.get(j))
        } else {
            pop[i].genome[j]
        }
    }).collect()
}

fn clamp(val: f64, bounds: Option<&(f64, f64)>) -> f64 {
    if let Some((lo, hi)) = bounds {
        val.max(*lo).min(*hi)
    } else {
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::sphere;

    #[test]
    fn test_de_converges_on_sphere() {
        let config = DEConfig {
            pop_size: 50,
            genome_len: 5,
            bounds: vec![(-5.0, 5.0); 5],
            max_generations: 300,
            strategy: DEStrategy::Rand1Bin { f: 0.8, cr: 0.9 },
        };
        let result = run_de(&config, &|x| sphere(x));
        assert!(result.best_individual.fitness.unwrap() > -0.5,
            "DE should converge near sphere optimum, got: {:?}", result.best_individual.fitness);
    }

    #[test]
    fn test_de_best1_bin_converges() {
        let config = DEConfig {
            pop_size: 50,
            genome_len: 5,
            bounds: vec![(-5.0, 5.0); 5],
            max_generations: 300,
            strategy: DEStrategy::Best1Bin { f: 0.8, cr: 0.9 },
        };
        let result = run_de(&config, &|x| sphere(x));
        assert!(result.best_individual.fitness.unwrap() > -0.5);
    }

    #[test]
    fn test_de_rand2_bin_converges() {
        let config = DEConfig {
            pop_size: 60,
            genome_len: 5,
            bounds: vec![(-5.0, 5.0); 5],
            max_generations: 300,
            strategy: DEStrategy::Rand2Bin { f: 0.5, cr: 0.9 },
        };
        let result = run_de(&config, &|x| sphere(x));
        assert!(result.best_individual.fitness.unwrap() > -1.0);
    }

    #[test]
    fn test_de_improves_over_generations() {
        let config = DEConfig {
            pop_size: 30,
            genome_len: 3,
            bounds: vec![(-5.0, 5.0); 3],
            max_generations: 50,
            strategy: DEStrategy::Rand1Bin { f: 0.8, cr: 0.9 },
        };
        let result = run_de(&config, &|x| sphere(x));
        assert!(result.best_fitness_history.last().unwrap() > result.best_fitness_history.first().unwrap());
    }
}
