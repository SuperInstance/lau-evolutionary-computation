use rand::Rng;
use crate::individual::{Individual, Population, evaluate_population};
use crate::selection::{SelectionMethod, select_parents};
use crate::crossover::{CrossoverMethod, crossover};
use crate::mutation::{MutationMethod, mutate};

/// Configuration for a genetic algorithm run.
#[derive(Clone, Debug)]
pub struct GAConfig {
    pub pop_size: usize,
    pub genome_len: usize,
    pub bounds: Vec<(f64, f64)>,
    pub max_generations: usize,
    pub elite_count: usize,
    pub crossover_rate: f64,
    pub selection: SelectionMethod,
    pub crossover: CrossoverMethod,
    pub mutation: MutationMethod,
}

impl Default for GAConfig {
    fn default() -> Self {
        Self {
            pop_size: 100,
            genome_len: 10,
            bounds: vec![(-5.0, 5.0); 10],
            max_generations: 200,
            elite_count: 2,
            crossover_rate: 0.8,
            selection: SelectionMethod::Tournament { k: 3 },
            crossover: CrossoverMethod::Uniform,
            mutation: MutationMethod::Gaussian { rate: 0.1, sigma: 0.5 },
        }
    }
}

/// Result of a GA run.
#[derive(Clone, Debug)]
pub struct GAResult {
    pub best_individual: Individual,
    pub best_fitness_history: Vec<f64>,
    pub generations: usize,
}

/// Run a genetic algorithm.
pub fn run_ga(config: &GAConfig, fitness_fn: &dyn Fn(&[f64]) -> f64) -> GAResult {
    let mut rng = rand::thread_rng();

    // Initialize population
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
        // Sort by fitness descending
        pop.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        // Elitism: keep top N
        let mut new_pop: Population = pop[..config.elite_count.min(pop.len())].to_vec();

        // Generate offspring
        let parent_indices = select_parents(&pop, config.pop_size * 2, &config.selection, &mut rng);
        let mut i = 0;
        while new_pop.len() < config.pop_size && i + 1 < parent_indices.len() {
            let p1 = &pop[parent_indices[i]];
            let p2 = &pop[parent_indices[i + 1]];
            i += 2;

            let (mut c1, mut c2) = if rng.gen::<f64>() < config.crossover_rate {
                crossover(p1, p2, &config.crossover, &mut rng)
            } else {
                (p1.clone(), p2.clone())
            };

            mutate(&mut c1, &config.mutation, &mut rng);
            mutate(&mut c2, &config.mutation, &mut rng);
            c1.fitness = None;
            c2.fitness = None;

            new_pop.push(c1);
            if new_pop.len() < config.pop_size {
                new_pop.push(c2);
            }
        }

        evaluate_population(&mut new_pop, fitness_fn);
        pop = new_pop;

        // Track best
        let gen_best = pop.iter().max_by(|a, b| {
            a.fitness.partial_cmp(&b.fitness).unwrap()
        }).unwrap();
        if gen_best.fitness > best_ever.fitness {
            best_ever = gen_best.clone();
        }
        best_fitness_history.push(best_ever.fitness.unwrap());
    }

    GAResult {
        best_individual: best_ever,
        best_fitness_history,
        generations: config.max_generations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::sphere;

    #[test]
    fn test_ga_converges_on_sphere() {
        let config = GAConfig {
            pop_size: 100,
            genome_len: 5,
            bounds: vec![(-5.0, 5.0); 5],
            max_generations: 200,
            elite_count: 5,
            crossover_rate: 0.9,
            selection: SelectionMethod::Tournament { k: 3 },
            crossover: CrossoverMethod::Blend { alpha: 0.5 },
            mutation: MutationMethod::Gaussian { rate: 0.3, sigma: 0.5 },
        };
        let result = run_ga(&config, &|x| sphere(x));
        // Should get reasonably close to 0 (fitness close to 0)
        assert!(result.best_individual.fitness.unwrap() > -1.0,
            "GA should converge near sphere optimum, got fitness: {:?}", result.best_individual.fitness);
        assert_eq!(result.best_fitness_history.len(), 201);
    }

    #[test]
    fn test_ga_improves_over_generations() {
        let config = GAConfig {
            pop_size: 50,
            genome_len: 3,
            bounds: vec![(-5.0, 5.0); 3],
            max_generations: 50,
            elite_count: 2,
            crossover_rate: 0.8,
            selection: SelectionMethod::Tournament { k: 2 },
            crossover: CrossoverMethod::Uniform,
            mutation: MutationMethod::Gaussian { rate: 0.2, sigma: 0.5 },
        };
        let result = run_ga(&config, &|x| sphere(x));
        assert!(result.best_fitness_history.last().unwrap() > result.best_fitness_history.first().unwrap());
    }

    #[test]
    fn test_ga_with_different_selection_methods() {
        for sel in &[
            SelectionMethod::Tournament { k: 3 },
            SelectionMethod::RouletteWheel,
            SelectionMethod::RankBased { pressure: 2.0 },
        ] {
            let config = GAConfig {
                pop_size: 50,
                genome_len: 3,
                bounds: vec![(-5.0, 5.0); 3],
                max_generations: 100,
                elite_count: 2,
                crossover_rate: 0.8,
                selection: sel.clone(),
                crossover: CrossoverMethod::Blend { alpha: 0.5 },
                mutation: MutationMethod::Gaussian { rate: 0.2, sigma: 0.5 },
            };
            let result = run_ga(&config, &|x| sphere(x));
            assert!(result.best_individual.fitness.unwrap() > -2.0);
        }
    }
}
