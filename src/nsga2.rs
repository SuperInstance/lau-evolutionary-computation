use rand::Rng;
use crate::individual::{Individual, Population};

/// Check if `a` Pareto-dominates `b` (all objectives >= and at least one >).
/// We assume we're maximizing all objectives.
pub fn dominates(a_obj: &[f64], b_obj: &[f64]) -> bool {
    let mut any_better = false;
    for (a, b) in a_obj.iter().zip(b_obj.iter()) {
        if a < b {
            return false;
        }
        if a > b {
            any_better = true;
        }
    }
    any_better
}

/// Fast non-dominated sort. Returns fronts (vectors of indices).
pub fn fast_non_dominated_sort(pop: &mut Population) -> Vec<Vec<usize>> {
    let n = pop.len();
    let mut domination_count = vec![0usize; n];
    let mut dominated_set: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut fronts: Vec<Vec<usize>> = vec![Vec::new()];

    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(&pop[i].objectives, &pop[j].objectives) {
                dominated_set[i].push(j);
                domination_count[j] += 1;
            } else if dominates(&pop[j].objectives, &pop[i].objectives) {
                dominated_set[j].push(i);
                domination_count[i] += 1;
            }
        }
        if domination_count[i] == 0 {
            pop[i].rank = Some(0);
            fronts[0].push(i);
        }
    }

    let mut fi = 0;
    while fi < fronts.len() && !fronts[fi].is_empty() {
        let mut next_front = Vec::new();
        for &i in &fronts[fi] {
            for &j in &dominated_set[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    pop[j].rank = Some(fi + 1);
                    next_front.push(j);
                }
            }
        }
        if !next_front.is_empty() {
            fronts.push(next_front);
        }
        fi += 1;
    }

    fronts
}

/// Compute crowding distance for a front.
pub fn crowding_distance(pop: &mut Population, front: &[usize]) {
    if front.len() <= 2 {
        for &i in front {
            pop[i].crowding_distance = f64::INFINITY;
        }
        return;
    }

    let num_objectives = pop[front[0]].objectives.len();
    for &i in front {
        pop[i].crowding_distance = 0.0;
    }

    for m in 0..num_objectives {
        let mut sorted: Vec<usize> = front.to_vec();
        sorted.sort_by(|&a, &b| {
            pop[a].objectives[m].partial_cmp(&pop[b].objectives[m]).unwrap()
        });

        let f_min = pop[sorted[0]].objectives[m];
        let f_max = pop[sorted[sorted.len() - 1]].objectives[m];

        pop[sorted[0]].crowding_distance = f64::INFINITY;
        pop[sorted[sorted.len() - 1]].crowding_distance = f64::INFINITY;

        if (f_max - f_min).abs() > 1e-10 {
            for k in 1..sorted.len() - 1 {
                let prev = pop[sorted[k - 1]].objectives[m];
                let next = pop[sorted[k + 1]].objectives[m];
                pop[sorted[k]].crowding_distance += (next - prev) / (f_max - f_min);
            }
        }
    }
}

/// Crowding-comparison operator: lower rank is better, then higher crowding distance.
pub fn crowding_comparison(a: &Individual, b: &Individual) -> std::cmp::Ordering {
    let rank_a = a.rank.unwrap_or(usize::MAX);
    let rank_b = b.rank.unwrap_or(usize::MAX);
    match rank_a.cmp(&rank_b) {
        std::cmp::Ordering::Equal => b.crowding_distance.partial_cmp(&a.crowding_distance).unwrap_or(std::cmp::Ordering::Equal),
        other => other,
    }
}

/// NSGA-II configuration.
#[derive(Clone, Debug)]
pub struct NSGA2Config {
    pub pop_size: usize,
    pub genome_len: usize,
    pub bounds: Vec<(f64, f64)>,
    pub max_generations: usize,
    pub crossover_rate: f64,
    pub mutation_sigma: f64,
    pub mutation_rate: f64,
}

impl Default for NSGA2Config {
    fn default() -> Self {
        Self {
            pop_size: 100,
            genome_len: 10,
            bounds: vec![(0.0, 1.0); 10],
            max_generations: 100,
            crossover_rate: 0.9,
            mutation_sigma: 0.1,
            mutation_rate: 0.1,
        }
    }
}

/// Result of NSGA-II run.
#[derive(Clone, Debug)]
pub struct NSGA2Result {
    pub final_population: Population,
    pub generations: usize,
}

/// Run NSGA-II for multi-objective optimization.
pub fn run_nsga2(
    config: &NSGA2Config,
    objectives: &[Box<dyn Fn(&[f64]) -> f64>],
) -> NSGA2Result {
    let mut rng = rand::thread_rng();
    let _num_obj = objectives.len();

    // Initialize
    let mut pop: Population = (0..config.pop_size)
        .map(|_| Individual::random(config.genome_len, &config.bounds, &mut rng))
        .collect();

    // Evaluate objectives
    for ind in pop.iter_mut() {
        ind.objectives = objectives.iter().map(|f| f(&ind.genome)).collect();
        ind.fitness = Some(ind.objectives.iter().sum());
    }

    for _gen in 0..config.max_generations {
        // Non-dominated sort
        let fronts = fast_non_dominated_sort(&mut pop);

        // Crowding distance
        for front in &fronts {
            let front_idx: Vec<usize> = front.clone();
            crowding_distance(&mut pop, &front_idx);
        }

        // Create offspring via tournament selection + crossover + mutation
        let mut offspring = Vec::with_capacity(config.pop_size);
        while offspring.len() < config.pop_size {
            let mut pick = || {
                let i = rng.gen_range(0..pop.len());
                let j = rng.gen_range(0..pop.len());
                if crowding_comparison(&pop[i], &pop[j]) == std::cmp::Ordering::Less {
                    j
                } else {
                    i
                }
            };

            let p1 = pick();
            let p2 = pick();

            let (mut c1, mut c2) = if rng.gen::<f64>() < config.crossover_rate {
                crate::crossover::simulated_binary_crossover(&pop[p1], &pop[p2], 20.0, &config.bounds, &mut rng)
            } else {
                (pop[p1].clone(), pop[p2].clone())
            };

            crate::mutation::gaussian_mutation(&mut c1, config.mutation_rate, config.mutation_sigma, &mut rng);
            crate::mutation::gaussian_mutation(&mut c2, config.mutation_rate, config.mutation_sigma, &mut rng);

            c1.objectives = objectives.iter().map(|f| f(&c1.genome)).collect();
            c1.fitness = Some(c1.objectives.iter().sum());
            c2.objectives = objectives.iter().map(|f| f(&c2.genome)).collect();
            c2.fitness = Some(c2.objectives.iter().sum());

            offspring.push(c1);
            if offspring.len() < config.pop_size {
                offspring.push(c2);
            }
        }

        // Combine parent + offspring
        let mut combined: Population = pop.into_iter().chain(offspring.into_iter()).collect();
        for ind in combined.iter_mut() {
            ind.rank = None;
            ind.crowding_distance = 0.0;
        }

        let fronts = fast_non_dominated_sort(&mut combined);
        for front in &fronts {
            let front_idx: Vec<usize> = front.clone();
            crowding_distance(&mut combined, &front_idx);
        }

        // Select next generation
        let mut new_pop = Vec::with_capacity(config.pop_size);
        for front in &fronts {
            if new_pop.len() + front.len() <= config.pop_size {
                for &i in front {
                    new_pop.push(combined[i].clone());
                }
            } else {
                let remaining = config.pop_size - new_pop.len();
                let mut sorted_front: Vec<usize> = front.clone();
                sorted_front.sort_by(|&a, &b| {
                    crowding_comparison(&combined[a], &combined[b])
                });
                for &i in sorted_front.iter().take(remaining) {
                    new_pop.push(combined[i].clone());
                }
                break;
            }
        }

        pop = new_pop;
    }

    NSGA2Result {
        final_population: pop,
        generations: config.max_generations,
    }
}

/// Extract the Pareto front from a population.
pub fn extract_pareto_front(pop: &Population) -> Vec<usize> {
    let n = pop.len();
    (0..n).filter(|&i| {
        (0..n).all(|j| j == i || !dominates(&pop[j].objectives, &pop[i].objectives))
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dominates_basic() {
        assert!(dominates(&[2.0, 2.0], &[1.0, 1.0]));
        assert!(!dominates(&[1.0, 2.0], &[2.0, 1.0]));
        assert!(!dominates(&[1.0, 1.0], &[1.0, 1.0]));
    }

    #[test]
    fn test_non_dominated_sort() {
        let mut pop: Population = vec![
            Individual::new(vec![0.0]).with_objectives(vec![1.0, 5.0]),
            Individual::new(vec![0.0]).with_objectives(vec![3.0, 3.0]),
            Individual::new(vec![0.0]).with_objectives(vec![5.0, 1.0]),
            Individual::new(vec![0.0]).with_objectives(vec![0.5, 0.5]),
        ];
        let fronts = fast_non_dominated_sort(&mut pop);
        // Individual 3 (0.5, 0.5) is dominated by all others
        assert!(fronts.len() >= 2);
        // Front 0 should contain indices 0, 1, 2 (non-dominated)
        assert!(fronts[0].contains(&0));
        assert!(fronts[0].contains(&1));
        assert!(fronts[0].contains(&2));
    }

    #[test]
    fn test_crowding_distance_boundary_infinite() {
        let mut pop: Population = vec![
            Individual::new(vec![0.0]).with_objectives(vec![1.0, 5.0]),
            Individual::new(vec![0.0]).with_objectives(vec![3.0, 3.0]),
            Individual::new(vec![0.0]).with_objectives(vec![5.0, 1.0]),
        ];
        crowding_distance(&mut pop, &[0, 1, 2]);
        assert_eq!(pop[0].crowding_distance, f64::INFINITY);
        assert_eq!(pop[2].crowding_distance, f64::INFINITY);
    }

    #[test]
    fn test_nsga2_basic_run() {
        let config = NSGA2Config {
            pop_size: 30,
            genome_len: 5,
            bounds: vec![(0.0, 1.0); 5],
            max_generations: 30,
            crossover_rate: 0.9,
            mutation_sigma: 0.05,
            mutation_rate: 0.1,
        };

        // Two objectives: minimize sum and minimize sum of squares (maximize negatives)
        let obj1: Box<dyn Fn(&[f64]) -> f64> = Box::new(|x: &[f64]| -x.iter().sum::<f64>());
        let obj2: Box<dyn Fn(&[f64]) -> f64> = Box::new(|x: &[f64]| -x.iter().map(|xi| xi * xi).sum::<f64>());

        let result = run_nsga2(&config, &[obj1, obj2]);
        assert_eq!(result.final_population.len(), 30);

        // Should have a Pareto front
        let front = extract_pareto_front(&result.final_population);
        assert!(front.len() > 0);
        assert!(front.len() <= 30);
    }

    #[test]
    fn test_pareto_front_extraction() {
        let pop: Population = vec![
            Individual::new(vec![0.0]).with_objectives(vec![1.0, 5.0]),
            Individual::new(vec![0.0]).with_objectives(vec![3.0, 3.0]),
            Individual::new(vec![0.0]).with_objectives(vec![5.0, 1.0]),
            Individual::new(vec![0.0]).with_objectives(vec![0.5, 0.5]),
        ];
        let front = extract_pareto_front(&pop);
        assert!(front.contains(&0));
        assert!(front.contains(&1));
        assert!(front.contains(&2));
        assert!(!front.contains(&3));
    }

    #[test]
    fn test_crowding_comparison() {
        let mut a = Individual::new(vec![0.0]);
        a.rank = Some(0);
        a.crowding_distance = 2.0;
        let mut b = Individual::new(vec![0.0]);
        b.rank = Some(1);
        b.crowding_distance = 5.0;
        assert_eq!(crowding_comparison(&a, &b), std::cmp::Ordering::Less);
    }
}
