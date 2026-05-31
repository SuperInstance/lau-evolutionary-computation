use rand::Rng;
use crate::individual::Population;

/// Tournament selection: pick `k` individuals randomly, return the best.
pub fn tournament_selection(
    pop: &Population,
    k: usize,
    rng: &mut impl Rng,
) -> usize {
    let mut best_idx = pop.len();
    let mut best_fitness = f64::NEG_INFINITY;
    for _ in 0..k {
        let idx = rng.gen_range(0..pop.len());
        let fit = pop[idx].fitness.unwrap_or(f64::NEG_INFINITY);
        if fit > best_fitness {
            best_fitness = fit;
            best_idx = idx;
        }
    }
    best_idx
}

/// Roulette wheel (fitness-proportionate) selection.
/// Returns the selected index. Fitnesses must be non-negative.
pub fn roulette_wheel_selection(pop: &Population, rng: &mut impl Rng) -> usize {
    let fitnesses: Vec<f64> = pop.iter().map(|i| i.fitness.unwrap_or(0.0).max(0.0)).collect();
    let total: f64 = fitnesses.iter().sum();
    if total <= 0.0 {
        return rng.gen_range(0..pop.len());
    }
    let mut r = rng.gen_range(0.0..total);
    for (i, &f) in fitnesses.iter().enumerate() {
        r -= f;
        if r <= 0.0 {
            return i;
        }
    }
    pop.len() - 1
}

/// Rank-based selection. Individuals are ranked by fitness; selection probability
/// is proportional to rank rather than raw fitness.
pub fn rank_based_selection(pop: &Population, pressure: f64, rng: &mut impl Rng) -> usize {
    let n = pop.len();
    if n == 0 { return 0; }

    let mut indexed: Vec<(usize, f64)> = pop
        .iter()
        .enumerate()
        .map(|(i, ind)| (i, ind.fitness.unwrap_or(0.0)))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Assign rank weights: rank 0 = best, gets highest weight
    let weights: Vec<f64> = (0..n)
        .map(|rank| 2.0 - pressure + 2.0 * (pressure - 1.0) * ((n - 1 - rank) as f64) / ((n - 1) as f64).max(1.0))
        .collect();

    let total: f64 = weights.iter().sum();
    let mut r = rng.gen_range(0.0..total);
    for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
        r -= weights[rank];
        if r <= 0.0 {
            return orig_idx;
        }
    }
    indexed.last().map(|(i, _)| *i).unwrap_or(0)
}

/// Select `n` parents using the given method.
#[derive(Clone, Debug)]
pub enum SelectionMethod {
    Tournament { k: usize },
    RouletteWheel,
    RankBased { pressure: f64 },
}

pub fn select_parents(
    pop: &Population,
    n: usize,
    method: &SelectionMethod,
    rng: &mut impl Rng,
) -> Vec<usize> {
    (0..n)
        .map(|_| match method {
            SelectionMethod::Tournament { k } => tournament_selection(pop, *k, rng),
            SelectionMethod::RouletteWheel => roulette_wheel_selection(pop, rng),
            SelectionMethod::RankBased { pressure } => rank_based_selection(pop, *pressure, rng),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::Individual;
    use super::*;

    fn make_pop(fitnesses: Vec<f64>) -> Population {
        fitnesses
            .into_iter()
            .enumerate()
            .map(|(i, f)| Individual::new(vec![i as f64]).with_fitness(f))
            .collect()
    }

    #[test]
    fn test_tournament_selects_from_pool() {
        let pop = make_pop(vec![1.0, 5.0, 3.0, 2.0]);
        let mut rng = rand::thread_rng();
        let idx = tournament_selection(&pop, 2, &mut rng);
        assert!(idx < pop.len());
    }

    #[test]
    fn test_tournament_biases_toward_fitter() {
        let pop = make_pop(vec![1.0, 100.0, 1.0, 1.0]);
        let mut rng = rand::thread_rng();
        let mut counts = vec![0usize; 4];
        for _ in 0..5000 {
            let idx = tournament_selection(&pop, 3, &mut rng);
            counts[idx] += 1;
        }
        // Index 1 (fitness 100) should be selected most often with k=3
        assert!(counts[1] > counts[0]);
        assert!(counts[1] > counts[2]);
        assert!(counts[1] > counts[3]);
    }

    #[test]
    fn test_roulette_wheel_biases() {
        let pop = make_pop(vec![1.0, 10.0, 1.0, 1.0]);
        let mut rng = rand::thread_rng();
        let mut counts = vec![0usize; 4];
        for _ in 0..10000 {
            let idx = roulette_wheel_selection(&pop, &mut rng);
            counts[idx] += 1;
        }
        assert!(counts[1] > counts[0]);
    }

    #[test]
    fn test_rank_based_biases() {
        let pop = make_pop(vec![1.0, 100.0, 50.0, 0.5]);
        let mut rng = rand::thread_rng();
        let mut counts = vec![0usize; 4];
        for _ in 0..5000 {
            let idx = rank_based_selection(&pop, 2.0, &mut rng);
            counts[idx] += 1;
        }
        // Index 1 (fitness 100 = rank 0) should be selected most
        assert!(counts[1] > counts[0]);
        assert!(counts[1] > counts[3]);
    }

    #[test]
    fn test_select_parents_tournament() {
        let pop = make_pop(vec![1.0, 2.0, 3.0, 4.0]);
        let mut rng = rand::thread_rng();
        let parents = select_parents(&pop, 6, &SelectionMethod::Tournament { k: 2 }, &mut rng);
        assert_eq!(parents.len(), 6);
        for &p in &parents {
            assert!(p < pop.len());
        }
    }

    #[test]
    fn test_roulette_handles_zero_fitness() {
        let pop = make_pop(vec![0.0, 0.0, 0.0]);
        let mut rng = rand::thread_rng();
        let idx = roulette_wheel_selection(&pop, &mut rng);
        assert!(idx < pop.len());
    }
}
