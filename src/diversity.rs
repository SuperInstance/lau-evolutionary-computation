use crate::individual::Population;

/// Fitness sharing: reduces fitness of similar individuals to maintain diversity.
/// Uses a sharing function based on Euclidean distance with threshold `sigma_share`.
pub fn fitness_sharing(pop: &mut Population, sigma_share: f64) {
    let n = pop.len();
    if n == 0 { return; }

    let _dim = pop[0].genome.len();
    let mut shared_fitnesses = vec![0.0; n];

    for i in 0..n {
        let mut sh_sum = 0.0;
        for j in 0..n {
            let dist = euclidean_distance(&pop[i].genome, &pop[j].genome);
            sh_sum += sharing_function(dist, sigma_share);
        }
        let raw_fitness = pop[i].fitness.unwrap_or(0.0);
        shared_fitnesses[i] = if sh_sum > 0.0 { raw_fitness / sh_sum } else { raw_fitness };
    }

    for (i, ind) in pop.iter_mut().enumerate() {
        ind.fitness = Some(shared_fitnesses[i]);
    }
}

fn sharing_function(dist: f64, sigma: f64) -> f64 {
    if dist < sigma {
        1.0 - (dist / sigma).powi(2)
    } else {
        0.0
    }
}

/// Crowding-based diversity maintenance: replace the most similar individual
/// in the population with a new one if it's better.
pub fn crowding_replacement(
    pop: &mut Population,
    new_ind: crate::individual::Individual,
) {
    if pop.is_empty() {
        pop.push(new_ind);
        return;
    }

    let new_fit = new_ind.fitness.unwrap_or(f64::NEG_INFINITY);

    // Find most similar individual
    let mut best_idx = 0;
    let mut min_dist = f64::INFINITY;
    for (i, ind) in pop.iter().enumerate() {
        let dist = euclidean_distance(&new_ind.genome, &ind.genome);
        if dist < min_dist {
            min_dist = dist;
            best_idx = i;
        }
    }

    // Replace if new individual is better
    let old_fit = pop[best_idx].fitness.unwrap_or(f64::NEG_INFINITY);
    if new_fit > old_fit {
        pop[best_idx] = new_ind;
    }
}

/// Compute Euclidean distance between two vectors.
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute population diversity as average pairwise distance.
pub fn avg_pairwise_distance(pop: &Population) -> f64 {
    let n = pop.len();
    if n < 2 { return 0.0; }
    let mut total = 0.0;
    let mut count = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            total += euclidean_distance(&pop[i].genome, &pop[j].genome);
            count += 1;
        }
    }
    total / count as f64
}

/// Compute population entropy as a diversity metric.
pub fn population_entropy(pop: &Population, bins: usize) -> f64 {
    if pop.is_empty() || bins == 0 { return 0.0; }

    let dim = pop[0].genome.len();
    if dim == 0 { return 0.0; }

    let mut total_entropy = 0.0;

    for d in 0..dim {
        let vals: Vec<f64> = pop.iter().map(|ind| ind.genome[d]).collect();
        let min_val = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max_val - min_val).abs() < 1e-10 { continue; }

        let mut histogram = vec![0usize; bins];
        for &v in &vals {
            let bin = ((v - min_val) / (max_val - min_val) * (bins as f64 - 1e-10)) as usize;
            let bin = bin.min(bins - 1);
            histogram[bin] += 1;
        }

        let n = pop.len() as f64;
        let entropy: f64 = histogram.iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n;
                -p * p.log2()
            })
            .sum();

        total_entropy += entropy;
    }

    total_entropy / dim as f64
}

/// Spacing metric: measures how evenly spread the Pareto front is.
pub fn spacing(pop: &Population) -> f64 {
    let n = pop.len();
    if n < 2 { return 0.0; }

    let num_obj = pop[0].objectives.len().max(1);
    let mut distances = Vec::with_capacity(n);

    for i in 0..n {
        let mut min_dist = f64::INFINITY;
        for j in 0..n {
            if i == j { continue; }
            let d: f64 = (0..num_obj)
                .map(|k| (pop[i].objectives.get(k).unwrap_or(&0.0) - pop[j].objectives.get(k).unwrap_or(&0.0)).abs())
                .sum();
            if d < min_dist {
                min_dist = d;
            }
        }
        distances.push(min_dist);
    }

    let mean = distances.iter().sum::<f64>() / n as f64;
    let variance: f64 = distances.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::individual::Individual;

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_fitness_sharing_reduces_similar_fitness() {
        let mut pop: Population = vec![
            Individual::new(vec![0.0, 0.0]).with_fitness(10.0),
            Individual::new(vec![0.1, 0.1]).with_fitness(10.0),
            Individual::new(vec![100.0, 100.0]).with_fitness(10.0),
        ];
        fitness_sharing(&mut pop, 5.0);
        // The similar pair should have reduced fitness
        assert!(pop[0].fitness.unwrap() < 10.0);
        assert!(pop[1].fitness.unwrap() < 10.0);
        // The isolated one should be unchanged or less reduced
        assert!(pop[2].fitness.unwrap() >= pop[0].fitness.unwrap());
    }

    #[test]
    fn test_crowding_replacement() {
        let mut pop: Population = vec![
            Individual::new(vec![1.0]).with_fitness(5.0),
            Individual::new(vec![2.0]).with_fitness(3.0),
        ];
        let new = Individual::new(vec![1.1]).with_fitness(10.0);
        crowding_replacement(&mut pop, new);
        // Should replace index 0 (closest) since new fitness is better
        assert!(pop.iter().any(|ind| ind.fitness == Some(10.0)));
    }

    #[test]
    fn test_avg_pairwise_distance() {
        let pop: Population = vec![
            Individual::new(vec![0.0]),
            Individual::new(vec![10.0]),
            Individual::new(vec![20.0]),
        ];
        let avg = avg_pairwise_distance(&pop);
        // Distances: 10, 20, 10 => avg = 40/3 ≈ 13.33
        assert!((avg - 40.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_population_entropy_diverse() {
        let pop: Population = (0..10)
            .map(|i| Individual::new(vec![i as f64 * 10.0]))
            .collect();
        let entropy = population_entropy(&pop, 5);
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_population_entropy_uniform() {
        let pop: Population = (0..10)
            .map(|_| Individual::new(vec![5.0]))
            .collect();
        let entropy = population_entropy(&pop, 5);
        assert!(entropy.abs() < 1e-10);
    }

    #[test]
    fn test_spacing_metric() {
        let pop: Population = vec![
            Individual::new(vec![0.0]).with_objectives(vec![0.0, 10.0]),
            Individual::new(vec![0.0]).with_objectives(vec![5.0, 5.0]),
            Individual::new(vec![0.0]).with_objectives(vec![10.0, 0.0]),
        ];
        let s = spacing(&pop);
        assert!(s >= 0.0);
    }
}
