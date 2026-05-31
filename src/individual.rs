use serde::{Deserialize, Serialize};

/// A single individual in the population, represented as a real-valued genome.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Individual {
    pub genome: Vec<f64>,
    pub fitness: Option<f64>,
    pub objectives: Vec<f64>,
    pub rank: Option<usize>,
    pub crowding_distance: f64,
}

impl Individual {
    pub fn new(genome: Vec<f64>) -> Self {
        let len = genome.len();
        Self {
            genome,
            fitness: None,
            objectives: vec![0.0; len.min(1)],
            rank: None,
            crowding_distance: 0.0,
        }
    }

    pub fn random(len: usize, bounds: &[(f64, f64)], rng: &mut impl rand::Rng) -> Self {
        let genome: Vec<f64> = bounds
            .iter()
            .take(len)
            .map(|(lo, hi)| rng.gen_range(*lo..*hi))
            .collect();
        Self::new(genome)
    }

    pub fn with_fitness(mut self, fitness: f64) -> Self {
        self.fitness = Some(fitness);
        self
    }

    pub fn with_objectives(mut self, objectives: Vec<f64>) -> Self {
        self.objectives = objectives;
        self
    }

    pub fn len(&self) -> usize {
        self.genome.len()
    }

    pub fn is_empty(&self) -> bool {
        self.genome.is_empty()
    }
}

/// A population of individuals.
pub type Population = Vec<Individual>;

/// Evaluate fitness for the entire population.
pub fn evaluate_population(pop: &mut Population, fitness_fn: &dyn Fn(&[f64]) -> f64) {
    for ind in pop.iter_mut() {
        if ind.fitness.is_none() {
            ind.fitness = Some(fitness_fn(&ind.genome));
        }
    }
}
