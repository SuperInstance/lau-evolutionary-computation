# lau-evolutionary-computation

[![crates.io](https://img.shields.io/badge/crates.io-0.1.0-orange)](https://crates.io/crates/lau-evolutionary-computation)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![docs](https://docs.rs/lau-evolutionary-computation/badge.svg)](https://docs.rs/lau-evolutionary-computation)

**Evolutionary computation in Rust** — genetic algorithms (GA), differential evolution (DE), NSGA-II multi-objective optimization, genetic programming (GP), fitness landscapes, and diversity maintenance.

55 tests · real-valued and tree-based genomes · pluggable selection/crossover/mutation · `serde`-serializable individuals

---

## What This Does

Evolutionary algorithms solve optimization problems by mimicking natural selection: maintain a population of candidate solutions, evaluate fitness, select the best, recombine them, mutate offspring, and repeat. This library provides:

- **Genetic Algorithm (GA)** — the workhorse. Tournament/roulette/rank selection, uniform/single-point/two-point/blend crossover, Gaussian/uniform/adaptive mutation, elitism
- **Differential Evolution (DE)** — powerful for continuous optimization. Three strategy variants: DE/rand/1/bin, DE/best/1/bin, DE/rand/2/bin
- **NSGA-II** — multi-objective optimization via fast non-dominated sorting + crowding distance. Returns the full Pareto front
- **Genetic Programming (GP)** — evolves expression trees, not vectors. Subtree crossover/mutation, ramped half-and-half initialization, eval with variable bindings
- **Fitness landscapes** — NK landscapes (tunable ruggedness) plus 5 classic benchmarks (Sphere, Rastrigin, Rosenbrock, Ackley, Griewank)
- **Diversity maintenance** — fitness sharing, crowding replacement, entropy-based diversity, pairwise distance, spacing metric for Pareto fronts

---

## Key Idea

All evolutionary algorithms share the same loop:

```
Initialize population → Evaluate fitness → [Select → Crossover → Mutate] → Repeat
```

The difference is in the *representation* and *operators*:

| Algorithm | Genome | Key Operator |
|-----------|--------|-------------|
| GA | Real-valued vector | Selection + crossover + mutation |
| DE | Real-valued vector | Differential mutation + binomial crossover |
| NSGA-II | Real-valued vector | Non-dominated sort + crowding distance |
| GP | Expression tree | Subtree crossover + subtree mutation |

This library unifies them under a shared `Individual` type and modular operator design. Mix and match selection, crossover, and mutation strategies.

---

## Install

```toml
[dependencies]
lau-evolutionary-computation = "0.1"
```

Requires **Rust 2021 edition**. Dependencies: `serde`, `nalgebra`, `rand`, `rand_distr`, `serde_json`.

---

## Quick Start

### Genetic Algorithm

```rust
use lau_evolutionary_computation::{
    ga::{GAConfig, run_ga},
    fitness::sphere,
    selection::SelectionMethod,
    crossover::CrossoverMethod,
    mutation::MutationMethod,
};

let config = GAConfig {
    pop_size: 100,
    genome_len: 10,
    bounds: vec![(-5.0, 5.0); 10],
    max_generations: 200,
    elite_count: 5,
    crossover_rate: 0.9,
    selection: SelectionMethod::Tournament { k: 3 },
    crossover: CrossoverMethod::Blend { alpha: 0.5 },
    mutation: MutationMethod::Gaussian { rate: 0.1, sigma: 0.5 },
};

let result = run_ga(&config, &|x| sphere(x));
println!("Best fitness: {:.6}", result.best_individual.fitness.unwrap());
println!("Best genome: {:?}", result.best_individual.genome);
println!("Generations: {}", result.generations);
```

### Differential Evolution

```rust
use lau_evolutionary_computation::{
    de::{DEConfig, DEStrategy, run_de},
    fitness::rastrigin,
};

let config = DEConfig {
    pop_size: 50,
    genome_len: 10,
    bounds: vec![(-5.12, 5.12); 10],
    max_generations: 300,
    strategy: DEStrategy::Best1Bin { f: 0.8, cr: 0.9 },
};

let result = run_de(&config, &|x| rastrigin(x));
println!("Best fitness: {:.6}", result.best_individual.fitness.unwrap());
```

### NSGA-II (Multi-Objective)

```rust
use lau_evolutionary_computation::nsga2::{NSGA2Config, run_nsga2};

let config = NSGA2Config {
    pop_size: 100,
    genome_len: 5,
    bounds: vec![(0.0, 1.0); 5],
    max_generations: 100,
    crossover_rate: 0.9,
    mutation_sigma: 0.1,
    mutation_rate: 0.1,
};

// Minimize both sum(x) and sum(x²) — maximize negatives
let obj1 = Box::new(|x: &[f64]| -x.iter().sum::<f64>()) as Box<dyn Fn(&[f64]) -> f64>;
let obj2 = Box::new(|x: &[f64]| -x.iter().map(|xi| xi * xi).sum::<f64>());

let result = run_nsga2(&config, &[obj1, obj2]);
println!("Final population: {} individuals", result.final_population.len());
```

### Genetic Programming

```rust
use lau_evolutionary_computation::gp::{GPTree, GPTerminal};
use std::collections::HashMap;

// Build a tree: (+ x (* 2 3))
let tree = GPTree::Function {
    name: "+".to_string(),
    children: vec![
        GPTree::Terminal(GPTerminal::Var("x".to_string())),
        GPTree::Function {
            name: "*".to_string(),
            children: vec![
                GPTree::Terminal(GPTerminal::Const(2.0)),
                GPTree::Terminal(GPTerminal::Const(3.0)),
            ],
        },
    ],
};

let mut vars = HashMap::new();
vars.insert("x".to_string(), 10.0);
assert_eq!(tree.eval(&vars), 16.0); // 10 + 2*3
```

---

## API Reference

### `individual` — Individuals and Populations

| Type | Description |
|------|-------------|
| `Individual` | A candidate solution: genome + fitness + objectives + rank + crowding distance |
| `Population` | Type alias for `Vec<Individual>` |
| `evaluate_population(pop, fitness_fn)` | Lazy fitness evaluation (skips already-evaluated individuals) |

`Individual` methods: `new(genome)`, `random(len, bounds, rng)`, `with_fitness(f)`, `with_objectives(objs)`, `len()`, `is_empty()`.

### `selection` — Parent Selection

| Method | Description |
|--------|-------------|
| `tournament_selection(pop, k, rng)` | Pick k random individuals, return the fittest |
| `roulette_wheel_selection(pop, rng)` | Fitness-proportionate selection |
| `rank_based_selection(pop, pressure, rng)` | Selection probability proportional to rank, not raw fitness |
| `SelectionMethod` enum | `Tournament { k }`, `RouletteWheel`, `RankBased { pressure }` |
| `select_parents(pop, n, method, rng)` | Select n parent indices |

### `crossover` — Recombination

| Function | Description |
|----------|-------------|
| `uniform_crossover(p1, p2, rng)` | Each gene randomly from either parent |
| `single_point_crossover(p1, p2, rng)` | Split genome at one point |
| `two_point_crossover(p1, p2, rng)` | Swap a segment between two points |
| `blend_crossover(p1, p2, alpha, rng)` | BLX-α: sample child from extended interval |
| `simulated_binary_crossover(p1, p2, eta, bounds, rng)` | SBX: simulates single-point crossover for real-valued genes |
| `CrossoverMethod` enum | `Uniform`, `SinglePoint`, `TwoPoint`, `Blend { alpha }` |

### `mutation` — Mutation Operators

| Function | Description |
|----------|-------------|
| `gaussian_mutation(ind, rate, sigma, rng)` | Add N(0, σ²) noise to each gene with probability `rate` |
| `uniform_mutation(ind, rate, bounds, rng)` | Replace each gene with a uniform random value with probability `rate` |
| `AdaptiveMutation` | Self-adjusting σ: shrinks on improvement, grows on stagnation |
| `MutationMethod` enum | `Gaussian { rate, sigma }`, `Uniform { rate, bounds }` |

### `fitness` — Benchmark Functions and Landscapes

| Function/Type | Description |
|---------------|-------------|
| `sphere(x)` | −Σxᵢ² — unimodal, global optimum at origin |
| `rastrigin(x)` | −Σ(xᵢ² − 10cos(2πxᵢ) + 10) — highly multimodal |
| `rosenbrock(x)` | −Σ(100(xᵢ₊₁ − xᵢ²)² + (1 − xᵢ)²) — narrow valley |
| `ackley(x)` | Classic multimodal with many local optima |
| `griewank(x)` | −(1 + Σxᵢ²/4000 − Πcos(xᵢ/√i)) |
| `NKLandscape` | Tunably rugged: N genes, K epistatic interactions per gene |
| `NKLandscape::ruggedness()` | Approximate local optima density via random sampling |

All benchmark functions return **negative** values (maximization convention). The global optimum of each is at or near 0.

### `ga` — Genetic Algorithm

| Type | Description |
|------|-------------|
| `GAConfig` | Full configuration: population size, genome length, bounds, generations, elitism, rates, operators |
| `GAResult` | `best_individual`, `best_fitness_history` (one entry per generation + initial), `generations` |
| `run_ga(config, fitness_fn)` | Execute the GA and return the result |

### `de` — Differential Evolution

| Type | Description |
|------|-------------|
| `DEConfig` | Population size, genome length, bounds, generations, strategy |
| `DEStrategy` | `Rand1Bin { f, cr }`, `Best1Bin { f, cr }`, `Rand2Bin { f, cr }` |
| `DEResult` | Same structure as `GAResult` |
| `run_de(config, fitness_fn)` | Execute DE with the chosen strategy |

### `nsga2` — Multi-Objective Optimization

| Type | Description |
|------|-------------|
| `NSGA2Config` | Population size, genome length, bounds, generations, crossover/mutation rates |
| `NSGA2Result` | `final_population` (the final Pareto-approximated set), `generations` |
| `run_nsga2(config, objectives)` | Run NSGA-II with multiple objective functions |
| `dominates(a, b)` | Pareto dominance check |
| `fast_non_dominated_sort(pop)` | Returns fronts as vectors of indices |
| `crowding_distance(pop, front)` | Compute crowding distances for a front |
| `extract_pareto_front(pop)` | Get indices of non-dominated individuals |

### `gp` — Genetic Programming

| Type | Description |
|------|-------------|
| `GPTree` | Expression tree: `Function { name, children }` or `Terminal(GPTerminal)` |
| `GPTerminal` | `Const(f64)` or `Var(String)` |
| `GPTree::eval(vars)` | Evaluate tree with variable bindings |
| `GPTree::random(max_depth, functions, terminals, rng)` | Ramped half-and-half generation |
| `GPTree::subtree_crossover(p1, p2, rng)` | Swap random subtrees between parents |
| `GPTree::subtree_mutation(max_depth, functions, terminals, rng)` | Replace random subtree with new random tree |
| `GPTree::size()`, `GPTree::depth()` | Tree metrics |

Supported operators: `+`, `-`, `*`, `/` (binary), `sin`, `cos` (unary). Division by zero returns 0.

### `diversity` — Diversity Maintenance

| Function | Description |
|----------|-------------|
| `fitness_sharing(pop, sigma_share)` | Reduce fitness of similar individuals to maintain diversity |
| `crowding_replacement(pop, new_ind)` | Replace the most similar individual if new one is better |
| `euclidean_distance(a, b)` | L2 distance between two vectors |
| `avg_pairwise_distance(pop)` | Mean pairwise distance across the population |
| `population_entropy(pop, bins)` | Shannon entropy of genome distribution per dimension |
| `spacing(pop)` | Pareto front evenness metric |

---

## How It Works

### Genetic Algorithm Flow

1. **Initialize**: random population within bounds
2. **Evaluate**: compute fitness for all individuals
3. **Loop** (max_generations times):
   - Sort by fitness, keep top `elite_count` (elitism)
   - Select parents via chosen method (tournament, roulette, rank)
   - Crossover parents with probability `crossover_rate`
   - Mutate offspring
   - Evaluate new population
   - Track best-ever individual
4. **Return**: best individual + fitness history

### Differential Evolution Flow

1. **Initialize**: random population
2. **Loop** (max_generations times):
   - For each individual i, create a **trial vector** via the DE strategy:
     - **rand/1**: v = x_r1 + F × (x_r2 − x_r3)
     - **best/1**: v = x_best + F × (x_r1 − x_r2)
     - **rand/2**: v = x_r1 + F × (x_r2 − x_r3) + F × (x_r4 − x_r5)
   - Binomial crossover with the target (probability CR)
   - **Greedy selection**: keep trial if better or equal to target
3. **Return**: best individual + fitness history

### NSGA-II Flow

1. **Initialize**: random population, evaluate all objectives
2. **Loop** (max_generations times):
   - **Non-dominated sort**: partition population into Pareto fronts (front 0 = best)
   - **Crowding distance**: for each front, compute how "spread out" individuals are
   - **Create offspring**: binary tournament (using crowding comparison) + SBX crossover + Gaussian mutation
   - **Merge**: combine parent + offspring (2N individuals)
   - **Select next generation**: fill from front 0, then front 1, etc. If a front doesn't fit entirely, use crowding distance to pick the most spread-out individuals
3. **Return**: final population (approximation of the Pareto front)

---

## The Math

### Tournament Selection

Pick k individuals uniformly at random. Return the one with highest fitness. Selection pressure increases with k — larger tournaments favor the fittest more strongly.

### BLX-α Crossover

For parent genes a and b, the child gene is sampled uniformly from:

> [min(a,b) − α(b−a), max(a,b) + α(b−a)]

α = 0 gives the standard interval; α > 0 extends the range, promoting exploration.

### Simulated Binary Crossover (SBX)

SBX approximates the spreading property of binary crossover for real-valued genes. The spread factor βₛ is:

> βₛ = (2u)^(1/(η+1)) if u ≤ 0.5
> βₛ = (1/(2(1−u)))^(1/(η+1)) if u > 0.5

Where u is uniform random and η is the distribution index. Large η = children close to parents.

### Differential Evolution Mutation

The DE/rand/1 mutation creates a donor vector:

> v = x_r1 + F × (x_r2 − x_r3)

F is the differential weight (typically 0.5–1.0). Binomial crossover then mixes v with the target:

> u_j = v_j if rand(0,1) < CR or j = j_rand, else x_j

### NK Landscapes

An NK landscape has N binary genes, each with K epistatic interactions. The fitness contribution of gene i depends on K+1 genes (itself + K others). Total fitness:

> F = (1/N) Σ f_i(pattern of dependent genes)

Higher K → more epistasis → more rugged landscape → harder optimization.

### Pareto Dominance

Solution a **dominates** b (a ≻ b) if a is at least as good in all objectives and strictly better in at least one:

> a ≻ b ⟺ ∀i: f_i(a) ≥ f_i(b) ∧ ∃j: f_j(a) > f_j(b)

The **Pareto front** is the set of all non-dominated solutions — you can't improve one objective without worsening another.

### Fitness Sharing

To maintain diversity, the fitness of individual i is divided by its **niche count**:

> f'_i = f_i / Σⱼ sh(d(i,j))

Where the sharing function sh(d) = max(0, 1 − (d/σ)²) and σ is the niche radius. Similar individuals share fitness, penalizing clustering.

---

## Test Coverage

| Module | Tests |
|--------|-------|
| `fitness` | 10 |
| `gp` | 8 |
| `mutation` | 6 |
| `selection` | 6 |
| `diversity` | 7 |
| `nsga2` | 6 |
| `de` | 4 |
| `ga` | 3 |
| `crossover` | 5 |
| **Total** | **55** |

Tests verify convergence on benchmark functions, operator correctness, diversity metrics, Pareto dominance, serde round-trips, and adaptive mutation behavior.

---

## License

MIT
