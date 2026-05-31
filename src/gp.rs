use rand::Rng;
use serde::{Deserialize, Serialize};

/// A tree node for genetic programming.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GPTree {
    /// A function node with an operator name and children.
    Function {
        name: String,
        children: Vec<GPTree>,
    },
    /// A terminal / leaf node.
    Terminal(GPTerminal),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GPTerminal {
    Const(f64),
    Var(String),
}

impl GPTree {
    /// Evaluate the tree given variable bindings.
    pub fn eval(&self, vars: &std::collections::HashMap<String, f64>) -> f64 {
        match self {
            GPTree::Function { name, children } => {
                let vals: Vec<f64> = children.iter().map(|c| c.eval(vars)).collect();
                match name.as_str() {
                    "+" => vals.iter().sum(),
                    "-" if vals.len() == 2 => vals[0] - vals[1],
                    "*" => vals.iter().product(),
                    "/" if vals.len() == 2 => {
                        if vals[1].abs() < 1e-10 { 0.0 } else { vals[0] / vals[1] }
                    }
                    "sin" if !vals.is_empty() => vals[0].sin(),
                    "cos" if !vals.is_empty() => vals[0].cos(),
                    _ => 0.0,
                }
            }
            GPTree::Terminal(GPTerminal::Const(v)) => *v,
            GPTree::Terminal(GPTerminal::Var(name)) => *vars.get(name).unwrap_or(&0.0),
        }
    }

    /// Count nodes in the tree.
    pub fn size(&self) -> usize {
        match self {
            GPTree::Function { children, .. } => 1 + children.iter().map(|c| c.size()).sum::<usize>(),
            GPTree::Terminal(_) => 1,
        }
    }

    /// Get the depth of the tree.
    pub fn depth(&self) -> usize {
        match self {
            GPTree::Function { children, .. } => {
                1 + children.iter().map(|c| c.depth()).max().unwrap_or(0)
            }
            GPTree::Terminal(_) => 1,
        }
    }

    /// Generate a random tree (ramped half-and-half).
    pub fn random(
        max_depth: usize,
        functions: &[&str],
        terminals: &[GPTerminal],
        rng: &mut impl Rng,
    ) -> Self {
        if max_depth <= 1 || rng.gen::<f64>() < 0.3 {
            let term = terminals[rng.gen_range(0..terminals.len())].clone();
            GPTree::Terminal(term)
        } else {
            let name = functions[rng.gen_range(0..functions.len())].to_string();
            let arity = if name == "sin" || name == "cos" { 1 } else { 2 };
            let children: Vec<GPTree> = (0..arity)
                .map(|_| GPTree::random(max_depth - 1, functions, terminals, rng))
                .collect();
            GPTree::Function { name, children }
        }
    }

    /// Pick a random node index (via BFS-like flattening) and return its path.
    fn random_node_index(&self, rng: &mut impl Rng) -> Vec<usize> {
        let total = self.size();
        let target = rng.gen_range(0..total);
        let mut path = Vec::new();
        let mut current = self;
        let mut remaining = target;
        'outer: loop {
            if let GPTree::Function { children, .. } = current {
                if remaining == 0 {
                    break;
                }
                remaining -= 1;
                for (ci, child) in children.iter().enumerate() {
                    let child_size = child.size();
                    if remaining < child_size {
                        path.push(ci);
                        current = child;
                        continue 'outer;
                    }
                    remaining -= child_size;
                }
                break;
            } else {
                break;
            }
        }
        path
    }

    fn get_subtree_mut(&mut self, path: &[usize]) -> &mut GPTree {
        if path.is_empty() {
            self
        } else if let GPTree::Function { children, .. } = self {
            children[path[0]].get_subtree_mut(&path[1..])
        } else {
            self
        }
    }

    /// Subtree crossover: swap random subtrees between two parents.
    pub fn subtree_crossover(
        p1: &GPTree,
        p2: &GPTree,
        rng: &mut impl Rng,
    ) -> (GPTree, GPTree) {
        let mut c1 = p1.clone();
        let mut c2 = p2.clone();
        let path1 = c1.random_node_index(rng);
        let path2 = c2.random_node_index(rng);
        let sub1 = c1.get_subtree_mut(&path1).clone();
        let sub2 = c2.get_subtree_mut(&path2).clone();
        *c1.get_subtree_mut(&path1) = sub2;
        *c2.get_subtree_mut(&path2) = sub1;
        (c1, c2)
    }

    /// Subtree mutation: replace a random subtree with a new random tree.
    pub fn subtree_mutation(
        &mut self,
        max_depth: usize,
        functions: &[&str],
        terminals: &[GPTerminal],
        rng: &mut impl Rng,
    ) {
        let path = self.random_node_index(rng);
        let new_sub = GPTree::random(max_depth, functions, terminals, rng);
        *self.get_subtree_mut(&path) = new_sub;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terminals() -> Vec<GPTerminal> {
        vec![
            GPTerminal::Const(1.0),
            GPTerminal::Const(2.0),
            GPTerminal::Var("x".to_string()),
        ]
    }

    fn functions() -> Vec<&'static str> {
        vec!["+", "-", "*", "/"]
    }

    #[test]
    fn test_gp_eval_addition() {
        let tree = GPTree::Function {
            name: "+".to_string(),
            children: vec![
                GPTree::Terminal(GPTerminal::Const(3.0)),
                GPTree::Terminal(GPTerminal::Const(4.0)),
            ],
        };
        let vars = std::collections::HashMap::new();
        assert!((tree.eval(&vars) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_gp_eval_variable() {
        let tree = GPTree::Terminal(GPTerminal::Var("x".to_string()));
        let mut vars = std::collections::HashMap::new();
        vars.insert("x".to_string(), 42.0);
        assert!((tree.eval(&vars) - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_gp_eval_complex() {
        // (+ x (* 2 3))
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
        let mut vars = std::collections::HashMap::new();
        vars.insert("x".to_string(), 10.0);
        assert!((tree.eval(&vars) - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_gp_division_by_zero() {
        let tree = GPTree::Function {
            name: "/".to_string(),
            children: vec![
                GPTree::Terminal(GPTerminal::Const(1.0)),
                GPTree::Terminal(GPTerminal::Const(0.0)),
            ],
        };
        let vars = std::collections::HashMap::new();
        assert_eq!(tree.eval(&vars), 0.0);
    }

    #[test]
    fn test_random_tree_has_valid_size() {
        let mut rng = rand::thread_rng();
        let tree = GPTree::random(5, &functions(), &terminals(), &mut rng);
        assert!(tree.size() >= 1);
        assert!(tree.depth() <= 5);
    }

    #[test]
    fn test_subtree_crossover_produces_valid_trees() {
        let mut rng = rand::thread_rng();
        let t1 = GPTree::random(4, &functions(), &terminals(), &mut rng);
        let t2 = GPTree::random(4, &functions(), &terminals(), &mut rng);
        let (c1, c2) = GPTree::subtree_crossover(&t1, &t2, &mut rng);
        assert!(c1.size() >= 1);
        assert!(c2.size() >= 1);
    }

    #[test]
    fn test_subtree_mutation_changes_tree() {
        let mut rng = rand::thread_rng();
        let mut tree = GPTree::random(4, &functions(), &terminals(), &mut rng);
        let _original_size = tree.size();
        tree.subtree_mutation(3, &functions(), &terminals(), &mut rng);
        assert!(tree.size() >= 1);
    }

    #[test]
    fn test_gp_serde_roundtrip() {
        let tree = GPTree::Function {
            name: "+".to_string(),
            children: vec![
                GPTree::Terminal(GPTerminal::Const(1.0)),
                GPTree::Terminal(GPTerminal::Var("x".to_string())),
            ],
        };
        let json = serde_json::to_string(&tree).unwrap();
        let deserialized: GPTree = serde_json::from_str(&json).unwrap();
        let mut vars = std::collections::HashMap::new();
        vars.insert("x".to_string(), 5.0);
        assert!((deserialized.eval(&vars) - 6.0).abs() < 1e-10);
    }
}
