use std::collections::HashSet;

use crate::{Graph, Idx};

struct SimpleGraph<T> {
    nodes: Vec<T>,
    adj_lists: Vec<HashSet<Idx>>,
    empty: HashSet<Idx>,
}

impl<T> SimpleGraph<T> {
    pub fn new() -> Self {
        Default::default()
    }

    fn connect_directed(&mut self, src: Idx, target: Idx) {
        if let Some(set) = self.adj_lists.get_mut(src) {
            set.insert(target);
        }
    }
}

impl<T> Default for SimpleGraph<T> {
    fn default() -> Self {
        SimpleGraph {
            nodes: Default::default(),
            adj_lists: Default::default(),
            empty: Default::default(),
        }
    }
}

impl<T> FromIterator<T> for SimpleGraph<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let nodes = iter.into_iter().collect::<Vec<T>>();
        let count = nodes.len();
        SimpleGraph {
            nodes,
            adj_lists: vec![Default::default(); count],
            empty: Default::default(),
        }
    }
}

impl<T> Graph<T> for SimpleGraph<T> {
    fn add(&mut self, t: T) -> Idx {
        let idx = self.nodes.len();
        self.nodes.push(t);
        self.adj_lists.push(HashSet::new());
        idx
    }

    fn get(&self, v: Idx) -> Option<&T> {
        self.nodes.get(v)
    }

    fn add_edge(&mut self, v: Idx, w: Idx) {
        let len = self.adj_lists.len();
        if v >= len || w >= len {
            dbg!((v, w));
            return;
        }

        self.connect_directed(v, w);
        self.connect_directed(w, v);
    }

    fn neighborhood(&self, v: Idx) -> impl Iterator<Item = &Idx> {
        if let Some(set) = self.adj_lists.get(v) {
            return set.iter();
        }

        self.empty.iter()
    }

    fn size(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    #[test]
    fn test_add() {
        let mut graph = SimpleGraph::new();
        let ten = graph.add(10);
        let two = graph.add(2);
        assert_eq!(graph.size(), 2);
        graph.add_edge(ten, two);
        assert!(graph.is_connected(two, ten));
    }

    #[test]
    fn test_is_connected() {
        let mut graph = SimpleGraph::from_iter(0..2);
        assert!(!graph.is_connected(0, 1));
        graph.add_edge(0, 1);
        assert!(graph.is_connected(0, 1));
    }

    fn unordered_eq<T, I1, I2>(a: I1, b: I2) -> bool
    where
        T: Eq + Hash,
        I1: IntoIterator<Item = T>,
        I2: IntoIterator<Item = T>,
    {
        let a: HashSet<_> = a.into_iter().collect();
        let b: HashSet<_> = b.into_iter().collect();

        a == b
    }

    #[test]
    fn test_neighborhood() {
        let mut graph = SimpleGraph::from_iter(0..10);
        for i in 1..6 {
            graph.add_edge(0, i);
        }
        dbg!(graph.neighborhood(0).collect::<Vec<_>>());
        assert!(unordered_eq(graph.neighborhood(0).map(|i| *i), 1..6));
    }
}
