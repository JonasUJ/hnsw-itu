use std::collections::HashSet;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{Graph, Idx};

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimpleGraph<T> {
    nodes: Vec<T>,
    adj_lists: Vec<HashSet<Idx>>,
    empty: HashSet<Idx>,
}

impl<T> SimpleGraph<T> {
    pub fn new() -> Self {
        Self::default()
    }

    fn is_in_bounds(&self, v: Idx, w: Idx) -> bool {
        let len = self.adj_lists.len();
        v < len && w < len
    }

    fn connect_directed(&mut self, src: Idx, target: Idx) {
        if let Some(set) = self.adj_lists.get_mut(src) {
            set.insert(target);
        }
    }

    fn disconnect_directed(&mut self, src: Idx, target: Idx) {
        if let Some(set) = self.adj_lists.get_mut(src) {
            set.remove(&target);
        }
    }
}

impl<T> Default for SimpleGraph<T> {
    fn default() -> Self {
        Self {
            nodes: Vec::default(),
            adj_lists: Vec::default(),
            empty: HashSet::default(),
        }
    }
}

impl<T> FromIterator<T> for SimpleGraph<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let nodes = iter.into_iter().collect::<Vec<T>>();
        let count = nodes.len();
        Self {
            nodes,
            adj_lists: vec![HashSet::default(); count],
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
        if !self.is_in_bounds(v, w) {
            return;
        }

        self.connect_directed(v, w);
        self.connect_directed(w, v);
    }

    fn remove_edge(&mut self, v: Idx, w: Idx) {
        if !self.is_in_bounds(v, w) {
            return;
        }

        self.disconnect_directed(v, w);
        self.disconnect_directed(w, v);
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
    use crate::test_utils::unordered_eq;

    use super::*;

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

    #[test]
    fn test_neighborhood() {
        let mut graph = SimpleGraph::from_iter(0..10);
        for i in 1..6 {
            graph.add_edge(0, i);
        }
        assert!(unordered_eq(graph.neighborhood(0).copied(), 1..6));
    }

    #[test]
    fn test_clear_edges() {
        let mut graph = SimpleGraph::from_iter(0..10);
        for i in 1..6 {
            graph.add_edge(0, i);
        }
        for i in 2..6 {
            graph.add_edge(1, i);
        }
        assert!(unordered_eq(graph.neighborhood(0).copied(), 1..6));
        assert!(unordered_eq(
            graph.neighborhood(1).copied(),
            vec![0, 2, 3, 4, 5]
        ));

        graph.clear_edges(1);
        assert!(unordered_eq(graph.neighborhood(0).copied(), 2..6));
        assert!(unordered_eq(graph.neighborhood(1).copied(), vec![]));
    }
}
