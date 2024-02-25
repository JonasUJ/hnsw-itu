use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
};

use crate::{Graph, Idx, MinK, Point, SimpleGraph, KNNS};

#[derive(Debug)]
pub struct NSW<T> {
    graph: SimpleGraph<T>,
    ef: usize,
    ep: Idx,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Dist {
    dist: usize,
    idx: Idx,
}

impl<T> NSW<T> {
    pub fn new(ep: T, ef: usize) -> Self {
        let mut graph = SimpleGraph::new();
        let ep = graph.add(ep);
        Self { graph, ef, ep }
    }
}

impl<T: Point + Clone> KNNS<T> for NSW<T> {
    fn search(&self, q: &T, ep: Vec<Idx>, k: usize) -> impl Iterator<Item = Idx> {
        let dists = ep
            .into_iter()
            .map(|idx| {
                let v = self.graph.get(idx).expect("entry point was not in graph");
                Dist {
                    dist: v.distance(q),
                    idx,
                }
            })
            .collect::<Vec<_>>();

        let mut visited = dists.iter().map(|d| d.idx).collect::<HashSet<_>>();
        let iter = dists.into_iter();
        let mut w = iter.clone().collect::<BinaryHeap<_>>();
        let mut cands = iter.map(Reverse).collect::<BinaryHeap<_>>();

        while !cands.is_empty() {
            let Reverse(c) = cands.pop().expect("cands can't be empty");
            let f = w.peek().expect("w can't be empty");

            if c.dist > f.dist {
                break;
            }

            for e in self.graph.neighborhood(c.idx) {
                if visited.contains(e) {
                    continue;
                }

                visited.insert(*e);
                let f = w.peek().expect("w can't be empty");

                let point = self.graph.get(*e).unwrap();
                let e_dist = Dist {
                    dist: point.distance(q),
                    idx: *e,
                };

                if e_dist.dist >= f.dist && w.len() >= self.ef {
                    continue;
                }

                cands.push(Reverse(e_dist.clone()));
                w.push(e_dist);

                if w.len() > self.ef {
                    w.pop();
                }
            }
        }

        w.into_iter().map(|dist| dist.idx).take(k)
    }

    fn insert(&mut self, q: T) {
        let q_idx = self.graph.add(q);
        let q = self.graph.get(q_idx).unwrap().clone();
        let w = self
            .search(&q, vec![self.ep], self.ef)
            .collect::<BinaryHeap<_>>();

        for e in &w {
            self.graph.add_edge(q_idx, *e);
        }

        for e in w {
            let e_elem = self.graph.get(e).unwrap();
            let e_conn = self.graph.neighborhood(e).copied().collect::<Vec<_>>();

            if e_conn.len() <= self.ef {
                continue;
            }

            let e_new_conn = e_conn
                .into_iter()
                .map(|idx| {
                    let v = self.graph.get(idx).unwrap();
                    Dist {
                        dist: v.distance(e_elem),
                        idx,
                    }
                })
                .min_k(self.ef);

            self.graph.clear_edges(e);
            self.graph
                .add_neighbors(e, e_new_conn.into_iter().map(|dist| dist.idx));
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::unordered_eq;

    use super::*;

    impl Point for i32 {
        fn distance(&self, other: &Self) -> usize {
            (other - self).unsigned_abs() as usize
        }
    }

    #[test]
    fn test_nsw() {
        let k = 4;
        let mut nsw = NSW::new(0, k);

        //for i in vec![4, 5, 2, 8, 7, 9, 1, 3, 6] {
        for i in 1..10 {
            nsw.insert(i);
        }

        let knns = nsw.search(&5, vec![nsw.ep], k).collect::<Vec<_>>();
        assert!(unordered_eq(knns, 3..=6));
    }
}
