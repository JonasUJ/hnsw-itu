use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
};

use crate::{Distance, Graph, Idx, Index, MinK, Point, SimpleGraph};

#[derive(Debug)]
pub struct NSW<P> {
    graph: SimpleGraph<P>,
    ef: usize,
    ep: Idx,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Dist {
    dist: usize,
    idx: Idx,
}

impl<P> NSW<P> {
    pub fn new(ep: P, ef: usize) -> Self {
        let mut graph = SimpleGraph::new();
        let ep = graph.add(ep);
        Self { graph, ef, ep }
    }
}

impl<P: Point + Clone> Index<P> for NSW<P> {
    fn add(&mut self, point: P) {
        let q_idx = self.graph.add(point);
        let q = self.graph.get(q_idx).unwrap().clone();
        let w = self
            .search(&q, self.ef)
            .into_iter()
            .map(|dist| dist.key())
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

    fn size(&self) -> usize {
        self.graph.size()
    }

    fn search<'a>(&'a self, query: &P, k: usize) -> Vec<Distance<'a, P>> {
        let ep_elem = self
            .graph
            .get(self.ep)
            .expect("entry point was not in graph");
        let dist = Distance::new(ep_elem.distance(query), self.ep, ep_elem);

        let mut visited: HashSet<Idx> = HashSet::from_iter([self.ep]);
        let mut w = BinaryHeap::from_iter([dist.clone()]);
        let mut cands = BinaryHeap::from_iter([Reverse(dist)]);

        while !cands.is_empty() {
            let Reverse(c) = cands.pop().expect("cands can't be empty");
            let f = w.peek().expect("w can't be empty");

            if c.distance() > f.distance() {
                break;
            }

            for e in self.graph.neighborhood(c.key()) {
                if visited.contains(e) {
                    continue;
                }

                visited.insert(*e);
                let f = w.peek().expect("w can't be empty");

                let point = self.graph.get(*e).unwrap();
                let e_dist = Distance::new(point.distance(query), *e, point);

                if e_dist.distance() >= f.distance() && w.len() >= self.ef {
                    continue;
                }

                cands.push(Reverse(e_dist.clone()));
                w.push(e_dist);

                if w.len() > self.ef {
                    w.pop();
                }
            }
        }

        w.into_iter().take(k).collect()
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
            nsw.add(i);
        }

        let knns = nsw
            .search(&5, k)
            .into_iter()
            .map(|dist| dist.point())
            .copied();
        assert!(unordered_eq(knns, 3..=6));
    }
}
