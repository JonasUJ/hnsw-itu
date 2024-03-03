use crate::{Distance, Graph, Idx, Index, IndexBuilder, MinK, Point, SimpleGraph};
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
};

fn insert2<P: Point>(graph: &mut impl Graph<P>, point: P, ef: usize, ep: Idx) {
    let point_idx = graph.add(point);

    insert(graph, point_idx, ef, ep)
}

fn insert<P: Point>(graph: &mut impl Graph<P>, point_idx: Idx, ef: usize, ep: Idx) {
    let point = graph.get(point_idx).unwrap();

    let w = search(graph, point, ef, ep)
        .into_iter()
        .map(|dist| dist.key())
        .collect::<BinaryHeap<_>>();

    for e in &w {
        graph.add_edge(point_idx, *e);
    }

    for e in w {
        let e_elem = graph.get(e).unwrap();
        let e_conn = graph.neighborhood(e).copied().collect::<Vec<_>>();

        if e_conn.len() <= ef {
            continue;
        }

        let e_new_conn = e_conn
            .into_iter()
            .map(|idx| {
                let v = graph.get(idx).unwrap();
                Dist {
                    dist: v.distance(e_elem),
                    idx,
                }
            })
            .min_k(ef);

        graph.clear_edges(e);
        graph.add_neighbors(e, e_new_conn.into_iter().map(|dist| dist.idx));
    }
}

fn search<'a, P: Point>(
    graph: &'a impl Graph<P>,
    query: &P,
    k: usize,
    ep: Idx,
) -> Vec<Distance<'a, P>> {
    let ep_elem = graph.get(ep).expect("entry point was not in graph");
    let dist = Distance::new(ep_elem.distance(query), ep, ep_elem);

    let mut visited: HashSet<Idx> = HashSet::from_iter([ep]);
    let mut w = BinaryHeap::from_iter([dist.clone()]);
    let mut cands = BinaryHeap::from_iter([Reverse(dist)]);

    while !cands.is_empty() {
        let Reverse(c) = cands.pop().expect("cands can't be empty");
        let f = w.peek().expect("w can't be empty");

        if c.distance() > f.distance() {
            break;
        }

        for e in graph.neighborhood(c.key()) {
            if visited.contains(e) {
                continue;
            }

            visited.insert(*e);
            let f = w.peek().expect("w can't be empty");

            let point = graph.get(*e).unwrap();
            let e_dist = Distance::new(point.distance(query), *e, point);

            if e_dist.distance() >= f.distance() && w.len() >= k {
                continue;
            }

            cands.push(Reverse(e_dist.clone()));
            w.push(e_dist);

            if w.len() > k {
                w.pop();
            }
        }
    }

    w.into_iter().take(k).collect()
}

#[derive(Debug)]
pub struct NSWBuilder<P> {
    graph: SimpleGraph<P>,
    ef: usize,
    ep: Option<Idx>,
}

impl<P> NSWBuilder<P> {
    pub fn new(ef: usize) -> Self {
        Self {
            graph: SimpleGraph::default(),
            ef,
            ep: None,
        }
    }
}

impl<P: Point> IndexBuilder<P> for NSWBuilder<P> {
    type Index = NSW<P>;

    fn add(&mut self, point: P) {
        match self.ep {
            Some(ep) => insert2(&mut self.graph, point, self.ef, ep),
            None => {
                let ep = self.graph.add(point);
                self.ep = Some(ep);
                insert(&mut self.graph, ep, self.ef, ep)
            }
        };
    }

    fn build(self) -> Self::Index {
        NSW {
            graph: self.graph,
            ep: self.ep,
        }
    }
}

#[derive(Debug)]
pub struct NSW<P> {
    graph: SimpleGraph<P>,
    ep: Option<Idx>,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Dist {
    dist: usize,
    idx: Idx,
}

impl<P: Point> Index<P> for NSW<P> {
    fn size(&self) -> usize {
        self.graph.size()
    }

    fn search<'a>(&'a self, query: &P, k: usize) -> Vec<Distance<'a, P>> {
        self.ep
            .map_or_else(Vec::default, |ep| search(&self.graph, query, k, ep))
    }
}

impl<P: Point> Extend<P> for NSWBuilder<P> {
    fn extend<T: IntoIterator<Item = P>>(&mut self, iter: T) {
        for i in iter {
            self.add(i);
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
        let mut builder = NSWBuilder::new(k);

        //for i in vec![4, 5, 2, 8, 7, 9, 1, 3, 6] {
        for i in 1..10 {
            builder.add(i);
        }

        let nsw = builder.build();
        let knns = nsw
            .search(&5, k)
            .into_iter()
            .map(|dist| dist.point())
            .copied();
        assert!(unordered_eq(knns, 3..=6));
    }
}
