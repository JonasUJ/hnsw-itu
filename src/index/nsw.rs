use object_pool::Pool;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use tracing::trace;

use crate::{BitSet, Distance, Graph, Idx, Index, IndexBuilder, MinK, Point, Reset, Set, SimpleGraph};
use std::{cmp::Reverse, collections::BinaryHeap};

fn select_neighbors<'a, P: Point>(
    mut candidates: BinaryHeap<Reverse<Distance<'a, P>>>,
    m: usize,
) -> Vec<Distance<'a, P>> {
    let mut return_list = Vec::<Distance<'a, P>>::new();

    while let Some(Reverse(e)) = candidates.pop() {
        if return_list.len() >= m {
            break;
        }

        if return_list
            .iter()
            .all(|r| e.point().distance(r.point()) > e.distance())
        {
            return_list.push(e);
        }
    }

    return_list
}

fn search_select_neighbors<P: Point>(
    graph: &impl Graph<P>,
    point: &P,
    m: usize,
    ef: usize,
    ep: Idx,
    pool: &Pool<BitSet>,
) -> Vec<usize> {
    let w = search(graph, point, ef, ep, pool)
        .into_iter()
        .map(Reverse)
        .collect::<BinaryHeap<_>>();

    select_neighbors(w, m)
        .into_iter()
        .map(|x| x.key())
        .collect()
}

fn insert_point<P: Point>(
    graph: &mut impl Graph<P>,
    point: P,
    m: usize,
    m_max: usize,
    ef: usize,
    ep: Idx,
    pool: &mut Pool<BitSet>,
) {
    let point_idx = graph.add(point);

    insert_idx(graph, point_idx, m, m_max, ef, ep, pool)
}

fn insert_idx<P: Point>(
    graph: &mut impl Graph<P>,
    point_idx: Idx,
    m: usize,
    m_max: usize,
    ef: usize,
    ep: Idx,
    pool: &mut Pool<BitSet>,
) {
    let point = graph.get(point_idx).unwrap();
    let neighbors = search_select_neighbors(graph, point, m, ef, ep, pool);

    insert_neighbors(graph, point_idx, neighbors, m_max);
}

fn insert_neighbors<P: Point>(
    graph: &mut impl Graph<P>,
    point_idx: Idx,
    neighbors: Vec<usize>,
    m_max: usize,
) {
    for e in &neighbors {
        graph.add_edge(point_idx, *e);
    }

    for e in neighbors {
        let e_elem = graph.get(e).unwrap();
        let e_conn = graph.neighborhood(e).copied().collect::<Vec<_>>();

        if e_conn.len() <= m_max {
            continue;
        }

        let candidates = e_conn
            .into_iter()
            .map(|idx| {
                let v = graph.get(idx).unwrap();
                Reverse(Distance::new(v.distance(e_elem), idx, v))
            })
            .collect::<BinaryHeap<_>>();

        let e_new_conn = select_neighbors(candidates, m_max);

        let a = e_new_conn
            .into_iter()
            .map(|dist| dist.key())
            .collect::<Vec<_>>();
        graph.clear_edges(e);
        graph.add_neighbors(e, a.into_iter());
    }
}

fn search<'a, P: Point>(
    graph: &'a impl Graph<P>,
    query: &P,
    ef: usize,
    ep: Idx,
    pool: &Pool<BitSet>,
) -> Vec<Distance<'a, P>> {
    let ep_elem = graph.get(ep).expect("entry point was not in graph");
    let dist = Distance::new(ep_elem.distance(query), ep, ep_elem);

    //let mut visited = HashSet::<Idx>::with_capacity(2048);
    //visited.insert(ep);
    let mut visited = pool.try_pull().unwrap();
    visited.reset();

    let mut w = BinaryHeap::from_iter([dist.clone()]);
    let mut cands = BinaryHeap::from_iter([Reverse(dist)]);

    while !cands.is_empty() {
        let Reverse(c) = cands.pop().expect("cands can't be empty");
        let f = w.peek().expect("w can't be empty");

        if c.distance() > f.distance() {
            break;
        }

        for e in graph.neighborhood(c.key()) {
            if visited.contains(*e) {
                continue;
            }

            visited.insert(*e);
            let f = w.peek().expect("w can't be empty");

            let point = graph.get(*e).unwrap();
            let e_dist = Distance::new(point.distance(query), *e, point);

            if e_dist.distance() >= f.distance() && w.len() >= ef {
                continue;
            }

            cands.push(Reverse(e_dist.clone()));
            w.push(e_dist);

            if w.len() > ef {
                w.pop();
            }
        }
    }

    trace!(visited = visited.len(), "visited");
    w.into_iter().take(ef).collect()
}

pub struct NSWOptions {
    pub ef_construction: usize,
    pub connections: usize,
    pub max_connections: usize,
    pub size: usize,
}

impl Default for NSWOptions {
    fn default() -> Self {
        Self {
            ef_construction: 100,
            connections: 16,
            max_connections: 32,
            size: 0,
        }
    }
}

pub struct NSWBuilder<P> {
    graph: SimpleGraph<P>,
    ep: Option<Idx>,
    ef_construction: usize,
    connections: usize,
    max_connections: usize,
    visited_pool: Pool<BitSet>,
}

impl<P> NSWBuilder<P> {
    pub fn new(options: NSWOptions) -> Self {
        Self {
            graph: SimpleGraph::default(),
            ep: None,
            ef_construction: options.ef_construction,
            connections: options.connections,
            max_connections: options.max_connections,
            visited_pool: Pool::new(rayon::current_num_threads(), || BitSet::new(options.size)),
        }
    }
}

impl<P: Point + Send + Sync> NSWBuilder<P> {
    pub fn extend_parallel<T: IntoIterator<Item = P>>(&mut self, iter: T) {
        let mut iter = iter.into_iter();

        if self.ep.is_none() {
            if let Some(point) = iter.next() {
                self.add(point);
            }
        }

        // There needs to be some amount of nodes already to not generate a truly horrible graph.
        self.extend(iter.by_ref().take(0.max(50_000 - self.graph.size())));

        let chunk_size = rayon::current_num_threads();

        loop {
            let chunk = iter.by_ref().take(chunk_size).collect::<Vec<_>>();

            if chunk.is_empty() {
                break;
            }

            for (point_idx, neighbors) in chunk
                .into_iter()
                .map(|point| self.graph.add(point))
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|point_idx| {
                    let point = self.graph.get(point_idx).unwrap();

                    let neighbors = search_select_neighbors(
                        &self.graph,
                        point,
                        self.connections,
                        self.ef_construction,
                        self.ep.unwrap(),
                        &self.visited_pool,
                    );

                    (point_idx, neighbors)
                })
                .collect::<Vec<_>>()
            {
                insert_neighbors(&mut self.graph, point_idx, neighbors, self.max_connections);
            }
        }
    }
}

impl<P: Point> Extend<P> for NSWBuilder<P> {
    fn extend<T: IntoIterator<Item = P>>(&mut self, iter: T) {
        for i in iter {
            self.add(i);
        }
    }
}

impl<P: Point> IndexBuilder<P, NSW<P>> for NSWBuilder<P> {
    type Index = NSWIndex<P>;

    fn add(&mut self, point: P) {
        match self.ep {
            Some(ep) => insert_point(
                &mut self.graph,
                point,
                self.connections,
                self.max_connections,
                self.ef_construction,
                ep,
                &mut self.visited_pool,
            ),
            None => {
                let ep = self.graph.add(point);
                self.ep = Some(ep);
                insert_idx(
                    &mut self.graph,
                    ep,
                    self.connections,
                    self.max_connections,
                    self.ef_construction,
                    ep,
                    &mut self.visited_pool,
                )
            }
        };
    }

    fn build(self) -> Self::Index {
        NSWIndex {
            graph: self.graph,
            ep: self.ep,
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NSWIndex<P> {
    graph: SimpleGraph<P>,
    ep: Option<Idx>,
}

impl<P> NSWIndex<P> {
    pub fn size(&self) -> usize {
        self.graph.size()
    }
}

impl<P> From<NSWIndex<P>> for NSW<P> {
    fn from(value: NSWIndex<P>) -> Self {
        let size = value.graph.size();
        NSW {
            graph: value.graph,
            ep: value.ep,
            visited_pool: Pool::new(rayon::current_num_threads(), || BitSet::new(size)),
        }
    }
}

pub struct NSW<P> {
    graph: SimpleGraph<P>,
    ep: Option<Idx>,
    visited_pool: Pool<BitSet>,
}

impl<P> Index<P> for NSW<P> {
    fn size(&self) -> usize {
        self.graph.size()
    }

    fn search<'a>(&'a self, query: &P, k: usize, ef: usize) -> Vec<Distance<'a, P>>
    where
        P: Point,
    {
        self.ep.map_or_else(Vec::default, |ep| {
            search(&self.graph, query, ef, ep, &self.visited_pool)
                .into_iter()
                .min_k(k)
        })
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
        let mut builder = NSWBuilder::new(NSWOptions {
            ef_construction: k,
            ..NSWOptions::default()
        });

        builder.extend(1..10);

        let nsw = Into::<NSW<_>>::into(builder.build());
        let knns = nsw
            .search(&5, k, k)
            .into_iter()
            .map(|dist| dist.point())
            .copied();
        assert!(unordered_eq(knns, 3..=6));
    }

    #[test]
    fn test_heuristic() {
        let k = 4;
        let q = 10;
        let mut builder = NSWBuilder::new(NSWOptions {
            ef_construction: k,
            ..NSWOptions::default()
        });
        let numbers = vec![1, 5, 6, 7, 16, 18];
        let expected = [7, 16];

        builder.extend(numbers.clone());

        let heap = numbers
            .iter()
            .map(|x| Reverse(Distance::new(x.distance(&q), 0, x)))
            .collect::<BinaryHeap<_>>();

        let actual = select_neighbors(heap, 3);

        dbg!(&actual);

        assert!(unordered_eq(
            actual.iter().map(|dist| dist.point()),
            expected.iter()
        ));
    }
}
