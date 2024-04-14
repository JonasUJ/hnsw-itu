use std::collections::HashSet;

use crate::{Distance, Graph, Idx, Index, IndexBuilder, Point, SimpleGraph};
use min_max_heap::MinMaxHeap;
use object_pool::Pool;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "tracing")]
use tracing::trace;

pub type SetPool = Pool<HashSet<Idx>>;

// Heuristic
pub(crate) fn select_neighbors<'a, P>(
    mut candidates: MinMaxHeap<Distance<'a, P>>,
    m: usize,
    distance_fn: impl Fn(&P, &P) -> usize,
) -> Vec<Distance<'a, P>> {
    let mut return_list = Vec::<Distance<'a, P>>::new();

    while let Some(e) = candidates.pop_min() {
        if return_list.len() >= m {
            break;
        }

        if return_list
            .iter()
            .all(|r| distance_fn(e.point(), r.point()) > e.distance())
        {
            return_list.push(e);
        }
    }

    return_list
}

// Simple heuristic
//pub(crate) fn select_neighbors<'a, P>(
//    mut candidates: MinMaxHeap<Distance<'a, P>>,
//    m: usize,
//    distance_fn: impl Fn(&P, &P) -> usize,
//) -> Vec<Distance<'a, P>> {
//    candidates.drain_asc().take(m).collect()
//}

pub(crate) fn search_select_neighbors<P>(
    graph: &impl Graph<P>,
    point: &P,
    m: usize,
    ef: usize,
    ep: Idx,
    distance_fn: &impl Fn(&P, &P) -> usize,
    pool: &SetPool,
) -> Vec<Idx> {
    let w = search(graph, point, ef, ep, distance_fn, pool);

    select_neighbors(w, m, distance_fn)
        .into_iter()
        .map(|x| x.key())
        .collect()
}

pub(crate) fn insert_point<P: Point>(
    graph: &mut impl Graph<P>,
    point: P,
    m: usize,
    m_max: usize,
    ef: usize,
    ep: Idx,
    pool: &mut SetPool,
) -> Idx {
    let point_idx = graph.add(point);

    insert_idx(graph, point_idx, m, m_max, ef, ep, Point::distance, pool)
}

pub(crate) fn insert_idx<P>(
    graph: &mut impl Graph<P>,
    point_idx: Idx,
    m: usize,
    m_max: usize,
    ef: usize,
    ep: Idx,
    distance_fn: impl Fn(&P, &P) -> usize,
    pool: &SetPool,
) -> Idx {
    let point = graph
        .get(point_idx)
        .expect("insert_idx expects point_idx to be in the graph");
    let neighbors = search_select_neighbors(graph, point, m, ef, ep, &distance_fn, pool);

    insert_neighbors(graph, point_idx, &neighbors, m_max, distance_fn);

    *neighbors
        .first()
        .expect("there should at least be the element we inserted")
}

pub(crate) fn insert_neighbors<P>(
    graph: &mut impl Graph<P>,
    point_idx: Idx,
    neighbors: &Vec<Idx>,
    m_max: usize,
    distance_fn: impl Fn(&P, &P) -> usize,
) {
    for e in neighbors {
        graph.add_edge(point_idx, *e);
    }

    for &e in neighbors {
        let e_elem = graph.get(e).unwrap();
        let e_conn = graph.neighborhood(e).copied().collect::<Vec<_>>();

        if e_conn.len() <= m_max {
            continue;
        }

        let candidates = e_conn
            .into_iter()
            .map(|idx| {
                let v = graph.get(idx).unwrap();
                Distance::new(distance_fn(v, e_elem), idx, v)
            })
            .collect::<MinMaxHeap<_>>();

        let e_new_conn = select_neighbors(candidates, m_max, &distance_fn);

        let keys = e_new_conn
            .into_iter()
            .map(|dist| dist.key())
            .collect::<Vec<_>>();
        graph.clear_edges(e);
        graph.add_neighbors(e, keys.into_iter());
        graph.add_edge(point_idx, e); // TODO: Needed?
    }
}

pub(crate) fn search<'a, P, Q>(
    graph: &'a impl Graph<P>,
    query: &Q,
    ef: usize,
    ep: Idx,
    distance_fn: impl Fn(&P, &Q) -> usize,
    pool: &SetPool,
) -> MinMaxHeap<Distance<'a, P>> {
    let ep_elem = graph.get(ep).expect("entry point was not in graph");
    let dist = Distance::new(distance_fn(ep_elem, query), ep, ep_elem);

    let mut visited = pool.try_pull().unwrap();
    visited.clear();
    visited.insert(ep);
    let mut w = MinMaxHeap::from_iter([dist.clone()]);
    let mut cands = MinMaxHeap::from_iter([dist]);

    while !cands.is_empty() {
        let c = cands.pop_min().expect("cands can't be empty");
        let f = w.peek_max().expect("w can't be empty");

        if c.distance() > f.distance() {
            break;
        }

        for e in graph.neighborhood(c.key()) {
            if visited.contains(e) {
                continue;
            }

            visited.insert(*e);
            let f = w.peek_max().expect("w can't be empty");

            let point = graph.get(*e).unwrap();
            let e_dist = Distance::new(distance_fn(point, query), *e, point);

            if e_dist.distance() >= f.distance() && w.len() >= ef {
                continue;
            }

            cands.push(e_dist.clone());
            w.push(e_dist);

            if w.len() > ef {
                w.pop_max();
            }
        }
    }

    #[cfg(feature = "tracing")]
    trace!(visited = visited.len(), "visited");

    w
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
    visited_pool: SetPool,
}

impl<P> NSWBuilder<P> {
    pub fn new(options: NSWOptions) -> Self {
        Self {
            graph: SimpleGraph::default(),
            ep: None,
            ef_construction: options.ef_construction,
            connections: options.connections,
            max_connections: options.max_connections,
            visited_pool: Pool::new(rayon::current_num_threads(), || {
                HashSet::with_capacity(2000)
            }),
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

        let chunk_size = rayon::current_num_threads() * 32;

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
                        &Point::distance,
                        &self.visited_pool,
                    );

                    (point_idx, neighbors)
                })
                .collect::<Vec<_>>()
            {
                insert_neighbors(
                    &mut self.graph,
                    point_idx,
                    &neighbors,
                    self.max_connections,
                    Point::distance,
                );
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
                    Point::distance,
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
        let _size = value.graph.size();
        NSW {
            graph: value.graph,
            ep: value.ep,
            visited_pool: Pool::new(rayon::current_num_threads(), || {
                HashSet::with_capacity(2000)
            }),
        }
    }
}

pub struct NSW<P> {
    graph: SimpleGraph<P>,
    ep: Option<Idx>,
    visited_pool: SetPool,
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
            search(
                &self.graph,
                query,
                ef,
                ep,
                Point::distance,
                &self.visited_pool,
            )
            .drain_asc()
            .take(k)
            .collect()
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
        let range = 1..20;
        let mut builder = NSWBuilder::new(NSWOptions {
            ef_construction: k,
            size: range.len(),
            ..NSWOptions::default()
        });

        builder.extend(range);

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
        let numbers = vec![1, 5, 6, 7, 16, 18];
        let mut builder = NSWBuilder::new(NSWOptions {
            ef_construction: k,
            size: numbers.len(),
            ..NSWOptions::default()
        });
        let expected = [7, 16];

        builder.extend(numbers.clone());

        let heap = numbers
            .iter()
            .map(|x| Distance::new(x.distance(&q), 0, x))
            .collect::<MinMaxHeap<_>>();

        let actual = select_neighbors(heap, 3, Point::distance);

        assert!(unordered_eq(
            actual.iter().map(|dist| dist.point()),
            expected.iter()
        ));
    }
}
