use crate::{Distance, Graph, Idx, Index, IndexBuilder, MinK, Point, SimpleGraph};
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
};

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

fn insert2<P: Point>(
    graph: &mut impl Graph<P>,
    point: P,
    m: usize,
    m_max: usize,
    ef: usize,
    ep: Idx,
) {
    let point_idx = graph.add(point);

    insert(graph, point_idx, m, m_max, ef, ep)
}

fn insert<P: Point>(
    graph: &mut impl Graph<P>,
    point_idx: Idx,
    m: usize,
    m_max: usize,
    ef: usize,
    ep: Idx,
) {
    let point = graph.get(point_idx).unwrap();

    let w = search(graph, point, ef, ep)
        .into_iter()
        .map(Reverse)
        .collect::<BinaryHeap<_>>();

    let neighbors = select_neighbors(w, m)
        .into_iter()
        .map(|x| x.key())
        .collect::<Vec<usize>>();

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

pub struct NSWOptions {
    pub ef_construction: usize,
    pub connections: usize,
    pub max_connections: usize,
}

impl Default for NSWOptions {
    fn default() -> Self {
        Self {
            ef_construction: 100,
            connections: 16,
            max_connections: 32,
        }
    }
}

#[derive(Debug)]
pub struct NSWBuilder<P> {
    graph: SimpleGraph<P>,
    ep: Option<Idx>,
    ef_construction: usize,
    connections: usize,
    max_connections: usize,
}

impl<P> NSWBuilder<P> {
    pub fn new(options: NSWOptions) -> Self {
        Self {
            graph: SimpleGraph::default(),
            ep: None,
            ef_construction: options.ef_construction,
            connections: options.connections,
            max_connections: options.max_connections,
        }
    }
}

impl<P: Point> IndexBuilder<P> for NSWBuilder<P> {
    type Index = NSW<P>;

    fn add(&mut self, point: P) {
        match self.ep {
            Some(ep) => insert2(
                &mut self.graph,
                point,
                self.connections,
                self.max_connections,
                self.ef_construction,
                ep,
            ),
            None => {
                let ep = self.graph.add(point);
                self.ep = Some(ep);
                insert(
                    &mut self.graph,
                    ep,
                    self.connections,
                    self.max_connections,
                    self.ef_construction,
                    ep,
                )
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

impl<P: Point> Index<P> for NSW<P> {
    fn size(&self) -> usize {
        self.graph.size()
    }

    fn search<'a>(&'a self, query: &P, k: usize) -> Vec<Distance<'a, P>> {
        self.ep.map_or_else(Vec::default, |ep| {
            search(&self.graph, query, 100, ep).into_iter().min_k(k)
        })
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
        let mut builder = NSWBuilder::new(NSWOptions {
            ef_construction: k,
            ..NSWOptions::default()
        });

        builder.extend(1..10);

        let nsw = builder.build();
        let knns = nsw
            .search(&5, k)
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
        let expected = vec![7, 16];

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
