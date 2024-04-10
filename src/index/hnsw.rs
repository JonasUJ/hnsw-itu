use std::{collections::HashSet, fmt::Debug};

use object_pool::Pool;
use rand::{rngs::ThreadRng, thread_rng, Rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    nsw, BitSet, Distance, Graph, Idx, Index, IndexBuilder, NSWOptions, Point, SetPool, SimpleGraph,
};

pub struct HNSWBuilder<P> {
    layers: Vec<SimpleGraph<(P, Idx)>>,
    base: SimpleGraph<P>,
    ep: Option<Idx>,
    rng: ThreadRng,
    ef_construction: usize,
    connections: usize,
    max_connections: usize,
    pool: SetPool,
}

impl<P> HNSWBuilder<P> {
    pub fn new(options: NSWOptions) -> Self {
        Self {
            layers: Default::default(),
            base: Default::default(),
            ep: None,
            rng: thread_rng(),
            ef_construction: options.ef_construction,
            connections: options.connections,
            max_connections: options.max_connections,
            pool: Pool::new(rayon::current_num_threads(), || {
                HashSet::with_capacity(2000)
            }),
        }
    }

    fn random_level(&mut self) -> usize {
        let val: f32 = self.rng.gen();
        (-val.ln() * (1.0 / (self.connections as f32).ln())) as usize
    }
}

impl<P: Point + Clone + Send + Sync> HNSWBuilder<P> {
    pub fn extend_parallel<T: IntoIterator<Item = P>>(&mut self, iter: T) {
        let mut iter = iter.into_iter();

        if self.ep.is_none() {
            if let Some(point) = iter.next() {
                self.add(point);
            }
        }

        // There needs to be some amount of nodes already to not generate a truly horrible graph.
        self.extend(iter.by_ref().take(0.max(50_000 - self.base.size())));

        let chunk_size = rayon::current_num_threads() * 32;

        loop {
            let chunk = iter.by_ref().take(chunk_size).collect::<Vec<_>>();

            if chunk.is_empty() {
                break;
            }

            let level = self.random_level();

            let mut new_ep = false;
            while self.layers.len() < level {
                self.layers.push(Default::default());
                new_ep = true;
            }

            // Add points to all layers (not insert) and remember their indices for insert later
            let chunk_idxs = chunk
                .into_iter()
                .map(|point| {
                    let base_idx = self.base.add(point.clone());
                    let idxs = self.layers[..level]
                        .iter_mut()
                        .fold(vec![base_idx], |mut v, l| {
                            let idx = *v.last().unwrap();
                            v.push(l.add((point.clone(), idx)));
                            v
                        });
                    (point, idxs)
                })
                .collect::<Vec<_>>();

            if new_ep {
                let idx = *chunk_idxs.first().unwrap().1.last().unwrap();
                self.ep = Some(idx);
            }

            let chunk_idxs = chunk_idxs
                .into_par_iter()
                .map(|(point, idxs)| {
                    let mut ep = self.ep.unwrap();

                    // Search until layer where we want to start inserting
                    for l in (level..self.layers.len()).rev() {
                        let layer = &self.layers[l];
                        let w = nsw::search(
                            layer,
                            &point,
                            1,
                            ep,
                            |(p, _), q| p.distance(q),
                            &self.pool,
                        );
                        ep = w.peek_min().unwrap().point().1;
                    }

                    (point, idxs, ep)
                })
                .collect::<Vec<_>>();

            // Insert in all layers below here
            for l in (0..level).rev() {
                let chunk_neighbors = chunk_idxs
                    .clone()
                    .into_par_iter()
                    .map(|(point, idxs, ep)| {
                        let neighbors = nsw::search_select_neighbors(
                            &self.layers[l],
                            // Idx can be default because it's unused in distance_fn
                            &(point, Idx::default()),
                            self.connections,
                            self.ef_construction,
                            ep,
                            &|(p, _), (q, _)| p.distance(q),
                            &self.pool,
                        );

                        (neighbors, idxs)
                    })
                    .collect::<Vec<_>>();

                for (neighbors, idxs) in chunk_neighbors {
                    nsw::insert_neighbors(
                        &mut self.layers[l],
                        idxs[l + 1],
                        &neighbors,
                        self.max_connections,
                        |(p, _), (q, _)| p.distance(q),
                    );
                }
            }

            // Search base layer
            let chunk_neighbors = chunk_idxs
                .into_par_iter()
                .map(|(point, idxs, ep)| {
                    let neighbors = nsw::search_select_neighbors(
                        &self.base,
                        &point,
                        self.connections,
                        self.ef_construction,
                        ep,
                        &Point::distance,
                        &self.pool,
                    );

                    (neighbors, idxs[0])
                })
                .collect::<Vec<_>>();

            // Insert in base layer
            for (neighbors, idx) in chunk_neighbors {
                nsw::insert_neighbors(
                    &mut self.base,
                    idx,
                    &neighbors,
                    self.max_connections,
                    Point::distance,
                );
            }
        }
    }
}

impl<P: Point + Clone> Extend<P> for HNSWBuilder<P> {
    fn extend<T: IntoIterator<Item = P>>(&mut self, iter: T) {
        for i in iter {
            self.add(i);
        }
    }
}

impl<P: Point + Clone> IndexBuilder<P, HNSW<P>> for HNSWBuilder<P> {
    type Index = HNSWIndex<P>;

    fn add(&mut self, point: P) {
        let base_idx = self.base.add(point.clone());
        let level = if self.ep.is_some() {
            self.random_level()
        } else {
            self.ep = Some(base_idx);
            self.layers.len()
        };

        // Add new layers and update entry point if required
        let mut new_ep = false;
        while self.layers.len() < level {
            self.layers.push(Default::default());
            new_ep = true;
        }

        let idxs = &self.layers[..level]
            .iter_mut()
            .fold(vec![base_idx], |mut v, l| {
                let idx = *v.last().unwrap();
                v.push(l.add((point.clone(), idx)));
                v
            })[1..];

        if new_ep {
            let idx = *idxs.last().unwrap();
            self.ep = Some(idx);
        }

        let mut ep = self.ep.unwrap();

        // Search until layer where we want to start inserting
        for l in (level..self.layers.len()).rev() {
            let layer = &self.layers[l];
            let w = nsw::search(layer, &point, 1, ep, |(p, _), q| p.distance(q), &self.pool);
            ep = w.peek_min().unwrap().point().1;
        }

        // Insert in all layers below here
        for (layer, &idx) in self.layers[..level].iter_mut().zip(idxs).rev() {
            ep = nsw::insert_idx(
                layer,
                idx,
                self.connections,
                self.max_connections,
                self.ef_construction,
                ep,
                |(p, _), (q, _)| p.distance(q),
                &self.pool,
            );
        }

        // Insert in base layer
        nsw::insert_idx(
            &mut self.base,
            base_idx,
            self.connections,
            self.max_connections,
            self.ef_construction,
            ep,
            Point::distance,
            &self.pool,
        );
    }

    fn build(self) -> Self::Index {
        HNSWIndex {
            layers: self.layers,
            base: self.base,
            ep: self.ep,
        }
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HNSWIndex<P> {
    layers: Vec<SimpleGraph<(P, Idx)>>,
    base: SimpleGraph<P>,
    ep: Option<Idx>,
}

impl<P> HNSWIndex<P> {
    pub fn size(&self) -> usize {
        self.base.size()
    }
}

impl<P> From<HNSWIndex<P>> for HNSW<P> {
    fn from(value: HNSWIndex<P>) -> Self {
        let size = value.base.size();
        Self {
            layers: value.layers,
            base: value.base,
            ep: value.ep,
            pool: Pool::new(rayon::current_num_threads(), || {
                HashSet::with_capacity(2000)
            }),
        }
    }
}

pub struct HNSW<P> {
    layers: Vec<SimpleGraph<(P, Idx)>>,
    base: SimpleGraph<P>,
    ep: Option<Idx>,
    pool: SetPool,
}

impl<P> Index<P> for HNSW<P> {
    fn size(&self) -> usize {
        self.base.size()
    }

    fn search<'a>(&'a self, query: &P, k: usize, ef: usize) -> Vec<Distance<'a, P>>
    where
        P: Point,
    {
        let Some(mut ep) = self.ep else { return vec![] };

        // Search layers from top to bottom
        for layer in self.layers.iter().rev() {
            let mut w = nsw::search(layer, query, 1, ep, |(p, _), q| p.distance(q), &self.pool);

            ep = w
                .pop_min()
                .expect("search must find something when graph is not empty")
                .point()
                .1;
        }

        // Search base layer last
        nsw::search(&self.base, query, ef, ep, Point::distance, &self.pool)
            .drain_asc()
            .take(k)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::unordered_eq;
    use min_max_heap::MinMaxHeap;

    #[test]
    fn test_hnsw() {
        let k = 4;
        let range = 0..20;
        let mut builder = HNSWBuilder::new(NSWOptions {
            ef_construction: k,
            connections: 3,
            size: range.len(),
            ..NSWOptions::default()
        });

        builder.extend(range.clone());

        let hnsw = Into::<HNSW<_>>::into(builder.build());
        let knns = hnsw
            .search(&5, k, k)
            .into_iter()
            .map(|dist| dist.point())
            .copied();
        assert!(unordered_eq(knns.clone(), 3..=6) || unordered_eq(knns.clone(), 4..=7));

        let len = hnsw.search(&0, hnsw.size(), hnsw.size()).len();
        assert_eq!(hnsw.size(), len);
    }

    #[test]
    fn test_heuristic() {
        let k = 4;
        let q = 10;
        let numbers = vec![1, 5, 6, 7, 16, 18];
        let mut builder = HNSWBuilder::new(NSWOptions {
            ef_construction: k,
            connections: 3,
            size: numbers.len(),
            ..NSWOptions::default()
        });
        let expected = [7, 16];

        builder.extend(numbers.clone());

        let heap = numbers
            .iter()
            .map(|x| Distance::new(x.distance(&q), 0, x))
            .collect::<MinMaxHeap<_>>();

        let actual = nsw::select_neighbors(heap, 3, Point::distance);

        assert!(unordered_eq(
            actual.iter().map(|dist| dist.point()),
            expected.iter()
        ));
    }
}
