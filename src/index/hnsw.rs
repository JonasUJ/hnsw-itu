use rand::{rngs::ThreadRng, thread_rng, Rng};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{nsw, Distance, Graph, Idx, Index, IndexBuilder, NSWOptions, Point, SimpleGraph};

pub struct HNSWBuilder<P> {
    layers: Vec<SimpleGraph<(P, Idx)>>,
    base: SimpleGraph<P>,
    ep: Option<Idx>,
    rng: ThreadRng,
    ef_construction: usize,
    connections: usize,
    max_connections: usize,
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
        }
    }

    fn random_level(&mut self) -> usize {
        let val: f32 = self.rng.gen();
        (-val.ln() * (1.0 / (self.connections as f32).ln())) as usize
    }
}

impl<P: Point + Clone> Extend<P> for HNSWBuilder<P> {
    fn extend<T: IntoIterator<Item = P>>(&mut self, iter: T) {
        for i in iter {
            self.add(i);
        }
    }
}

impl<P: Point + Clone> IndexBuilder<P> for HNSWBuilder<P> {
    type Index = HNSW<P>;

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
            let w = nsw::search(layer, &point, 1, ep, |(p, _), q| p.distance(q));
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
        );
    }

    fn build(self) -> Self::Index {
        HNSW {
            layers: self.layers,
            base: self.base,
            ep: self.ep,
        }
    }
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HNSW<P> {
    layers: Vec<SimpleGraph<(P, Idx)>>,
    base: SimpleGraph<P>,
    ep: Option<Idx>,
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
            let mut w = nsw::search(layer, query, 1, ep, |(p, _), q| p.distance(q));

            ep = w
                .pop_min()
                .expect("search must find something when graph is not empty")
                .point()
                .1;
        }

        // Search base layer last
        nsw::search(&self.base, query, ef, ep, Point::distance)
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
        let range = 0..2000;
        let mut builder = HNSWBuilder::new(NSWOptions {
            ef_construction: k,
            connections: 3,
            ..NSWOptions::default()
        });

        builder.extend(range.clone());

        let hnsw = builder.build();
        let knns = hnsw
            .search(&5, k, k)
            .into_iter()
            .map(|dist| dist.point())
            .copied();
        assert!(unordered_eq(knns, 3..=6));

        let len = hnsw.search(&0, hnsw.size(), hnsw.size()).len();
        assert_eq!(hnsw.size(), len);
    }

    #[test]
    fn test_heuristic() {
        let k = 4;
        let q = 10;
        let mut builder = HNSWBuilder::new(NSWOptions {
            ef_construction: k,
            connections: 3,
            ..NSWOptions::default()
        });
        let numbers = vec![1, 5, 6, 7, 16, 18];
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
