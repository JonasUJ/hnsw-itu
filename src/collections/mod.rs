pub mod bitset;
pub mod simplegraph;

use std::collections::BinaryHeap;

pub use crate::bitset::*;
pub use crate::simplegraph::*;

pub type Idx = u32;

pub trait Graph<T> {
    fn add(&mut self, t: T) -> Idx;

    fn get(&self, v: Idx) -> Option<&T>;

    fn add_edge(&mut self, v: Idx, w: Idx);

    fn remove_edge(&mut self, v: Idx, w: Idx);

    fn neighborhood(&self, v: Idx) -> impl Iterator<Item = Idx>;

    fn size(&self) -> usize;

    fn is_connected(&self, v: Idx, w: Idx) -> bool {
        self.neighborhood(v).any(|i| i == w)
    }

    fn degree(&self, v: Idx) -> usize {
        self.neighborhood(v).count()
    }

    fn clear_edges(&mut self, v: Idx) {
        let neighbors = self.neighborhood(v).collect::<Vec<_>>();
        for w in neighbors {
            self.remove_edge(v, w);
        }
    }

    fn add_edges(&mut self, edges: impl Iterator<Item = (Idx, Idx)>) {
        for (v, w) in edges {
            self.add_edge(v, w);
        }
    }

    fn add_neighbors(&mut self, v: Idx, neighbors: impl Iterator<Item = Idx>) {
        self.add_edges(neighbors.map(|w| (v, w)));
    }
}

pub trait MinK: Iterator {
    fn min_k(mut self, k: usize) -> Vec<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        if k == 0 {
            return vec![];
        }

        let iter = self.by_ref();
        let mut heap: BinaryHeap<Self::Item> = iter.take(k).collect();

        for i in iter {
            let mut top = heap
                .peek_mut()
                .expect("k is greater than 0 but heap was emptied");

            if top.gt(&i) {
                *top = i;
            }
        }

        heap.into_vec()
    }
}

impl<T> MinK for T where T: Iterator {}

pub trait Set<T>
where
    T: Into<usize> + Clone,
{
    fn insert(&mut self, t: T);

    fn contains(&self, t: T) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_k() {
        let mut v = vec![0, 9, 1, 8, 2, 7, 3, 6, 4, 5, 5, 4, 6, 3, 7, 2, 8, 1, 9, 0];
        v = v.into_iter().min_k(5);
        v.sort();
        assert_eq!(v, vec![0, 0, 1, 1, 2]);
    }
}
