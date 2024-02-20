use std::collections::BinaryHeap;

use crate::{Distance, Point};

use super::Index;

pub struct Bruteforce<P> {
    points: Vec<P>,
}

impl<P> Default for Bruteforce<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> Bruteforce<P> {
    pub const fn new() -> Self {
        Self { points: vec![] }
    }
}

impl<P> Index<P> for Bruteforce<P> {
    fn add(&mut self, sketch: P) {
        self.points.push(sketch);
    }

    fn search<'a>(&'a self, query: &P, ef: usize) -> Vec<Distance<'a, P>>
    where
        P: Point,
    {
        self.points
            .iter()
            .enumerate()
            .map(|(key, point)| Distance::new(query.distance(point), key, point))
            .min_k(ef)
    }

    fn size(&self) -> usize {
        self.points.len()
    }
}

impl<P> FromIterator<P> for Bruteforce<P> {
    fn from_iter<T: IntoIterator<Item = P>>(iter: T) -> Self {
        let mut this = Self::new();
        for i in iter {
            this.add(i);
        }
        this
    }
}

trait MinK: Iterator {
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
