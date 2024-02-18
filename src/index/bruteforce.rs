use std::{collections::BinaryHeap, fmt::Debug};

use crate::{Distance, Sketch};

use super::Index;

pub struct Bruteforce {
    sketches: Vec<Sketch>,
}

impl Default for Bruteforce {
    fn default() -> Self {
        Self::new()
    }
}

impl Bruteforce {
    pub fn new() -> Self {
        Bruteforce { sketches: vec![] }
    }
}

impl Index for Bruteforce {
    fn add(&mut self, sketch: Sketch) {
        self.sketches.push(sketch)
    }

    fn search<'a>(&'a self, query: &Sketch, ef: usize) -> Vec<Distance<'a>> {
        self.sketches
            .iter()
            .enumerate()
            .map(|(key, sketch)| Distance::new(query.distance(sketch), key, sketch))
            .min_k(ef)
    }
}

trait MinK: Iterator {
    fn min_k(mut self, k: usize) -> Vec<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord + Debug,
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