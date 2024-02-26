use crate::{Distance, MinK, Point};

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
    fn add(&mut self, point: P) {
        self.points.push(point);
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

impl<P> Extend<P> for Bruteforce<P> {
    fn extend<T: IntoIterator<Item = P>>(&mut self, iter: T) {
        for i in iter {
            self.add(i);
        }
    }
}
