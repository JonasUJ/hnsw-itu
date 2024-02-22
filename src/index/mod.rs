pub mod bruteforce;
use std::{cmp::Ordering, marker::Sync};

pub use bruteforce::Bruteforce;
use rayon::iter::{IntoParallelIterator, ParallelIterator as _};

pub trait Index<P> {
    fn add(&mut self, sketch: P);
    fn size(&self) -> usize;
    fn search<'a>(&'a self, query: &P, ef: usize) -> Vec<Distance<'a, P>>
    where
        P: Point;

    fn knns<I>(&self, queries: I, ef: usize) -> Vec<Vec<Distance<'_, P>>>
    where
        Self: Sync,
        I: IntoIterator<Item = P>,
        P: Point + Sync,
    {
        queries
            .into_iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|q| self.search(q, ef))
            .collect()
    }
}

pub trait Point {
    fn distance(&self, other: &Self) -> usize;
}

#[derive(Debug)]
pub struct Distance<'a, P> {
    distance: usize,
    key: usize,
    point: &'a P,
}

impl<'a, P> Distance<'a, P> {
    pub const fn new(distance: usize, key: usize, point: &'a P) -> Self {
        Self {
            distance,
            key,
            point,
        }
    }

    pub const fn distance(&self) -> usize {
        self.distance
    }

    pub const fn key(&self) -> usize {
        self.key
    }

    pub const fn point(&self) -> &'a P {
        self.point
    }
}

impl<'a, P> PartialEq for Distance<'a, P> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<'a, P> PartialOrd for Distance<'a, P> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, P> Eq for Distance<'a, P> {}

impl<'a, P> Ord for Distance<'a, P> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.distance.cmp(&other.distance) {
            Ordering::Equal => self.key.cmp(&other.key),
            ordering => ordering,
        }
    }
}
