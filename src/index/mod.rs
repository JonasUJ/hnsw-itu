pub mod bruteforce;
pub mod hnsw;
pub mod nsw;
use std::cmp::Ordering;

pub use bruteforce::*;
pub use hnsw::*;
pub use nsw::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator as _};

#[cfg(feature = "tracing")]
use tracing::{debug, instrument};

pub trait IndexBuilder<P> {
    type Index: Index<P>;

    fn add(&mut self, point: P);
    fn build(self) -> Self::Index;
}

pub trait Index<P> {
    fn size(&self) -> usize;
    fn search<'a>(&'a self, query: &P, k: usize, ef: usize) -> Vec<Distance<'a, P>>
    where
        P: Point;

    #[cfg_attr(feature = "tracing", instrument(skip(self, queries)))]
    fn knns<I>(&self, queries: I, k: usize, ef: usize) -> Vec<Vec<Distance<'_, P>>>
    where
        Self: Sync,
        I: IntoIterator<Item = P>,
        P: Point + Sync,
    {
        #[cfg(feature = "tracing")]
        debug!(threads = rayon::current_num_threads());
        queries
            .into_iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|q| self.search(q, k, ef))
            .collect()
    }
}

pub trait Point {
    fn distance(&self, other: &Self) -> usize;
}

#[derive(Debug)]
pub struct Distance<'a, P> {
    pub distance: usize,
    pub key: usize,
    pub point: &'a P,
}

impl<'a, P> Clone for Distance<'a, P> {
    fn clone(&self) -> Self {
        Self {
            distance: self.distance,
            key: self.key,
            point: self.point,
        }
    }
}

impl<'a, P> Distance<'a, P> {
    pub const fn new(distance: usize, key: usize, point: &'a P) -> Self {
        Self {
            distance,
            key,
            point,
        }
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
