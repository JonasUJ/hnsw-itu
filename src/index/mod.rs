pub mod bruteforce;
pub mod hnsw;
pub mod nsw;
pub use bruteforce::*;
pub use hnsw::*;
pub use nsw::*;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator as _};
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

#[cfg(feature = "tracing")]
use tracing::{debug, instrument};

pub trait IndexBuilder<P> {
    type Index: Index<P>;

    fn add(&mut self, point: P);
    fn build(self) -> Self::Index;
}

pub trait Index<P> {
    type Options<'a>: Debug + Send + Sync;

    fn size(&self) -> usize;

    fn search(&'_ self, query: &P, k: usize, options: &Self::Options<'_>) -> Vec<Distance<'_, P>>
    where
        P: Point;

    #[cfg_attr(feature = "tracing", instrument(skip(self, queries)))]
    fn knns<'q, I>(
        &'_ self,
        queries: &'q I,
        k: usize,
        options: &Self::Options<'_>,
    ) -> Vec<Vec<Distance<'_, P>>>
    where
        Self: Sync,
        I: IntoParallelRefIterator<'q, Item = &'q P>,
        P: Point + Sync,
        P: 'q,
    {
        #[cfg(feature = "tracing")]
        debug!(threads = rayon::current_num_threads());
        queries
            .par_iter()
            .map(|q| self.search(q, k, options))
            .collect()
    }
}

pub trait IndexVis<P>: Index<P> {
    fn search_vis<'a>(
        &'a self,
        query: &P,
        k: usize,
        options: &Self::Options<'_>,
        vis: &mut HashSet<Distance<'a, P>>,
    ) -> Vec<Distance<'a, P>>
    where
        P: Point;
}

pub trait Point {
    fn distance(&self, other: &Self) -> f32;
}

impl<P: Point> Point for &P {
    fn distance(&self, other: &Self) -> f32 {
        (*self).distance(other)
    }
}

#[derive(Debug)]
pub struct Distance<'a, P> {
    pub distance: f32,
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
    pub const fn new(distance: f32, key: usize, point: &'a P) -> Self {
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
        match self.distance.partial_cmp(&other.distance).unwrap() {
            Ordering::Equal => self.key.cmp(&other.key),
            ordering => ordering,
        }
    }
}

impl<'a, P> Hash for Distance<'a, P> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}
