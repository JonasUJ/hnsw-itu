pub mod bruteforce;
pub use bruteforce::Bruteforce;
use rayon::iter::{IntoParallelIterator, ParallelIterator as _};

use crate::{Distance, Sketch};

pub trait Index {
    fn add(&mut self, sketch: Sketch);
    fn search<'a>(&'a self, query: &Sketch, ef: usize) -> Vec<Distance<'a>>;
    fn size(&self) -> usize;

    fn knns<'a>(&'a self, queries: Vec<&Sketch>, ef: usize) -> Vec<Vec<Distance<'a>>>
    where
        Self: std::marker::Sync,
    {
        queries
            .into_par_iter()
            .map(|q| self.search(q, ef))
            .collect()
    }
}
