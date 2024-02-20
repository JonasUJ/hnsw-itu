pub mod bruteforce;
pub use bruteforce::Bruteforce;
use rayon::iter::{IntoParallelIterator, ParallelIterator as _};

use crate::{Distance, Sketch};

pub trait Index {
    fn add(&mut self, sketch: Sketch);
    fn size(&self) -> usize;
    fn search<'a, Q>(&'a self, query: Q, ef: usize) -> Vec<Distance<'a>>
    where
        Q: AsRef<Sketch>;

    fn knns<'a, Q>(
        &'a self,
        queries: impl IntoIterator<Item = Q>,
        ef: usize,
    ) -> Vec<Vec<Distance<'a>>>
    where
        Self: std::marker::Sync,
        Q: AsRef<Sketch>,
    {
        let queries: Vec<_> = queries.into_iter().collect();
        let sketches: Vec<_> = queries.iter().map(|q| q.as_ref()).collect();

        sketches
            .into_par_iter()
            .map(|q| self.search(q, ef))
            .collect()
    }
}
