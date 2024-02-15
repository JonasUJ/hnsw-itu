pub mod bruteforce;
pub use bruteforce::Bruteforce;

use crate::{Distance, Sketch};

pub trait Index {
    fn add(&mut self, sketch: Sketch);
    fn search<'a>(&'a self, query: &Sketch, ef: usize) -> Vec<Distance<'a>>;
}
