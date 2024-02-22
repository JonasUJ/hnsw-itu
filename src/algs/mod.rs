pub mod nsw;

pub use crate::nsw::*;
use crate::Idx;

pub trait KNNS<T> {
    fn search(&self, q: &T, ep: Vec<Idx>, k: usize) -> impl Iterator<Item = Idx>;
    fn insert(&mut self, q: T);
}
