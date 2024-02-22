pub mod nsw;

pub use crate::nsw::*;

pub trait KNNS<T> {
    fn search(&self, q: T, k: usize) -> impl Iterator<Item = &T>;
    fn insert(&mut self, p: T, f: i32, w: i32);
}
