pub mod simplegraph;

pub use crate::simplegraph::*;

pub type Idx = usize;

pub trait Graph<T> {
    fn add(&mut self, t: T) -> Idx;
    fn get(&self, v: Idx) -> Option<&T>;
    fn add_edge(&mut self, v: Idx, w: Idx);
    fn neighborhood(&self, v: Idx) -> impl Iterator<Item = &Idx>;
    fn size(&self) -> usize;
    fn is_connected(&self, v: Idx, w: Idx) -> bool {
        self.neighborhood(v).position(|&i| i == w).is_some()
    }
    fn degree(&self, v: Idx) -> usize {
        self.neighborhood(v).count()
    }
}
