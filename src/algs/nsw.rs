use crate::SimpleGraph;

pub struct NSW<T> {
    graph: SimpleGraph<T>,
    ef: usize,
    ep: &T,
}

impl<T> KNNS<T> for NSW<T> {
    fn search(&self, q: T, k: usize) -> impl Iterator<Item = &T> {
        let mut visited = 
    }
}
