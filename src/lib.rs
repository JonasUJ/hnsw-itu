mod algs;
mod collections;
mod index;

pub use crate::algs::*;
pub use crate::collections::*;
pub use crate::index::*;

#[cfg(test)]
mod test_utils {
    use std::{collections::HashSet, hash::Hash};

    pub fn unordered_eq<T, I1, I2>(a: I1, b: I2) -> bool
    where
        T: Eq + Hash,
        I1: IntoIterator<Item = T>,
        I2: IntoIterator<Item = T>,
    {
        let a: HashSet<_> = a.into_iter().collect();
        let b: HashSet<_> = b.into_iter().collect();

        a == b
    }
}
