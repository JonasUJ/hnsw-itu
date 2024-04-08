use crate::{Reset, Set};

type ValType = u16;

#[derive(Debug)]
pub struct GenerationSet {
    vals: Vec<ValType>,
    generation: ValType,
}

impl GenerationSet {
    pub fn new(n: usize) -> Self {
        Self {
            vals: vec![0; n],
            generation: 1,
        }
    }
}

impl<T> Set<T> for GenerationSet
where
    T: Into<usize> + Clone,
{
    fn insert(&mut self, t: T) {
        self.vals[Into::<usize>::into(t.clone())] = self.generation
    }

    fn contains(&self, t: T) -> bool {
        self.vals[Into::<usize>::into(t.clone())] == self.generation
    }
}

impl Reset for GenerationSet {
    fn reset(&mut self) {
        match self.generation.checked_add(1) {
            Some(v) => self.generation = v,
            None => {
                self.generation = 1;
                // Reset in order to ensure set is always correct
                // NOTE: Maybe "Approximate" Nearest Neighbor makes this unneccesary...
                // Anyway, we only run 10_000 queries and this supports ~65_000 generations,
                // so will not happen for SISAP
                // self.vals = self.vals.iter().map(|_| 0).collect();
            }
        }
    }
}

impl Clone for GenerationSet {
    fn clone(&self) -> Self {
        Self {
            vals: self.vals.clone(),
            generation: self.generation.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generationset() {
        let mut set = GenerationSet::new(100);
        set.insert(3usize);
        set.insert(93usize);
        assert!(set.contains(3usize));
        assert!(set.contains(93usize));
        assert!(!set.contains(67usize));
        assert!(!set.contains(29usize));
        assert!(!set.contains(30usize));
        set.reset();
        assert!(!set.contains(3usize));
        assert!(!set.contains(93usize));
        assert!(!set.contains(67usize));
        assert!(!set.contains(29usize));
        assert!(!set.contains(30usize));
    }
}
