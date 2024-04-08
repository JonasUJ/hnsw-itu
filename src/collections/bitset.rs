use crate::{Idx, Reset, Set};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BitSet {
    bits: Vec<usize>,
}

impl BitSet {
    pub fn new(n: usize) -> Self {
        Self {
            bits: vec![0; n.div_ceil(std::mem::size_of::<usize>())],
        }
    }
}

impl Set<Idx> for BitSet {
    fn insert(&mut self, t: Idx) {
        let s = std::mem::size_of::<Idx>();
        self.bits[t / s] |= 1 << (t % s);
    }

    fn contains(&self, t: Idx) -> bool {
        let s = std::mem::size_of::<Idx>();
        self.bits[t / s] & (1 << (t % s)) != 0
    }

    fn len(&self) -> usize {
        self.bits
            .iter()
            .fold(0, |a, &i| a + i.count_ones() as usize)
    }
}

impl Reset for BitSet {
    fn reset(&mut self) {
        self.bits = self.bits.iter().map(|_| 0).collect();
    }
}

impl Clone for BitSet {
    fn clone(&self) -> Self {
        Self {
            bits: self.bits.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitset() {
        let mut set = BitSet::new(100);
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
    }
}
