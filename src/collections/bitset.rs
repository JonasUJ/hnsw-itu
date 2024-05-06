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
        let t = t as usize;
        let s = std::mem::size_of::<Idx>();
        self.bits[t / s] |= 1 << (t % s);
    }

    fn contains(&self, t: Idx) -> bool {
        let t = t as usize;
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
        set.insert(3);
        set.insert(93);
        assert!(set.contains(3));
        assert!(set.contains(93));
        assert!(!set.contains(67));
        assert!(!set.contains(29));
        assert!(!set.contains(30));

        set.reset();

        assert!(!set.contains(3));
        assert!(!set.contains(93));
    }
}
