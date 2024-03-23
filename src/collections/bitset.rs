use crate::Set;

#[derive(Debug)]
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

impl<T> Set<T> for BitSet
where
    T: Into<usize> + Clone,
{
    fn insert(&mut self, t: T) {
        let s = std::mem::size_of::<usize>();
        self.bits[Into::<usize>::into(t.clone()) / s] |= 1 << (Into::<usize>::into(t.clone()) % s);
    }

    fn contains(&self, t: T) -> bool {
        let s = std::mem::size_of::<usize>();
        self.bits[Into::<usize>::into(t.clone()) / s] & (1 << (Into::<usize>::into(t.clone()) % s))
            != 0
    }
}

impl Clone for BitSet {
    fn clone(&self) -> Self {
        Self {
            bits: self.bits.clone()
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
    }
}
