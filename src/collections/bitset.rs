use crate::Set;

#[derive(Debug)]
pub struct BitSet {
    bits: Vec<u64>,
}

impl BitSet {
    pub fn new(n: usize) -> Self {
        Self {
            bits: vec![0; n.div_ceil(64)],
        }
    }
}

impl<T> Set<T> for BitSet
where
    T: Into<u64> + Clone,
{
    fn insert(&mut self, t: T) {
        self.bits[(Into::<u64>::into(t.clone()) / 64u64) as usize] |=
            1 << (Into::<u64>::into(t.clone()) % 64u64);
    }

    fn contains(&self, t: T) -> bool {
        self.bits[(Into::<u64>::into(t.clone()) / 64u64) as usize]
            & (1 << (Into::<u64>::into(t.clone()) % 64u64))
            != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitset() {
        let mut set = BitSet::new(100);
        set.insert(3u64);
        set.insert(93u64);
        assert!(set.contains(3u64));
        assert!(set.contains(93u64));
        assert!(!set.contains(67u64));
        assert!(!set.contains(29u64));
        assert!(!set.contains(30u64));
    }
}
