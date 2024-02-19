use std::cmp::Ordering::Equal;

use ndarray::{arr1, Array1};

#[derive(Clone, Debug)]
pub struct Sketch {
    data: [u64; 16],
}

impl Sketch {
    pub const fn new(data: [u64; 16]) -> Self {
        Self { data }
    }

    pub fn distance(&self, other: &Self) -> usize {
        self.data
            .iter()
            .zip(other.data.iter())
            .fold(0, |acc, (lhs, rhs)| acc + (lhs ^ rhs).count_ones() as usize)
    }
}

impl From<Array1<u64>> for Sketch {
    fn from(value: Array1<u64>) -> Self {
        Self::new([
            value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
            value[8], value[9], value[10], value[11], value[12], value[13], value[14], value[15],
        ])
    }
}

// TODO: TryFrom
impl From<Sketch> for Array1<u64> {
    fn from(value: Sketch) -> Self {
        arr1(&value.data)
    }
}

#[derive(Debug)]
pub struct Distance<'a> {
    distance: usize,
    key: usize,
    sketch: &'a Sketch,
}

impl<'a> Distance<'a> {
    pub const fn new(distance: usize, key: usize, sketch: &'a Sketch) -> Self {
        Self {
            distance,
            key,
            sketch,
        }
    }

    pub const fn distance(&self) -> usize {
        self.distance
    }

    pub const fn key(&self) -> usize {
        self.key
    }

    pub const fn sketch(&self) -> &'a Sketch {
        self.sketch
    }
}

impl<'a> PartialEq for Distance<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<'a> PartialOrd for Distance<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Eq for Distance<'a> {}

impl<'a> Ord for Distance<'a> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.distance.cmp(&other.distance) {
            Equal => self.key.cmp(&other.key),
            ordering => ordering,
        }
    }
}

#[allow(unused_imports)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_distance() {
        let a = Sketch::new([0b1111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0b1001]);
        let b = Sketch::new([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0b1011]);

        assert_eq!(a.distance(&b), 5);
    }
}
