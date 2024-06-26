use hnsw_itu::Point;
use ndarray::{arr1, Array1};
use serde::{Deserialize, Serialize};
#[cfg(feature = "instrument")]
use tracing::trace;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Sketch {
    pub data: [u64; 16],
}

impl Sketch {
    pub const fn new(data: [u64; 16]) -> Self {
        Self { data }
    }
}

impl Point for Sketch {
    #[inline(always)]
    fn distance(&self, other: &Self) -> usize {
        #[cfg(feature = "instrument")]
        trace!("distance");

        self.data
            .iter()
            .zip(other.data.iter())
            .fold(0, |acc, (lhs, rhs)| acc + (lhs ^ rhs).count_ones() as usize)
    }
}

// It's just easier to panic than TryFrom
impl From<Array1<u64>> for Sketch {
    fn from(value: Array1<u64>) -> Self {
        Self::new([
            value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
            value[8], value[9], value[10], value[11], value[12], value[13], value[14], value[15],
        ])
    }
}

impl From<Sketch> for Array1<u64> {
    fn from(value: Sketch) -> Self {
        arr1(&value.data)
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
