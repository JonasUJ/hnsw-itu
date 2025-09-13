use hnsw_itu::Point;
use ndarray::{arr1, Array1};
use serde::{Deserialize, Serialize};
#[cfg(feature = "instrument")]
use tracing::trace;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Sketch {
    pub data: Array1<f32>,
}

impl Sketch {
    pub const fn new(data: Array1<f32>) -> Self {
        Self { data }
    }
}

impl Point for Sketch {
    #[inline(always)]
    fn distance(&self, other: &Self) -> f32 {
        #[cfg(feature = "instrument")]
        trace!("distance");

        self.data.dot(&other.data)
    }
}

// It's just easier to panic than TryFrom
impl From<Array1<f32>> for Sketch {
    fn from(value: Array1<f32>) -> Self {
        Self::new(value)
    }
}

impl From<Sketch> for Array1<f32> {
    fn from(value: Sketch) -> Self {
        value.data
    }
}

#[allow(unused_imports)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_distance() {
        //let a = Sketch::new([0b1111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0b1001]);
        //let b = Sketch::new([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0b1011]);

        //assert_eq!(a.distance(&b), 5);
    }
}
