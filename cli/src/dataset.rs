use std::path::PathBuf;

use hnsw_itu::Sketch;
use ndarray::{array, s, Array1, Array2};

use hdf5::{Dataset, File, Result};

#[derive(Clone)]
pub struct SketchDataset {
    file: File,
    dataset: Dataset,
}

impl SketchDataset {
    pub fn create(path: PathBuf, dataset: &str) -> Result<Self> {
        let file = File::open(path)?;
        let dataset = file.dataset(dataset)?;
        Ok(SketchDataset { file, dataset })
    }
}

impl IntoIterator for SketchDataset {
    type Item = Sketch;

    type IntoIter = SketchDatasetIter;

    fn into_iter(self) -> Self::IntoIter {
        SketchDatasetIter {
            dataset: self.dataset.clone(),
            buffer: ArrayIter::empty(),
            cur: 0,
            len: *self.dataset.shape().first().expect("dataset has no shape"),
        }
    }
}

pub struct SketchDatasetIter {
    dataset: Dataset,
    buffer: ArrayIter,
    cur: usize,
    len: usize,
}

impl Iterator for SketchDatasetIter {
    type Item = Sketch;

    fn next(&mut self) -> Option<Self::Item> {
        const BUFFER_SIZE: usize = 50_000;

        if self.cur == self.len {
            return None;
        }

        if let Some(arr) = self.buffer.next() {
            self.cur += 1;
            return Some(Sketch::new(ndarray_to_array(arr)));
        }

        let to = self.len.min(self.cur + BUFFER_SIZE);

        let array = self
            .dataset
            .read_slice_2d(s![self.cur..to, ..])
            .expect("could not read expected rows");

        self.buffer = ArrayIter {
            array,
            cur: 0,
            len: to - self.cur,
        };

        self.cur += 1;
        self.buffer.next().map(ndarray_to_array).map(Sketch::new)
    }
}

fn ndarray_to_array(array: Array1<u64>) -> [u64; 16] {
    [
        array[0], array[1], array[2], array[3], array[4], array[5], array[6], array[7], array[8],
        array[9], array[10], array[11], array[12], array[13], array[14], array[15],
    ]
}

struct ArrayIter {
    array: Array2<u64>,
    cur: usize,
    len: usize,
}

impl ArrayIter {
    fn empty() -> Self {
        ArrayIter {
            array: array![[], []],
            cur: 0,
            len: 0,
        }
    }
}

impl Iterator for ArrayIter {
    type Item = Array1<u64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur == self.len {
            return None;
        }

        self.cur += 1;
        Some(self.array.row(self.cur - 1).to_owned())
    }
}
