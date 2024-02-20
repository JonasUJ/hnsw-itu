use std::{borrow::Cow, marker::PhantomData, path::Path};

use ndarray::{array, s, Array1, Array2};

use hdf5::{Dataset, Extents, File, H5Type, Result};

#[derive(Clone)]
pub struct BufferedDataset<'f, T, D> {
    file: Cow<'f, File>,
    dataset: Dataset,
    _phantom: PhantomData<(T, D)>,
}

impl<'f, T, D> BufferedDataset<'f, T, D> {
    pub fn open<P>(path: P, dataset: &str) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let file = File::open(path)?;
        let dataset = file.dataset(dataset)?;
        Ok(BufferedDataset {
            file: Cow::Owned(file),
            dataset,
            _phantom: PhantomData,
        })
    }

    pub fn create<P, S>(path: P, shape: S, dataset: &str) -> Result<Self>
    where
        P: AsRef<Path>,
        S: Into<Extents>,
    {
        let file = File::create(path)?;
        let dataset = file.new_dataset::<u64>().shape(shape).create(dataset)?;
        Ok(BufferedDataset {
            file: Cow::Owned(file),
            dataset,
            _phantom: PhantomData,
        })
    }

    pub fn with_file<S>(file: &'f File, shape: S, dataset: &str) -> Result<Self>
    where
        S: Into<Extents>,
    {
        let dataset = file.new_dataset::<u64>().shape(shape).create(dataset)?;
        Ok(BufferedDataset {
            file: Cow::Borrowed(file),
            dataset,
            _phantom: PhantomData,
        })
    }

    pub fn add_attr<'n, V, N>(&self, name: N, value: &V) -> Result<()>
    where
        V: H5Type,
        N: Into<&'n str>,
    {
        self.file.new_attr::<V>().create(name)?.write_scalar(value)
    }

    pub fn size(&self) -> usize {
        *self.dataset.shape().first().expect("dataset has no shape")
    }
}

impl<'f, T, D> BufferedDataset<'f, T, D>
where
    T: From<Array1<D>>,
    D: H5Type + Clone,
{
    pub fn write_row(&self, data: T, row: usize) -> Result<()>
    where
        T: Into<Array1<D>>,
    {
        let arr: Array1<D> = data.into();
        self.dataset.write_slice(arr.view(), s![row, ..])
    }
}

impl<'f, T, D> IntoIterator for BufferedDataset<'f, T, D>
where
    T: From<Array1<D>>,
    D: H5Type + Clone,
{
    type Item = T;

    type IntoIter = BufferedDatasetIter<T, D>;

    fn into_iter(self) -> Self::IntoIter {
        BufferedDatasetIter {
            dataset: self.dataset.clone(),
            buffer: ArrayIter::empty(),
            cur: 0,
            len: self.size(),
            _phantom: PhantomData,
        }
    }
}

pub struct BufferedDatasetIter<T, D> {
    dataset: Dataset,
    buffer: ArrayIter<D>,
    cur: usize,
    len: usize,
    _phantom: PhantomData<T>,
}

impl<T, D> Iterator for BufferedDatasetIter<T, D>
where
    T: From<Array1<D>>,
    D: H5Type + Clone,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        const BUFFER_SIZE: usize = 50_000;

        if self.cur == self.len {
            return None;
        }

        if let Some(arr) = self.buffer.next() {
            self.cur += 1;
            return Some(arr.into());
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
        self.buffer.next().map(Into::into)
    }
}

struct ArrayIter<D> {
    array: Array2<D>,
    cur: usize,
    len: usize,
}

impl<D> ArrayIter<D> {
    fn empty() -> Self {
        Self {
            array: array![[], []],
            cur: 0,
            len: 0,
        }
    }
}

impl<D: H5Type + Clone> Iterator for ArrayIter<D> {
    type Item = Array1<D>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur == self.len {
            return None;
        }

        self.cur += 1;
        Some(self.array.row(self.cur - 1).to_owned())
    }
}
