//! Module that provides an n-ary heap.

/// Simple n-ary heap.
#[derive(Debug)]
pub struct NHeap<const N: usize, T: Ord> {
    data: Vec<T>,
}

impl<const N: usize, T: Ord> NHeap<N, T> {
    /// Creates a new heap with the given width.
    ///
    /// # Arguments
    ///
    /// * `width` - width of the heap.
    pub const fn new() -> Self {
        assert!(N > 0, "N must be greater than 0");

        NHeap { data: vec![] }
    }

    /// Get the width of the heap.
    pub fn width(&self) -> usize {
        N
    }

    /// Checks if the heap is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Empties the heap.
    pub fn clear(&mut self) {
        self.data.clear()
    }

    /// Returns the length of the heap.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Insert a new item into the heap.
    pub fn push(&mut self, item: T) {
        self.data.push(item);
        self.sift_up(self.data.len() - 1);
    }

    /// Remove the maximal element from the heap and return it.
    pub fn pop(&mut self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }

        let res = self.data.swap_remove(0);
        self.sift_down(0);

        Some(res)
    }

    /// Pops the top item and pushes the new one.
    pub fn poppush(&mut self, item: T) -> Option<T> {
        if self.data.is_empty() {
            self.push(item);
            return None;
        }

        let top = std::mem::replace(&mut self.data[0], item);
        self.sift_down(0);

        Some(top)
    }

    /// Peek at the top item in the heap.
    pub fn peek(&self) -> Option<&T> {
        if self.data.is_empty() {
            None
        } else {
            Some(&self.data[0])
        }
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 && self.data[i / N] < self.data[i] {
            self.data.swap(i / N, i);
            i /= N;
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        while N * i < self.len() {
            let j = N * i;

            // Find max in all of node children
            let other = self.data[j..self.len().min(j + N)]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.cmp(b))
                .map(|t| t.0);

            if other.is_none() {
                break;
            }

            let nj = j + other.unwrap();

            if self.data[i] >= self.data[nj] {
                break;
            }

            self.data.swap(i, nj);
            i = nj;
        }
    }

    fn rebuild(&mut self) {
        for i in (0..self.len() / 2 + 1).rev() {
            self.sift_down(i);
        }
    }
}

impl<const N: usize, T: Ord> IntoIterator for NHeap<N, T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<const N: usize, T: Ord> FromIterator<T> for NHeap<N, T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut heap = NHeap::new();
        heap.data = iter.into_iter().collect();
        heap.rebuild();
        heap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        NHeap::<2, u32>::new();
    }

    #[test]
    #[should_panic]
    fn new_panic() {
        NHeap::<0, u32>::new();
    }

    #[test]
    fn from_iter() {
        let heap = NHeap::<2, _>::from_iter(vec![1, 2, 3, 4, 3, 2, 1]);
        assert!(Iterator::eq(
            heap.into_iter(),
            vec![4, 3, 3, 2, 1, 2, 1].into_iter()
        ));
    }

    #[test]
    fn from_iter_empty() {
        let heap = NHeap::<6, _>::from_iter(Vec::<i32>::new());
        assert!(heap.is_empty());
    }

    #[test]
    #[should_panic]
    fn from_iter_panic() {
        NHeap::<0, u32>::from_iter(vec![]);
    }

    #[test]
    fn width() {
        assert_eq!(NHeap::<2, u32>::new().width(), 2);
    }

    #[test]
    fn is_empty() {
        let mut heap = NHeap::<2, _>::new();
        assert!(heap.is_empty());
        heap.push(1);
        assert!(!heap.is_empty());
    }

    #[test]
    fn clear() {
        let mut heap = NHeap::<2, _>::new();
        heap.push(1);
        assert!(!heap.is_empty());
        heap.clear();
        assert!(heap.is_empty());
    }

    #[test]
    fn len() {
        let mut heap = NHeap::<2, _>::new();
        assert_eq!(heap.len(), 0);
        heap.push(1);
        assert_eq!(heap.len(), 1);
        heap.push(1);
        assert_eq!(heap.len(), 2);
        heap.pop();
        assert_eq!(heap.len(), 1);
        heap.clear();
        assert_eq!(heap.len(), 0);
    }

    #[test]
    fn into_iter() {
        let heap = NHeap::<4, _>::from_iter(vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 9, 8, 7, 6, 5, 4, 3, 2, 1,
        ]);

        let mut items = vec![0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9];
        for n in heap {
            items.remove(items.iter().position(|&i| i == n).unwrap());
        }
        assert!(items.is_empty());
    }

    #[test]
    fn push() {
        let mut heap = NHeap::<4, _>::new();
        heap.push(1);
        heap.push(1);
        heap.push(1);
        heap.push(1);
        heap.push(2);
        assert_eq!(heap.len(), 5);
        assert_eq!(heap.peek(), Some(&2));
    }

    #[test]
    fn pop() {
        let mut heap = NHeap::<3, _>::from_iter(vec![1, 2, 1, 3, 1, 4, 2, 3]);
        assert_eq!(heap.pop(), Some(4));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(2));
        assert_eq!(heap.pop(), Some(2));
        assert_eq!(heap.pop(), Some(1));
        assert_eq!(heap.pop(), Some(1));
        assert_eq!(heap.pop(), Some(1));
        assert_eq!(heap.pop(), None);
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn poppush() {
        let mut heap = NHeap::<2, _>::from_iter(vec![1, 3, 4]);
        assert_eq!(heap.poppush(2), Some(4));
        assert_eq!(heap.poppush(4), Some(3));
        assert_eq!(heap.poppush(1), Some(4));
        heap.clear();
        assert_eq!(heap.poppush(1), None);
    }

    #[test]
    fn peek() {
        let mut heap =
            NHeap::<4, _>::from_iter(vec![1, 5, 3, 8, 9, 3, 6, 9, 6, 2, 0, 5, 0, 0, 0, 5, 3, 1]);
        assert_eq!(heap.peek(), Some(&9));
        heap.clear();
        assert_eq!(heap.peek(), None);
    }
}
