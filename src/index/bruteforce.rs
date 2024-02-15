use crate::{collections::NHeap, Distance, Sketch};

use super::Index;

pub struct Bruteforce {
    sketches: Vec<Sketch>,
}

impl Default for Bruteforce {
    fn default() -> Self {
        Self::new()
    }
}

impl Bruteforce {
    pub fn new() -> Self {
        Bruteforce { sketches: vec![] }
    }
}

impl Index for Bruteforce {
    fn add(&mut self, sketch: Sketch) {
        self.sketches.push(sketch)
    }

    fn search<'a>(&'a self, query: &Sketch, ef: usize) -> Vec<Distance<'a>> {
        if ef == 0 {
            return vec![];
        }

        let mut iter = self.sketches.iter().enumerate();
        let mut heap = iter
            .by_ref()
            .take(ef)
            .map(|(k, s)| Distance::new(query.distance(s), k, s))
            .collect::<NHeap<2, _>>();

        for (key, sketch) in iter {
            let distance = query.distance(sketch);
            let Some(farthest) = heap.peek() else {
                panic!("ef is greater than 0 but heap was emptied")
            };

            if distance < farthest.distance() {
                heap.poppush(Distance::new(distance, key, sketch));
            }
        }

        heap.into()
    }
}
