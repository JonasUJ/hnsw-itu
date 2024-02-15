#[derive(Debug)]
pub struct Sketch {
    data: [u64; 16],
}

impl Sketch {
    pub fn new(data: [u64; 16]) -> Self {
        Sketch { data }
    }

    pub fn distance(&self, other: &Sketch) -> usize {
        self.data
            .iter()
            .zip(other.data.iter())
            .fold(0, |acc, (lhs, rhs)| acc + (lhs ^ rhs).count_ones() as usize)
    }
}

#[derive(Debug)]
pub struct Distance<'a> {
    distance: usize,
    key: usize,
    sketch: &'a Sketch,
}

impl<'a> Distance<'a> {
    pub fn new(distance: usize, key: usize, sketch: &'a Sketch) -> Self {
        Distance {
            distance,
            key,
            sketch,
        }
    }

    pub fn distance(&self) -> usize {
        self.distance
    }

    pub fn key(&self) -> usize {
        self.key
    }

    pub fn sketch(&self) -> &'a Sketch {
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
        self.distance.cmp(&other.distance)
    }
}
