use hnsw_itu::{Distance, HNSWBuilder, Index, IndexBuilder, NSWOptions, Point};

#[derive(Clone, Debug)]
struct Point3D(i32, i32, i32);

impl Point for Point3D {
    fn distance(&self, other: &Self) -> usize {
        // Define distance as the Euclidian distance in 3D space
        ((other.0 - self.0).pow(2) + (other.1 - self.1).pow(2) + (other.2 - self.2).pow(2)) as usize
    }
}

fn main() {
    // Dataset of points
    let points = (0..10)
        .flat_map(|x| (0..10).map(move |y| (x, y)))
        .flat_map(|(x, y)| (0..10).map(move |z| Point3D(x, y, z)))
        .collect::<Vec<_>>();

    // Graph builder with construction options
    let mut builder = HNSWBuilder::new(NSWOptions {
        connections: 8,
        ef_construction: 24,
        max_connections: 32,
        size: points.len(),
    });

    // Add dataset to graph
    // The builder also allows inserting multiple points in parallel. To do so use
    // `builder.extend_parallel` insead.
    builder.extend(points);

    // Create immutable index from builder
    let index = builder.build();

    let query = Point3D(2, 4, 16);
    let k = 10; // Number of NN to find
    let ef = 20; // Beamwidth must be >= k

    // Perform query
    // The index also support performing multliple queries in parallel. To do so use `index.knns`
    // instead. This method takes an iterator of queries instead of a single query.
    let result = index.search(&query, k, ef);

    println!("Distance : Point");
    for Distance {
        distance, point, ..
    } in result
    {
        println!("{distance} : {point:?}");
    }

    // Output:
    //
    // Distance : Point
    // 49 : Point3D(2, 4, 9)
    // 50 : Point3D(1, 4, 9)
    // 50 : Point3D(2, 3, 9)
    // 50 : Point3D(2, 5, 9)
    // 50 : Point3D(3, 4, 9)
    // 51 : Point3D(1, 3, 9)
    // 51 : Point3D(1, 5, 9)
    // 51 : Point3D(3, 3, 9)
    // 51 : Point3D(3, 5, 9)
    // 53 : Point3D(0, 4, 9)
}
