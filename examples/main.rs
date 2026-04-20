use hnsw_itu::{Distance, HNSWBuilder, Index, IndexBuilder, NSWOptions, Point};

#[derive(Clone, Debug)]
struct Point3D(f32, f32, f32);

impl Point for Point3D {
    fn distance(&self, other: &Self) -> f32 {
        // Define distance as the Euclidean distance in 3D space
        (other.0 - self.0).powf(2.0) + (other.1 - self.1).powf(2.0) + (other.2 - self.2).powf(2.0)
    }
}

fn main() {
    // Dataset of points
    let points = (0..10)
        .flat_map(|x| (0..10).map(move |y| (x, y)))
        .flat_map(|(x, y)| (0..10).map(move |z| Point3D(x as f32, y as f32, z as f32)))
        .collect::<Vec<_>>();

    // Graph builder with construction options
    let mut builder = HNSWBuilder::new(NSWOptions {
        connections: 8,
        ef_construction: 24,
        max_connections: 32,
    });

    // Add dataset to graph
    // The builder also allows inserting multiple points in parallel. To do so use
    // `builder.extend_parallel` instead.
    builder.extend(points);

    // Create an immutable index from a builder
    let index = builder.build();

    let query = Point3D(2.0, 4.0, 16.0);
    let k = 10; // Number of NN to find
    let ef = 20; // Beamwidth must be >= k

    // Perform query
    // The index also supports performing multiple queries in parallel. To do so use `index.knns`
    // instead. This method takes an iterator of queries instead of a single query.
    let result = index.search(&query, k, &ef);

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
    // 49 : Point3D(2.0, 4.0, 9.0)
    // 50 : Point3D(1.0, 4.0, 9.0)
    // 50 : Point3D(2.0, 3.0, 9.0)
    // 50 : Point3D(2.0, 5.0, 9.0)
    // 50 : Point3D(3.0, 4.0, 9.0)
    // 51 : Point3D(1.0, 3.0, 9.0)
    // 51 : Point3D(1.0, 5.0, 9.0)
    // 51 : Point3D(3.0, 3.0, 9.0)
    // 51 : Point3D(3.0, 5.0, 9.0)
    // 53 : Point3D(0.0, 4.0, 9.0)
}
