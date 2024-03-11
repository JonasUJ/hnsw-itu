use rand;
use std::iter;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use hnsw_itu::Point;
use hnsw_itu_cli::Sketch;

#[inline(always)]
fn std_distance(a: Sketch, b: Sketch) -> usize {
    a.data
        .iter()
        .zip(b.data.iter())
        .fold(0, |acc, (lhs, rhs)| acc + (lhs ^ rhs).count_ones() as usize)
}

#[inline(always)]
fn cur_distance(a: Sketch, b: Sketch) -> usize {
    a.distance(&b)
}

pub fn distance_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance");

    let mut points = iter::repeat_with(|| Sketch::new(rand::random()));

    group.bench_function("std_distance", |b| {
        b.iter_batched(
            || (points.next().unwrap(), points.next().unwrap()),
            |(a, b)| std_distance(a, b),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("cur_distance", |b| {
        b.iter_batched(
            || (points.next().unwrap(), points.next().unwrap()),
            |(a, b)| cur_distance(a, b),
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, distance_benchmark);
criterion_main!(benches);
