use std::{collections::HashSet, iter};

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use hnsw_itu::{BitSet, Set};

macro_rules! bench {
    ($name:literal, $func:ident, $values:ident, $group:ident) => {
        $group.bench_function($name, |b| {
            b.iter_batched(
                || $values.next().unwrap(),
                |lst| $func(lst),
                BatchSize::SmallInput,
            )
        });
    };
}

fn hashset_insert(lst: Vec<usize>) {
    let mut set = HashSet::new();
    lst.into_iter().for_each(|p| {
        set.insert(p);
    });
}

fn bitset_insert(lst: Vec<usize>) {
    let mut set = BitSet::new(1000);
    lst.into_iter().for_each(|p| {
        set.insert(p);
    });
}

pub fn set_insert_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Set insert");

    let size = 1000;
    let mut points = iter::repeat_with(|| (0..size).collect());

    bench!("Hashset insert", hashset_insert, points, group);
    bench!("Bitset insert", bitset_insert, points, group);
}

criterion_group!(benches, set_insert_benchmark);
criterion_main!(benches);
