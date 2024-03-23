use std::{collections::HashSet, iter};

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use hnsw_itu::{BitSet, Set};

macro_rules! bench {
    ($name:literal, $func:ident, $set:ident, $values:ident, $group:ident) => {
        $group.bench_function($name, |b| {
            b.iter_batched(
                || ($set.clone(), $values.next().unwrap()),
                |(set, lst)| $func(set, lst),
                BatchSize::SmallInput,
            )
        });
    };
}

fn hashset_contains(set: HashSet<usize>, lst: Vec<usize>) {
    lst.into_iter().for_each(|p| {
        set.contains(&p);
    });
}

fn bitset_contains(set: BitSet, lst: Vec<usize>) {
    lst.into_iter().for_each(|p| {
        set.contains(p);
    });
}

pub fn set_contains_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Set contains");

    let size = 1000;
    let mut points = iter::repeat_with(|| (0..size).collect());

    let mut hashset = HashSet::new();
    let mut bitset = BitSet::new(1000);

    (0..size/2).into_iter().for_each(|p| {
        hashset.insert(p);
        bitset.insert(p);
    });

    bench!("Hashset contains", hashset_contains, hashset, points, group);
    bench!("Bitset contains", bitset_contains, bitset, points, group);
}

criterion_group!(benches, set_contains_benchmark);
criterion_main!(benches);
