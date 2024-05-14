use std::{collections::HashSet, iter};

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use hnsw_itu::{BitSet, GenerationSet, Set};
use rand::Rng;

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

fn hashset_new_insert(lst: Vec<usize>) {
    let mut set = HashSet::new();
    lst.into_iter().for_each(|p| {
        set.insert(p);
    });
}

fn hashset_with_capacity_insert(lst: Vec<usize>) {
    let mut set = HashSet::with_capacity(2000);
    lst.into_iter().for_each(|p| {
        set.insert(p);
    });
}

fn bitset_insert(lst: Vec<usize>) {
    let mut set = BitSet::new(10_000_000);
    lst.into_iter().for_each(|p| {
        set.insert(p);
    });
}

fn generationset_insert(lst: Vec<usize>) {
    let mut set = GenerationSet::new(10_000_000);
    lst.into_iter().for_each(|p| {
        set.insert(p);
    });
}

pub fn set_insert_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Set insert");

    let size = 2_000;
    let mut values = iter::repeat_with(|| {
        (0..size)
            .map(|_| rand::thread_rng().gen_range(0..size))
            .collect()
    });

    bench!("Hashset new insert", hashset_new_insert, values, group);
    bench!(
        "Hashset with capacity insert",
        hashset_with_capacity_insert,
        values,
        group
    );
    bench!("Bitset insert", bitset_insert, values, group);
    bench!("Generationset insert", generationset_insert, values, group);
}

criterion_group!(benches, set_insert_benchmark);
criterion_main!(benches);
