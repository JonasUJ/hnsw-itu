#![feature(portable_simd)]
use core::simd::*;
use std::iter;
use std::ops::{BitAnd, Shr};
use std::simd::num::SimdUint;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};

use hnsw_itu::Point;
use hnsw_itu_cli::Sketch;

macro_rules! bench {
    ($name:literal, $distance:ident, $points:ident, $group:ident) => {
        $group.bench_function($name, |b| {
            b.iter_batched(
                || ($points.next().unwrap(), $points.next().unwrap()),
                |(a, b)| $distance(a, b),
                BatchSize::SmallInput,
            )
        });
    };
}

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

#[inline(always)]
fn avx_count(v: u64x16) -> u64 {
    let _lookup = [
        1u64,           // 0, 1
        4294967298u64,  // 1, 2
        4294967298u64,  // 1, 2
        8589934595u64,  // 2, 3
        4294967298u64,  // 1, 2
        8589934595u64,  // 2, 3
        8589934595u64,  // 2, 3
        12884901892u64, // 3, 4
        1u64,           // 0, 1
        4294967298u64,  // 1, 2
        4294967298u64,  // 1, 2
        8589934595u64,  // 2, 3
        4294967298u64,  // 1, 2
        8589934595u64,  // 2, 3
        8589934595u64,  // 2, 3
        12884901892u64, // 3, 4
    ];
    let lookup: u64x16 = u64x16::from_array(_lookup);
    let low_mask = u64x16::from_array([0xf0000000f; 16]);
    let lo = u64x16::bitand(lookup, low_mask);
    let hi = u64x16::bitand(v.shr(4), low_mask);
    let popcnt1 = u64x16::interleave(lookup, lo).0;
    let popcnt2 = u64x16::interleave(lookup, hi).0;
    let total: u64x16 = popcnt1 + popcnt2;
    total.reduce_sum()
}

#[inline(always)]
fn avx_distance(a: Sketch, b: Sketch) -> usize {
    let a = u64x16::from(a.data);
    let b = u64x16::from(b.data);
    avx_count(a ^ b) as usize
}

#[inline(always)]
fn avx_xor_distance(a: Sketch, b: Sketch) -> usize {
    let a = u64x16::from(a.data);
    let b = u64x16::from(b.data);
    (a ^ b)
        .as_array()
        .iter()
        .fold(0, |acc, x| acc + x.count_ones()) as usize
}

pub fn distance_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance");

    let mut points = iter::repeat_with(|| Sketch::new(rand::random()));

    bench!("std_distance", std_distance, points, group);
    bench!("cur_distance", cur_distance, points, group);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            bench!("avx2_distance", avx_distance, points, group);
            bench!("avx2_xor_distance", avx_xor_distance, points, group);
        }
    }
}

criterion_group!(benches, distance_benchmark);
criterion_main!(benches);

#[cfg(test)]
mod tests {
    #[test]
    fn test_avx_distance() {
        let mut points = iter::repeat_with(|| Sketch::new(rand::random()));
        for _ in 0..100 {
            let a = points.next();
            let b = points.next();
            assert_eq!(std_distance(a, b), avx_distance(a, b));
        }
    }
}
