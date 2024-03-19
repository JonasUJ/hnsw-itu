#![feature(portable_simd)]
use std::{
    arch::x86_64::{
        __m256i, _mm256_add_epi64, _mm256_add_epi8, _mm256_and_si256, _mm256_loadu_si256,
        _mm256_sad_epu8, _mm256_set1_epi8, _mm256_setr_epi8, _mm256_setzero_si256,
        _mm256_shuffle_epi8, _mm256_srli_epi32,
    },
    iter,
    simd::u64x16,
};

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
fn avx_count(v: __m256i) -> __m256i {
    unsafe {
        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );
        let low_mask = _mm256_set1_epi8(0x0f);
        let lo = _mm256_and_si256(v, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), low_mask);
        let popcnt1 = _mm256_shuffle_epi8(lookup, lo);
        let popcnt2 = _mm256_shuffle_epi8(lookup, hi);
        let total = _mm256_add_epi8(popcnt1, popcnt2);
        _mm256_sad_epu8(total, _mm256_setzero_si256())
    }
}

#[inline(always)]
fn avx_distance(a: Sketch, b: Sketch) -> usize {
    let a = u64x16::from(a.data);
    let b = u64x16::from(b.data);
    let d = (a ^ b).to_array();
    unsafe {
        let v1 = avx_count(_mm256_loadu_si256(d.as_ptr() as *const __m256i));
        let v2 = avx_count(_mm256_loadu_si256(d[4..8].as_ptr() as *const __m256i));
        let v3 = avx_count(_mm256_loadu_si256(d[8..12].as_ptr() as *const __m256i));
        let v4 = avx_count(_mm256_loadu_si256(d[12..16].as_ptr() as *const __m256i));
        let r1 = _mm256_add_epi64(v1, v2);
        let r2 = _mm256_add_epi64(v3, v4);
        let res: [u64; 4] = std::mem::transmute(_mm256_add_epi64(r1, r2));
        res.iter().fold(0, |acc, x| acc + x) as usize
    }
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
