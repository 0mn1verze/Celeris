use crate::accumulator::SideAccumulator;
#[cfg(any(target_feature = "avx2", target_feature = "avx512f"))]
use crate::params::QA;

pub fn flatten(acc: &SideAccumulator, weights: &SideAccumulator) -> i32 {
    #[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
    {
        fallback::flatten(acc, weights)
    }

    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    unsafe {
        avx2::flatten(acc, weights)
    }

    #[cfg(target_feature = "avx512f")]
    unsafe {
        avx512::flatten(acc, weights)
    }
}

#[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
mod fallback {
    use crate::{accumulator::SideAccumulator, params::QA};

    #[inline]
    pub fn screlu(x: i16) -> i32 {
        (x.clamp(0, QA as i16) as i32).pow(2)
    }

    #[inline]
    pub fn flatten(acc: &SideAccumulator, weights: &SideAccumulator) -> i32 {
        let mut sum = 0;

        for (&x, &w) in acc.0.iter().zip(&weights.0) {
            sum += screlu(x) * i32::from(w);
        }

        sum
    }
}

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
mod avx2 {
    use std::arch::x86_64::*;

    use crate::{
        accumulator::SideAccumulator,
        params::{L1, QA},
    };

    pub unsafe fn flatten(acc: &SideAccumulator, weights: &SideAccumulator) -> i32 {
        use std::arch::x86_64::*;

        const CHUNK: usize = 16;

        let mut sum = _mm256_setzero_si256();
        let min = _mm256_setzero_si256();
        let max = _mm256_set1_epi16(QA as i16);

        for i in 0..L1 / CHUNK {
            let mut v = load_i16s(acc, i * CHUNK);
            v = _mm256_min_epi16(_mm256_max_epi16(v, min), max);
            let w = load_i16s(weights, i * CHUNK);
            let product = _mm256_madd_epi16(v, _mm256_mullo_epi16(v, w));
            sum = _mm256_add_epi32(sum, product);
        }

        let upper_128 = _mm256_extracti128_si256::<1>(sum);
        let lower_128 = _mm256_castsi256_si128(sum);
        let sum_128 = _mm_add_epi32(upper_128, lower_128);
        let upper_64 = _mm_unpackhi_epi64(sum_128, sum_128);
        let sum_64 = _mm_add_epi32(upper_64, sum_128);
        let upper_32 = _mm_shuffle_epi32::<0b00_00_00_01>(sum_64);
        let sum_32 = _mm_add_epi32(upper_32, sum_64);

        horizontal_sum_i32(sum)
    }

    #[inline]
    unsafe fn load_i16s(acc: &SideAccumulator, start_idx: usize) -> __m256i {
        _mm256_load_si256(acc.0.as_ptr().add(start_idx).cast())
    }

    #[inline]
    unsafe fn horizontal_sum_i32(sum: __m256i) -> i32 {
        let upper_128 = _mm256_extracti128_si256::<1>(sum);
        let lower_128 = _mm256_castsi256_si128(sum);
        let sum_128 = _mm_add_epi32(upper_128, lower_128);
        let upper_64 = _mm_unpackhi_epi64(sum_128, sum_128);
        let sum_64 = _mm_add_epi32(upper_64, sum_128);
        let upper_32 = _mm_shuffle_epi32::<0b00_00_00_01>(sum_64);
        let sum_32 = _mm_add_epi32(upper_32, sum_64);

        _mm_cvtsi128_si32(sum_32)
    }
}

#[cfg(target_feature = "avx512f")]
mod avx512 {
    use std::arch::x86_64::*;

    use crate::{
        accumulator::SideAccumulator,
        params::{L1, QA},
    };

    pub unsafe fn flatten(acc: &SideAccumulator, weights: &SideAccumulator) -> i32 {
        const CHUNK: usize = 32;

        let mut sum = _mm512_setzero_si512();
        let min = _mm512_setzero_si512();
        let max = _mm512_set1_epi16(QA as i16);

        for i in 0..L1 / CHUNK {
            let mut v = load_i16s(acc, i * CHUNK);
            v = _mm512_min_epi16(_mm512_max_epi16(v, min), max);
            let w = load_i16s(weights, i * CHUNK);
            let product = _mm512_madd_epi16(v, _mm512_mullo_epi16(v, w));
            sum = _mm512_add_epi32(sum, product);
        }

        _mm512_reduce_add_epi32(sum)
    }

    #[inline]
    unsafe fn load_i16s(acc: &SideAccumulator, start_idx: usize) -> __m512i {
        _mm512_load_si512(acc.0.as_ptr().add(start_idx).cast())
    }
}
