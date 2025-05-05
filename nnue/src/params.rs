use std::mem;

use crate::utils::Align64;

// Layer sizes
pub const INPUT: usize = 768;
pub const L1: usize = 1024;

// Quantization factors
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const QAB: i32 = QA * QB;
pub const SCALE: i32 = 400;

// Squared Clipped ReLU
#[inline]
pub fn screlu(x: i16) -> i32 {
    (x.clamp(0, QA as i16) as i32).pow(2)
}

#[repr(C)]
#[rustfmt::skip]
pub struct NNUEParams {
    pub feature_weights: Align64<[i16; INPUT * L1]>,
    pub feature_bias:    Align64<[i16; L1]>,
    pub output_weights:  Align64<[i16; L1 * 2]>,
    pub output_bias: i16,
}

pub static MODEL: NNUEParams = unsafe { mem::transmute(*include_bytes!("../../data/small.nnue")) };
