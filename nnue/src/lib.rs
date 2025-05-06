#![cfg_attr(
    all(target_arch = "x86_64", target_feature = "avx512f"),
    feature(stdarch_x86_avx512)
)]

pub mod accummulator;
pub mod flatten;
pub mod params;
pub mod utils;
