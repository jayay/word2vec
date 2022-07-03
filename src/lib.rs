#![feature(test)]
#![feature(portable_simd)]
extern crate byteorder;
extern crate test;

pub mod errors;
mod utils;
pub mod vectorreader;
pub mod wordclusters;
pub mod wordvectors;
