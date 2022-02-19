#![feature(test)]
#![feature(async_closure)]
#![feature(generic_associated_types)]
extern crate test;
extern crate byteorder;

pub mod vectorreader;
pub mod wordvectors;
pub mod wordclusters;
mod utils;
pub mod errors;
