#![feature(test)]
#![feature(async_closure)]
#![feature(generic_associated_types)]
extern crate byteorder;
extern crate test;

pub mod errors;
mod utils;
pub mod vectorreader;
pub mod wordclusters;
pub mod wordvectors;
