use crate::errors::Word2VecError;
use crate::utils;
use crate::vectorreader::WordVectorReader;
use std::cmp::Ordering;
use std::collections::hash_map::Keys;
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

/// Representation of a word vector space
///
/// Each word of a vocabulary is represented by a vector. All words span a vector space. This data
/// structure manages this vector space of words.
pub struct WordVector {
    vocabulary: HashMap<String, Vec<f32>>,
    vector_size: usize,
}

impl WordVector {
    /// Load a word vector space from file
    ///
    /// Word2vec is able to store the word vectors in a binary file. This function parses the file
    /// and loads the vectors into RAM.
    pub fn load_from_binary(file_name: &str) -> Result<WordVector, Word2VecError> {
        let file = File::open(file_name)?;
        let reader = BufReader::new(file);

        WordVector::load_from_reader(reader)
    }

    /// Load a word vector space from a reader
    ///
    /// Word2vec is able to store the word vectors in a binary format. This function parses the bytes in that format
    /// and loads the vectors into RAM.
    pub fn load_from_reader<R: BufRead>(reader: R) -> Result<WordVector, Word2VecError> {
        let reader = WordVectorReader::new_from_reader(reader)?;
        let vector_size = reader.vector_size();

        let mut vocabulary: HashMap<String, Vec<f32>> =
            HashMap::with_capacity(reader.vocabulary_size());
        for item in reader {
            let (word, mut vector) = item;
            utils::vector_norm(&mut vector);

            vocabulary.insert(word, vector);
        }

        Ok(WordVector {
            vocabulary,
            vector_size,
        })
    }

    /// Get word vector for the given word.
    pub fn get_vector(&self, word: &str) -> Option<&Vec<f32>> {
        self.vocabulary.get(word)
    }

    /// Compute consine distance to similar words.
    ///
    /// The words in the vector space are characterized through the position and angle to each
    /// other. This method calculates the `n` closest words via the cosine of the requested word to
    /// all other words.
    pub fn cosine(&self, word: &str, n: usize) -> Option<Vec<(String, f32)>> {
        let word_vector = self.vocabulary.get(word);
        match word_vector {
            Some(val) => {
                // save index and cosine distance to current word
                let mut metrics: Vec<(String, f32)> = self
                    .vocabulary
                    .iter()
                    .map(|(i, other_val)| (i.to_owned(), utils::dot_product(other_val, val)))
                    .collect();
                metrics.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                Some(metrics[1..n + 1].iter().map(|v| v.to_owned()).collect())
            }
            None => None,
        }
    }

    pub fn analogy(&self, pos: Vec<&str>, neg: Vec<&str>, n: usize) -> Option<Vec<(String, f32)>> {
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        let mut exclude: Vec<String> = Vec::new();
        for word in pos {
            exclude.push(word.to_string());
            match self.vocabulary.get(word) {
                Some(val) => vectors.push(val.to_owned()),
                None => {}
            }
        }
        for word in neg.iter() {
            exclude.push(word.to_string());
            match self.vocabulary.get(word.to_owned()) {
                Some(val) => vectors.push(val.iter().map(|x| -x).collect::<Vec<f32>>()),
                None => {}
            }
        }
        if exclude.is_empty() {
            return None;
        }
        let mut mean: Vec<f32> = Vec::with_capacity(self.vector_size);
        for i in 0..self.vector_size {
            mean.push(utils::mean(vectors.iter().map(|v| v[i])));
        }
        let mut metrics: Vec<(&String, f32)> = Vec::new();
        for word in self.vocabulary.iter() {
            metrics.push((word.0, utils::dot_product(word.1, &mean)));
        }
        metrics.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        metrics.retain(|x| !exclude.contains(x.0));
        Some(
            metrics
                .iter()
                .take(n)
                .map(|&(x, y)| (x.clone(), y))
                .collect(),
        )
    }

    /// Get the number of all known words from the vocabulary.
    pub fn word_count(&self) -> usize {
        self.vocabulary.len()
    }

    /// Return the number of columns of the word vector.
    pub fn get_col_count(&self) -> usize {
        self.vector_size // size == column count
    }

    /// Get all known words from the vocabulary.
    pub fn get_words(&self) -> Words {
        Words::new(&self.vocabulary)
    }
}

#[derive(Debug)]
pub struct Words<'parent> {
    iter: Keys<'parent, String, Vec<f32>>,
}

impl<'a> Words<'a> {
    fn new(x: &'a HashMap<String, Vec<f32>>) -> Words<'a> {
        Words { iter: x.keys() }
    }
}

impl<'a> Iterator for Words<'a> {
    type Item = &'a String;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}
