use errors::Word2VecError;
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;

pub struct WordClusters {
    clusters: HashMap<i32, Vec<String>>,
}

impl WordClusters {
    pub fn load_from_file(file_name: &str) -> Result<WordClusters, Word2VecError> {
        let file = File::open(file_name)?;
        let reader = BufReader::new(file);

        WordClusters::load_from_reader(reader)
    }

    pub fn load_from_reader<R: BufRead>(mut reader: R) -> Result<WordClusters, Word2VecError> {
        let mut buffer = String::new();
        let mut clusters: HashMap<i32, Vec<String>> = HashMap::new();
        while reader.read_line(&mut buffer)? > 0 {
            {
                let mut iter = buffer.split_whitespace();
                let word = iter.next().unwrap();
                let cluster_number = iter.next().unwrap().trim().parse::<i32>().ok().unwrap();
                let cluster = clusters.entry(cluster_number).or_default();
                cluster.push(word.to_string());
            }
            buffer.clear();
        }
        Ok(WordClusters { clusters })
    }

    pub fn get_words_on_cluster(&self, index: i32) -> Option<&Vec<String>> {
        self.clusters.get(&index)
    }

    pub fn get_cluster(&self, word: &str) -> Option<&i32> {
        let word = word.to_string();
        for (key, val) in self.clusters.iter() {
            if val.contains(&word) {
                return Some(key);
            }
        }
        None
    }
}
