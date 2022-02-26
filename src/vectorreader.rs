use std::io::BufRead;

use byteorder::{LittleEndian, ReadBytesExt};

use errors::Word2VecError;

pub struct WordVectorReader<R: BufRead> {
    vocabulary_size: usize,
    vector_size: usize,
    reader: R,
}

impl<R: BufRead> WordVectorReader<R> {
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    pub fn vector_size(&self) -> usize {
        self.vector_size
    }

    pub fn new_from_reader(mut reader: R) -> Result<WordVectorReader<R>, Word2VecError> {
        // Read UTF8 header string from start of file
        let mut header = String::with_capacity(128);
        reader.read_line(&mut header)?;

        //Parse 2 integers, separated by whitespace
        let header_info = header
            .split_whitespace()
            .filter_map(|x| x.parse::<usize>().ok())
            .take(2)
            .collect::<Vec<usize>>();
        match header_info.len() {
            2 => Ok(WordVectorReader {
                vocabulary_size: header_info[0],
                vector_size: header_info[1],
                reader,
            }),
            _ => Err(Word2VecError::WrongHeader),
        }
    }
}

impl<R: BufRead> Iterator for WordVectorReader<R> {
    type Item = (String, Vec<f32>);

    fn next(&mut self) -> Option<(String, Vec<f32>)> {
        let mut buf = Vec::with_capacity(32);
        self.reader.read_until(b' ', &mut buf).ok()?;

        let word = String::from_utf8(buf).ok()?.trim().into();
        let mut vector: Vec<f32> = Vec::with_capacity(self.vector_size);

        for _ in 0..self.vector_size {
            vector.push(self.reader.read_f32::<LittleEndian>().ok()?);
        }

        Some((word, vector))
    }
}
