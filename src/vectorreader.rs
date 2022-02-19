use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt};
use crate::errors::Word2VecError;
use crate::utils;
use crate::wordvectors::WordVector;

use async_trait::async_trait;

pub struct WordVectorReader<R: 'static + tokio::io::AsyncBufRead> {
    vocabulary_size: usize,
    vector_size: usize,
    reader: &'static mut R,
}

impl<R: AsyncBufRead + Unpin> WordVectorReader<R> {
    pub async fn new_from_reader(
        reader: &'static mut R,
    ) -> Result<WordVectorReader<R>, Word2VecError> {
        let mut header = String::new();
        // Read UTF8 header string from start of file
        let _ = &reader.read_line(&mut header).await.unwrap();

        //Parse 2 integers, separated by whitespace
        let header_info = header
            .split_whitespace()
            .filter_map(|x| x.parse::<usize>().ok())
            .take(2)
            .collect::<Vec<usize>>();
        if header_info.len() != 2 {
            return Err(Word2VecError::WrongHeader);
        }

        //We've successfully read the header, ready to read vectors
        Ok(WordVectorReader {
            vocabulary_size: header_info[0],
            vector_size: header_info[1],
            reader,
        })
    }
}

#[async_trait]
pub trait WordVectorBuilder {
    async fn build_vocabulary(self) -> Result<&'static WordVector, Word2VecError>;
}

#[async_trait]
impl<R: AsyncBufRead + Unpin + Send> WordVectorBuilder for WordVectorReader<R> {
    async fn build_vocabulary(self) -> Result<&'static WordVector, Word2VecError> {
        let mut vocabulary = Vec::with_capacity(self.vocabulary_size);
        for _ in 0..self.vocabulary_size {
            let mut word_bytes: Vec<u8> = Vec::new();
            if let Err(e) = self.reader.read_until(b' ', &mut word_bytes).await {
                return Err(Word2VecError::from(e));
            }

            // trim newlines, some vector files have newlines in front of a new word, others don't
            let word = match String::from_utf8(word_bytes) {
                Err(e) => {
                    return Err(Word2VecError::from(e));
                }
                Ok(word) => word.trim().into(),
            };

            // Read floats of the vector
            let mut vector = Vec::<f32>::with_capacity(self.vector_size);

            for _ in 0..self.vector_size {
                match self.reader.read_f32_le().await {
                    Err(e) => return Err(Word2VecError::from(e)),
                    Ok(value) => vector.push(value),
                }
            }

            utils::vector_norm(&mut vector);
            // one iteration
            vocabulary.push((word, vector));
        }
        let vector_size = self.vector_size;
        let bomx = Box::new(WordVector {
            vocabulary,
            vector_size,
        });
        let ptr = Box::leak(bomx);

        Ok(ptr)
    }
}
