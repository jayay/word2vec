extern crate word2vec;
use word2vec::vectorreader::WordVectorReader;
use word2vec::wordvectors::WordVector;
use word2vec::wordvectors::Words;

const PATH: &str = "vectors.bin";

#[test]
fn test_word_cosine() {
    let model = WordVector::load_from_binary(PATH).unwrap();
    let res = model
        .cosine("winter", 10)
        .expect("word not found in vocabulary");
    assert_eq!(res.len(), 10);
    let only_words: Vec<&str> = res.iter().map(|x| x.0.as_ref()).collect();
    assert!(!only_words.contains(&"winter"))
}

#[test]
fn test_nonexisting_word_cosine() {
    let model = WordVector::load_from_binary(PATH).unwrap();
    let result = model.cosine("somenotexistingword", 10);
    assert!(result.is_none());
}

#[test]
fn test_word_analogy() {
    let model = WordVector::load_from_binary(PATH).unwrap();
    let mut pos = Vec::new();
    pos.push("woman");
    pos.push("king");
    let mut neg = Vec::new();
    neg.push("man");
    let res = model
        .analogy(pos, neg, 10)
        .expect("couldn't find all of the given words");
    assert_eq!(res.len(), 10);
    let only_words: Vec<&str> = res.iter().map(|x| x.0.as_ref()).collect();
    assert!(!only_words.contains(&"woman"));
    assert!(!only_words.contains(&"king"));
    assert!(!only_words.contains(&"man"));
}

#[test]
fn test_word_analogy_with_empty_params() {
    let model = WordVector::load_from_binary(PATH).unwrap();
    let result = model.analogy(Vec::new(), Vec::new(), 10);
    assert!(result.is_none());
}

#[test]
fn test_word_count_is_correctly_returned() {
    let v = WordVector::load_from_binary(PATH).unwrap();
    assert_eq!(v.word_count(), 71291);
}

#[test]
fn test_words() {
    let v = WordVector::load_from_binary(PATH).unwrap();
    assert_eq!(v.get_words().count(), 71291);
}

fn is_normal<T: Sized + Send + Unpin>() {}

#[test]
fn normal_types() {
    is_normal::<WordVector>();
    is_normal::<WordVectorReader<std::io::BufReader<std::fs::File>>>();
    is_normal::<Words>();
}
