#![feature(test)]
extern crate test;
extern crate word2vec;

use word2vec::wordvectors::WordVector;

const PATH: &str = "vectors.bin";

#[tokio::test]
async fn test_word_cosine() {
    let model = WordVector::load_from_binary(PATH).await.unwrap();
    let res = model
        .cosine("winter", 10)
        .await
        .expect("word not found in vocabulary");
    assert_eq!(res.len(), 10);
    assert!(!res.iter().map(|x| x.0.as_ref() as &str).any(|x| x == "winter"))
}

#[tokio::test]
async fn test_unexisting_word_cosine() {
    let model = WordVector::load_from_binary(PATH).await.unwrap();
    let result = model.cosine("somenotexistingword", 10).await;
    assert!(result.is_none())
}

#[tokio::test]
async fn test_word_analogy() {
    let model = WordVector::load_from_binary(PATH).await.unwrap();
    let pos = vec!["woman", "king"];
    let neg = vec!["man"];
    let res = model
        .analogy(&pos, &neg, 10)
        .await
        .expect("couldn't find all of the given words");
    assert_eq!(res.len(), 10);
    let only_words: Vec<&str> = res.iter().map(|x| x.0.as_ref()).collect();
    assert!(!only_words.contains(&"woman"));
    assert!(!only_words.contains(&"king"));
    assert!(!only_words.contains(&"man"));
}

#[tokio::test]
async fn test_word_analogy_with_empty_params() {
    let model = WordVector::load_from_binary(PATH).await.unwrap();
    let result = model.analogy(&Vec::new(), &Vec::new(), 10).await;
    assert!(result.is_none());
}

#[tokio::test]
async fn test_word_count_is_correctly_returned() {
    let v = WordVector::load_from_binary(PATH).await.unwrap();
    assert_eq!(v.word_count().await, 71291);
}

#[tokio::test]
async fn test_words() {
    let v = WordVector::load_from_binary(PATH).await.unwrap();
    assert_eq!(v.get_words().await.count(), 71291);
}
