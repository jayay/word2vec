extern crate word2vec;

#[tokio::main]
async fn main(){
	let model = word2vec::wordvectors::WordVector::load_from_binary(
		"vectors.bin").await.expect("Unable to load word vector model");
	println!("{:?}", model.cosine("snow", 10).await);
	let positive = vec!["woman", "king"];
	let negative = vec!["man"];
	println!("{:?}", model.analogy(positive, negative, 10).await);
	
	let clusters = word2vec::wordclusters::WordClusters::load_from_file(
		"classes.txt").expect("Unable to load word clusters");
	println!("{:?}", clusters.get_cluster("belarus"));
	println!("{:?}", clusters.get_words_on_cluster(6));
}

