from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search, cos_sim
from preprocess import DataPreprocessor


def initialize_model(model_name: str) -> SentenceTransformer:
    model = SentenceTransformer(model_name)
    return model


def find_embeddings(model: SentenceTransformer, input_text: str):
    embeddings = model.encode(input_text)
    return embeddings


def find_similarity(query_embeddings, corpus_embeddings):
    similarity = semantic_search(
        query_embeddings, corpus_embeddings, score_function=cos_sim, top_k=2
    )
    return similarity[0]


def main():
    model = initialize_model("bert-base-uncased")
    preprocess_data = DataPreprocessor("../../data/raw/train.txt")
    input_data = preprocess_data.read_file()
    corpus_embeddings = find_embeddings(model, input_data)
    similarity = find_similarity(corpus_embeddings[0], corpus_embeddings)
    print(similarity)


if __name__ == "__main__":
    main()
