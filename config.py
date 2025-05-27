CONFIG = {
    "data_path": "data/textbook/textbook.pdf",
    "chunk_size": 1000,
    "chunk_overlap": 600,
    "embedding_model": "./models/text2vec-base-chinese",
    "vector_store": "chroma",
    "llm_model": "deepseek-chat",
    "temperature": 0.7,
    "top_k": 3,
    "vector_store_path": "chroma_db"
}
