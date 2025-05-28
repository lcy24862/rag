CONFIG = {
    "data_path": "data/textbook/textbook.pdf",
    "chunk_size": 1000,
    "chunk_overlap": 600,
    "embedding_model": "./models/text2vec-base-chinese",
    "base_url": "https://api.deepseek.com/v1",
    "api_key": "sk-6151b5dae67941d8b5b17d323aae9fe6",
    "vector_store": "chroma",
    "llm_model": "deepseek-chat",
    "temperature": 0.7,
    "top_k": 3,
    "vector_store_path": "chroma_db"
}
