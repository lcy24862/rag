# src/vector_store.py
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def create_vector_store(documents, embedding_model_name, collection_name="mechanical_rag"):
    """
    直接用Chroma.from_documents创建向量数据库
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = Chroma.from_documents(documents, embeddings, collection_name=collection_name)
    return db
