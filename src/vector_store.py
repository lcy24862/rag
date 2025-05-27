# src/vector_store.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

def create_vector_store(documents, embedding_model_name, collection_name="mechanical_rag", persist_directory=None):
    """
    创建向量数据库，自动持久化到本地
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # 如果指定了持久化目录，则启用自动持久化
    if persist_directory:
        os.makedirs(persist_directory, exist_ok=True)
        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory  # 自动保存到磁盘
        )
    else:
        db = Chroma.from_documents(documents, embeddings, collection_name=collection_name)
    
    return db
