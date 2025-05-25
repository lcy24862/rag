import chromadb
from chromadb.config import Settings

def create_vector_store(embeddings, texts):
    """
    创建向量存储
    """
    client = chromadb.Client(Settings(
        persist_directory="vector_store"
    ))
    
    collection = client.create_collection(name="mechanical_rag")
    
    # 添加文档和向量
    collection.add(
        embeddings=embeddings.tolist(),
        documents=texts,
        ids=[str(i) for i in range(len(texts))]
    )
    
    return collection
