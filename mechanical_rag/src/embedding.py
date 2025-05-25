from sentence_transformers import SentenceTransformer

def create_embeddings(texts, model_name="text2vec-base-chinese"):
    """
    创建文本的向量嵌入
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings
