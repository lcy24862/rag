import os
import json
from config import CONFIG
from src.rag_engine import RAGEngine
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from data import queries, responses
from ragas import EvaluationDataset

CACHE_FILE = "dataset_cache.json"

def load_or_generate_dataset(use_cache=True):
    """加载缓存或生成新的数据集"""
    if use_cache and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # 初始化 RAG 引擎
    embeddings = HuggingFaceEmbeddings(model_name=CONFIG["embedding_model"])
    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings,
        collection_name="mechanical_rag"
    )
    rag_engine = RAGEngine(
        vector_store=vector_store,
        openai_api_key=CONFIG["api_key"],
        base_url=CONFIG["base_url"]
    )

    # 生成数据集
    dataset = []
    for query, reference in zip(queries, responses):
        answer, context = rag_engine.get_answer_with_context(query)
        dataset.append({
            "user_input": query,
            "retrieved_contexts": context,
            "response": answer,
            "reference": reference
        })
    evaluation_dataset = EvaluationDataset.from_list(dataset)

    # 缓存结果
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(evaluation_dataset, f, ensure_ascii=False, indent=2)
    
    return evaluation_dataset