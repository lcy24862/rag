from src.evaluation import RAGEvaluator
from src.rag_engine import RAGEngine
from config import CONFIG
import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

def load_evaluation_data(file_path):
    """从 JSON 文件加载评估数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # 初始化评估器
    evaluator = RAGEvaluator()
    load_dotenv()
    # 加载评估数据
    evaluation_data = load_evaluation_data("evaluation_data.json")
    test_questions = [item["question"] for item in evaluation_data]
    ground_truths = [item["answer"] for item in evaluation_data]
    
    # 1. 加载向量存储（从 chroma_db 目录）
    persist_directory = "chroma_db"  # 与 initialize.py 中的目录一致
    if not os.path.exists(persist_directory):
        print("错误：找不到向量存储目录。请先运行 initialize.py 进行系统初始化。")
        return
    
    print("加载向量存储...")
    embeddings = HuggingFaceEmbeddings(model_name=CONFIG["embedding_model"])
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="mechanical_rag"  # 与 initialize.py 中的名称一致
    )

    # 初始化RAG引擎
    api_key = "sk-6151b5dae67941d8b5b17d323aae9fe6"  # 替换为有效API密钥
    rag_engine = RAGEngine(
        vector_store=vector_store,
        openai_api_key=api_key,  # 关键修改：显式传递参数名
        base_url="https://api.deepseek.com/v1"
    )
    
    # 获取系统答案和上下文
    answers = []
    contexts = []
    
    for question in test_questions:
        answer, context = rag_engine.get_answer_with_context(question)
        answers.append(answer)
        contexts.append(context)
    
    # 执行评估
    results = evaluator.evaluate_batch(
        questions=test_questions,
        ground_truths=ground_truths,
        answers=answers,
        contexts=contexts
    )
    
    # 打印评估结果
    print("\n评估结果：")
    print("-" * 50)
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    main() 