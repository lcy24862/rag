from src.evaluation import RAGEvaluator
from src.rag_engine import RAGEngine
from config import CONFIG
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from data import queries, responses  # 从 data.py 导入数据
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper

def main():
    # 初始化评估器
    evaluator = RAGEvaluator()
    load_dotenv()

    # 加载向量存储
    persist_directory = "chroma_db"
    if not os.path.exists(persist_directory):
        print("错误：找不到向量存储目录。请先运行 initialize.py 进行系统初始化。")
        return

    print("加载向量存储...")
    embeddings = HuggingFaceEmbeddings(model_name=CONFIG["embedding_model"])
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="mechanical_rag"
    )

    # 初始化RAG引擎
    api_key = CONFIG["api_key"]
    rag_engine = RAGEngine(
        vector_store=vector_store,
        openai_api_key=api_key,
        base_url=CONFIG["base_url"]
    )

    dataset = []

    for query,reference in zip(queries,responses):

        infer_response,relevant_docs = rag_engine.get_answer_with_context(query)
        dataset.append(
            {
                "user_input":query,
                "retrieved_contexts":relevant_docs,
                "response":infer_response,
                "reference":reference
            }
    )

    print(dataset)

"""
    evaluator_llm = LangchainLLMWrapper(llm)
    from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

    result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=evaluator_llm)
    result"""

if __name__ == "__main__":
    main() 