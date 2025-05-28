from config import CONFIG
from src.rag_engine import RAGEngine
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

def main():
    load_dotenv()
    
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
    
    # 2. 初始化RAG引擎
    api_key = CONFIG["api_key"]  # 替换为有效API密钥
    rag_engine = RAGEngine(
        vector_store=vector_store,
        openai_api_key=api_key,  # 关键修改：显式传递参数名
        base_url=CONFIG["base_url"]
    )
    
    # 3. 启动问答系统
    print("欢迎使用RAG问答系统！输入'quit'退出。")
    while True:
        question = input("\n请输入您的问题：")
        if question.lower() == 'quit':
            break
            
        try:
            answer, contexts = rag_engine.get_answer_with_context(question)
            print("\n回答：", answer)
            print("\n参考上下文：")
            for i, context in enumerate(contexts, 1):
                print(f"\n{i}. {context}")
        except Exception as e:
            print(f"发生错误：{str(e)}")

if __name__ == "__main__":
    main()