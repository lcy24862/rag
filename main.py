from src.data_loader import load_textbook
from src.text_splitter import split_documents
from src.vector_store import create_vector_store
from src.rag_engine import RAGEngine
from config import CONFIG
import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    # 1. 加载教材
    documents = load_textbook(CONFIG["data_path"])
    
    # 2. 文本分割
    chunks = split_documents(documents, CONFIG["chunk_size"], CONFIG["chunk_overlap"])
    
    # 3. 创建向量存储
    vector_store = create_vector_store(chunks, CONFIG["embedding_model"])
    
    # 4. 初始化RAG引擎
    api_key = "sk-6151b5dae67941d8b5b17d323aae9fe6"
    rag_engine = RAGEngine(vector_store, api_key, "https://api.deepseek.com")
    
    # 5. 启动问答系统
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