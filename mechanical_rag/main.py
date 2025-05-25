# main.py
from src.data_loader import load_textbook
from src.text_splitter import split_text
from src.embedding import create_embeddings
from src.vector_store import create_vector_store
from src.rag_engine import RAGEngine
from config import CONFIG

def main():
    # 1. 加载教材
    text = load_textbook(CONFIG["data_path"])
    
    # 2. 文本分割
    chunks = split_text(text, CONFIG["chunk_size"], CONFIG["chunk_overlap"])
    
    # 3. 创建向量嵌入
    embeddings = create_embeddings(chunks, CONFIG["embedding_model"])
    
    # 4. 创建向量存储
    vector_store = create_vector_store(embeddings, chunks)
    
    # 5. 初始化RAG引擎
    rag_engine = RAGEngine(vector_store)
    
    # 6. 启动问答系统
    while True:
        question = input("请输入您的问题（输入'quit'退出）：")
        if question.lower() == 'quit':
            break
        answer = rag_engine.get_answer(question)
        print(f"回答：{answer}")

if __name__ == "__main__":
    main()