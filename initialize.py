from src.data_loader import load_textbook
from src.text_splitter import split_documents
from src.vector_store import create_vector_store
from config import CONFIG
from langchain.schema import Document

def initialize_rag():
    print("开始初始化RAG系统...")
    
    # 1. 加载教材
    print("1. 加载教材...")
    documents = load_textbook(CONFIG["data_path"])
    print(f"加载完成，共加载 {len(documents)} 篇文档。")

    # 拼接所有文档内容（保留 Document 对象结构）
    combined_documents = [
        Document(page_content="\n".join([doc.page_content for doc in documents]))
    ]

    # 2. 文本分割
    print("\n2. 进行文本分割...")
    chunks = split_documents(combined_documents, CONFIG["chunk_size"], CONFIG["chunk_overlap"])
    print(f"分割完成，共生成 {len(chunks)} 个文本块。")
    print("示例文本块内容（前100字符）：")
    if chunks:
        print(chunks[0].page_content[:100] + "...")

    # 3. 创建向量存储（启用持久化）
    print("\n3. 创建向量存储...")
    persist_directory = "chroma_db"  # 持久化目录
    vector_store = create_vector_store(
        chunks,
        CONFIG["embedding_model"],
        persist_directory=persist_directory
    )
    print(f"向量存储创建完成，已保存到目录：{persist_directory}")
    
    # 无需手动保存，Chroma 已通过 persist() 保存
    print("\n初始化完成！")

if __name__ == "__main__":
    initialize_rag() 