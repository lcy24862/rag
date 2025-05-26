from langchain_community.document_loaders import PyPDFLoader
import os

def load_textbook(pdf_path):
    """
    使用LangChain的PyPDFLoader加载PDF教材并提取文本内容
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"找不到文件：{pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    # 支持同步加载
    pages = loader.load()
    return pages
