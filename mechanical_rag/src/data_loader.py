import PyPDF2
import os

def load_textbook(pdf_path):
    """
    加载PDF教材并提取文本内容
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"找不到文件：{pdf_path}")
    
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text
