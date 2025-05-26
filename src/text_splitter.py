from langchain.text_splitter import RecursiveCharacterTextSplitter



def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """
    将Document对象列表分割成小块
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)
