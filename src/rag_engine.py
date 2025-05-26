# src/rag_engine.py
from openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class RAGEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.client = OpenAI(
            api_key="sk-6151b5dae67941d8b5b17d323aae9fe6",
            base_url="https://api.deepseek.com"
        )
        
        # 创建检索器
        self.retriever = self._similarity_search
        
        # 创建上下文压缩器
        compressor = LLMChainExtractor.from_llm(self._get_llm())
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.retriever
        )
    
    def _similarity_search(self, query, k=3):
        """
        使用 chromadb 进行相似度搜索
        """
        results = self.vector_store.query(
            query_texts=[query],
            n_results=k
        )
        
        # 将结果转换为 Document 对象列表
        from langchain.schema import Document
        docs = []
        for i in range(len(results['documents'][0])):
            docs.append(Document(
                page_content=results['documents'][0][i],
                metadata={'id': results['ids'][0][i]}
            ))
        return docs
    
    def _get_llm(self):
        """获取LLM实例"""
        return self.client
    
    def get_answer(self, question):
        """
        获取问题的答案
        """
        # 检索相关文档
        docs = self.retriever(question, k=3)
        
        # 构建提示
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"""基于以下上下文回答问题。如果上下文中没有相关信息，请说明无法回答。

上下文：
{context}

问题：{question}

回答："""
        
        # 调用DeepSeek API
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个有帮助的AI助手，请基于提供的上下文回答问题。"},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        
        return response.choices[0].message.content
    
    def get_answer_with_context(self, question):
        """
        获取问题的答案和使用的上下文
        """
        # 检索相关文档
        docs = self.retriever(question, k=3)
        contexts = [doc.page_content for doc in docs]
        
        # 获取答案
        answer = self.get_answer(question)
        
        return answer, contexts