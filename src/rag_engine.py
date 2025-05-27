# src/rag_engine.py
from openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

class RAGEngine:
    def __init__(self, vector_store, openai_api_key, base_url):
        self.vector_store = vector_store
        self.client = OpenAI(api_key=openai_api_key, base_url=base_url)
        self.retriever = vector_store.as_retriever()
    
    def get_answer_with_context(self, question, k=3):
        """获取答案及参考上下文"""
        docs = self.vector_store.similarity_search(question, k=k)
        contexts = [doc.page_content for doc in docs]
        
        # 改用字符串拼接避免f-string中的反斜杠
        user_content = (
            "上下文：\n" +
            "\n".join(contexts) +
            "\n\n问题：" +
            question
        )
        
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个有帮助的AI助手，请基于提供的上下文回答问题。"},
                {"role": "user", "content": user_content}
            ],
            stream=False
        )
        
        answer = response.choices[0].message.content
        return answer, contexts