# src/rag_engine.py
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

class RAGEngine:
    def __init__(self, vector_store, openai_api_key, base_url):
        self.vector_store = vector_store
        self.client = OpenAI(api_key=openai_api_key, base_url=base_url)
        
        # 初始化检索器和压缩器
        self.retriever = vector_store.as_retriever()
        self._init_compression_retriever()
    
    def _init_compression_retriever(self):
        """初始化上下文压缩检索器"""
        prompt = PromptTemplate(
            input_variables=["context"],
            template="压缩内容，保留关键信息：{context}"
        )
        llm_chain = LLMChain(
            llm=OpenAI(
                model="deepseek-chat",
                temperature=0.7,
                api_key=self.client.api_key
            ),
            prompt=prompt
        )
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=LLMChainExtractor(llm_chain=llm_chain),
            base_retriever=self.retriever
        )
    
    def get_answer_with_context(self, question, k=3):
        """获取答案及参考上下文"""
        docs = self.vector_store.similarity_search(question, k=k)
        contexts = [doc.page_content for doc in docs]
        
        # 修复：避免在 f-string 中使用反斜杠
        prompt = (
            "基于以下上下文回答问题：\n" +
            "\n".join(contexts) +
            "\n\n问题：" +
            question
        )
        answer = self.client.invoke(prompt)
        return answer, contexts