# src/rag_engine.py
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


class RAGEngine:
    def __init__(self, vector_store, api_key, base_url):
        self.vector_store = vector_store
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        
        # 创建检索器
        self.retriever = vector_store.as_retriever()  # 修改这里
        
        # 创建LLMChain
        prompt = PromptTemplate(
            input_variables=["context"],
            template="请压缩以下内容，仅保留与问题相关的信息：{context}"
        )
        llm_chain = LLMChain(llm=self._get_llm(), prompt=prompt)

        # 新写法
        chain = prompt | llm_chain
        result = chain.invoke({"context": "你的内容"})

        # 创建上下文压缩器
        compressor = LLMChainExtractor(llm_chain=llm_chain)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.retriever  # 这里传递检索器对象
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
        # 确保返回的是 langchain.llms.OpenAI 类型
        return OpenAI(model="text-davinci-003", temperature=0.7)
    
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
    
    def get_answer_with_context(self, question, k=3):
        docs = self.vector_store.similarity_search(question, k=k)
        contexts = [doc.page_content for doc in docs]
        context_str = "\n\n".join([f"Document {i+1}:\n{c}" for i, c in enumerate(contexts)])
        prompt = f"""请基于所提供的上下文回答问题。
如果上下文中不包含答案，请回答‘对不起，您所提供的上下文中不包含回答问题的信息。’
上下文：
{context_str}

问题：{question}"""
        answer = self.client.invoke(prompt)  # prompt 必须是字符串
        return answer, contexts