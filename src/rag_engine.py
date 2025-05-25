# src/rag_engine.py
from langchain.llms import ChatGLM
from langchain.chains import RetrievalQA

class RAGEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.llm = ChatGLM(
            model_name="chatglm3-6b",
            temperature=0.7
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
    
    def get_answer(self, question):
        """
        获取问题的答案
        """
        return self.qa_chain.run(question)