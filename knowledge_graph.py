import spacy
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Tuple
import networkx as nx
import matplotlib.pyplot as plt
from config import CONFIG
from transformers import pipeline
from ltp import LTP # type: ignore

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

class KnowledgeGraphBuilder:
    def __init__(self):
        """
        初始化知识图谱构建器，加载 Chroma 数据库。
        :param persist_directory: Chroma 数据库的持久化目录路径。
        """
        self.nlp = spacy.load("zh_core_web_sm")
        self.graph = nx.Graph()  # 使用 NetworkX 存储知识图谱
        self.rebel_model = pipeline("text2text-generation", model="Babelscape/rebel-large")
        self.ltp = LTP()

    def extract_entities(self, text: str) -> List[Dict]:
        """
        从文本中提取实体。
        :param text: 输入的文本。
        :return: 实体列表，每个实体包含文本和标签。
        """
        doc = self.nlp(text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        return entities

    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """使用 LTP 提取中文关系"""
        # 分句（如果文本包含多句）
        sents = [text]  # 假设单句输入
        relations = []
        for sent in sents:
            # 语义角色标注 (SRL)
            srl_result = self.ltp.srl(sent)
            for predicate, args in srl_result.items():
                for arg in args:
                    relations.append((arg[0], predicate, arg[1]))  # (主体, 关系, 客体)
        return relations

    def build_graph_from_documents(self, k: int = 2):
        """
        从 Chroma 数据库中加载文档并构建知识图谱。
        :param k: 加载的文档数量。
        """
        docs = self.db.get()
        print("原始文本样例:", docs["documents"][0])  # 检查文本内容
        for doc in docs["documents"][:k]:
            entities = self.extract_entities(doc)
            relations = self.extract_relations(doc)

            # 将实体和关系添加到图中
            for entity in entities:
                self.graph.add_node(entity["text"], label=entity["label"])
            for rel in relations:
                self.graph.add_edge(rel[0], rel[2], label=rel[1])

    def visualize_graph(self):
        """可视化知识图谱"""
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_size=2000, font_size=10)
        edge_labels = nx.get_edge_attributes(self.graph, "label")
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.show()

if __name__ == "__main__":
    kg_builder = KnowledgeGraphBuilder()
    embeddings = HuggingFaceEmbeddings(model_name=CONFIG["embedding_model"])
    vector_store = Chroma(
        persist_directory=CONFIG["vector_store_path"],
        embedding_function=embeddings,
        collection_name="mechanical_rag"  # 与 initialize.py 中的名称一致
    )
    kg_builder.db = vector_store
    kg_builder.build_graph_from_documents(k=1)  # 从 Chroma 中加载 5 个文档构建图谱
    kg_builder.visualize_graph()  # 可视化图谱
