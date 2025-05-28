from typing import List, Dict
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas import EvaluationDataset

class RAGEvaluator:
    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
    
    def evaluate(self, dataset: EvaluationDataset) -> Dict[str, float]:
        """
        使用 Ragas 评估 RAG 系统性能
        
        Args:
            dataset: 包含问题、标准答案、系统答案和上下文的 EvaluationDataset 对象
        
        Returns:
            包含各项指标得分的字典
        """
        return evaluate(
            dataset=dataset,
            metrics=self.metrics,
        )
    
    def evaluate_batch(self,
                      questions: List[str],
                      ground_truths: List[str],
                      answers: List[str],
                      contexts: List[List[str]]) -> Dict[str, float]:
        """
        批量评估 RAG 系统性能
        
        Args:
            questions: 问题列表
            ground_truths: 标准答案列表
            answers: 系统生成的答案列表
            contexts: 检索到的上下文列表
        
        Returns:
            包含各项指标得分的字典
        """
        data = {
            "question": questions,
            "ground_truth": ground_truths,
            "answer": answers,
            "contexts": contexts,
        }
        dataset = EvaluationDataset.from_dict(data)
        return self.evaluate(dataset) 