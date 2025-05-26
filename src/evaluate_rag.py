from evaluation import RAGEvaluator
from rag_engine import RAGEngine
from config import CONFIG

def main():
    # 初始化评估器
    evaluator = RAGEvaluator()
    
    # 示例评估数据
    test_questions = [
        "什么是机器学习？",
        "深度学习与传统机器学习有什么区别？",
        "什么是神经网络？"
    ]
    
    ground_truths = [
        "机器学习是人工智能的一个分支，它使计算机系统能够从数据中学习和改进，而无需明确编程。",
        "深度学习是机器学习的一个子集，它使用多层神经网络来学习数据的层次表示。与传统机器学习相比，深度学习能够自动学习特征，不需要手动特征工程。",
        "神经网络是一种模仿人脑神经元结构的计算模型，由多个相互连接的节点（神经元）组成，用于处理和学习复杂的数据模式。"
    ]
    
    # 初始化RAG引擎
    rag_engine = RAGEngine(CONFIG["vector_store_path"])
    
    # 获取系统答案和上下文
    answers = []
    contexts = []
    
    for question in test_questions:
        answer, context = rag_engine.get_answer_with_context(question)
        answers.append(answer)
        contexts.append(context)
    
    # 执行评估
    results = evaluator.evaluate_batch(
        questions=test_questions,
        ground_truths=ground_truths,
        answers=answers,
        contexts=contexts
    )
    
    # 打印评估结果
    print("\n评估结果：")
    print("-" * 50)
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    main() 