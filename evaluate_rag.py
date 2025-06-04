from src.rag_processor import load_or_generate_dataset
from src.evaluator import evaluate_dataset

def main():
    # 加载或生成数据集（优先使用缓存）
    dataset = load_or_generate_dataset(use_cache=True)
    
    # 评估数据集
    results = evaluate_dataset(dataset)
    print(results)

if __name__ == "__main__":
    main()
