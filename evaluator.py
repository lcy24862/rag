from ragas import evaluate
from langchain_openai import ChatOpenAI
from config import CONFIG
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas import EvaluationDataset

def evaluate_dataset(dataset):
    """评估数据集"""


    evaluation_dataset = EvaluationDataset.from_list(dataset)
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        api_key=CONFIG["api_key"],
        base_url=CONFIG["base_url"]
    )
    result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],llm=llm)
    return result