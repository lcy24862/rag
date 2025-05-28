import json

# 读取 JSON 文件
with open('evaluation_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 提取问题和答案
sample_queries = [item["question"] for item in data]
expected_responses = [item["answer"] for item in data]

# 生成数据集
dataset = []

for query, reference in zip(sample_queries, expected_responses):
    dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": [],  # 这里假设需要填充检索到的上下文
            "response": "",  # 这里假设需要填充生成的回答
            "reference": reference
        }
    )

# 输出或保存数据集
print(json.dumps(dataset, indent=2, ensure_ascii=False))

# 如果需要保存到文件
with open('formatted_dataset.json', 'w', encoding='utf-8') as outfile:
    json.dump(dataset, outfile, indent=2, ensure_ascii=False)