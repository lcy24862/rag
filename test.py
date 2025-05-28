
from openai import OpenAI

client = OpenAI(api_key="sk-6151b5dae67941d8b5b17d323aae9fe6", base_url="https://api.deepseek.com/v1")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你好"},
    ],
    stream=False
)

print(response.choices[0].message.content)