from config import CONFIG
from openai import OpenAI

client = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你好"},
    ],
    stream=False
)

print(response.choices[0].message.content)