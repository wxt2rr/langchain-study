import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(override=True)


def dirct_use_model():
    DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=DeepSeek_API_KEY, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是王小涛开发的语言助手，根据用户的问题分析给出答案"},
            {"role": "user", "content": "你是谁"}
        ]
    )

    print(response.choices[0].message.content)


def langchain_sdk_use_model():
    # langchain会自动从env文件中获取指定模型提供者的key配置
    model = init_chat_model(model="deepseek-chat", model_provider="deepseek")
    question = "你好，你是谁"

    result = model.invoke(question)
    print(result.content)


if __name__ == '__main__':
    # 直接使用open sdk 链接模型
    # dirct_use_model()

    langchain_sdk_use_model()
