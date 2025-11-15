from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
import sys

print(f"!!! IDE 正在使用的 Python 路径是: {sys.executable}")
from langchain.output_parsers import StructuredOutputParser

# 加载 .env 文件
load_dotenv(override=True)

model = init_chat_model(model="deepseek-chat", model_provider="deepseek")


def use_model_and_print_result():
    chain = model | StrOutputParser()

    question = "你是谁"
    result = chain.invoke(question)
    print(result)


def use_model_and_set_system_prompt():
    prompt_template = ChatPromptTemplate([
        ("system", "你是一个乐意助人的助手，请根据用户的问题给出回答"),
        ("user", "这是用户的问题： {topic}， 请用 yes 或 no 来回答")
    ])

    chain = prompt_template | model | StrOutputParser()
    question = "1 + 1 = 2 吗"
    result = chain.invoke(question)
    print(result)


def mutil_chain():
    prompt = PromptTemplate.format_prompt("请根据一下内容生成一首诗：\n\n标题: {name}")
    chain_1 = prompt | model

    parser = output_parser = StructuredOutputParser.from_response_schemas([
        {"name": "title", "description": "The title of the text"},
        {"name": "summary", "description": "A short summary"}
    ])

    summary_prompt = PromptTemplate.from_template(
        "请从下面这段新闻内容中提取关键信息，并返回结构化JSON格式：\n\n{aa}\n\n{bb}"
    )

    chain_2 = summary_prompt.partial(format_instructions=parser.get_format_instructions()) | model | parser

    full_chain = chain_1 | chain_2
    res = full_chain.invoke({"a": "大海"})
    print(res)


if __name__ == '__main__':
    # use_model_and_print_result()
    # use_model_and_set_system_prompt()
    mutil_chain()
