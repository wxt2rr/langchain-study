from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate

# 创建agent的三要素：模型、系统提示词、工具列表

load_dotenv(override=True)

tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

tools = [tavily_search_tool]

model = init_chat_model(model_provider="deepseek", model="deepseek-chat")

if __name__ == '__main__':
    agent = create_agent(model, tools,
                         system_prompt="你是一名助人为乐的助手，并且可以调用工具进行网络搜索，获取实时信息。")
    
    res = agent.invoke({"messages": [{"role": "user", "content": "今天有什么新闻"}]})
    print(res)
