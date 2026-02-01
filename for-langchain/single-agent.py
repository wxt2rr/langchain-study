# for-langchain/for-langgraph sdk的本质就是将 native-code-agent 中核心的流程通过组件进行了封装
# 使用langchain开发agent的核心流程仍然是 定义系统提示词、配置llm模型、维护messages、封装工具
# 我们仍然先以一个简单的 react agent 说明
# 思考 -> 系统提示词（SystemMessage）   行动 -> 调用工具（AssistantMessage）  观察 -> 查看工具结果（ToolMessage）    在思考 ->  系统提示词 + 上下文（SystemMessage + AssistantMessage + ToolMessage）

from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.messages import SystemMessage, HumanMessage
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()


# 实现单次调用的 agent
@tool
def getWeather(city: str) -> str:
    """获取天气
     :arg: city 城市
    """
    return "天气晴，25度"


@tool
def getCity() -> str:
    """获取用户所在的城市名称
    :return: 城市名称
    """
    return "北京"


llm = ChatOpenAI(model="ark-code-latest", api_key=os.getenv("OPENAI_API_KEY"),
                 base_url="https://ark.cn-beijing.volces.com/api/coding/v3")

agent = llm.bind_tools([getWeather, getCity])

system_prompt = """
你是一个天气查询助手，首先要分析用户的意图，如果需要调用工具，则调用工具，否则直接返回结果。
"""

messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "今天天气这么样"}]

# ('content', '我来帮您查询今天的天气。首先让我获取您所在的城市信息。\n\n')
# ('additional_kwargs', {'refusal': None})
# ('response_metadata', {'token_usage': {'completion_tokens': 41, 'prompt_tokens': 369, 'total_tokens': 410, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 0, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'deepseek-v3-2', 'system_fingerprint': None, 'id': '0217688078281839c33e9c339b2fbf9ef6ce2d7c2f130ab6db576', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None})
# ('type', 'ai')
# ('name', None)
# ('id', 'lc_run--019bd529-8a5f-7aa1-8b65-1d13abd09fec-0')
# ('tool_calls', [{'name': 'getCity', 'args': {}, 'id': 'call_2ntdk0216xz3el92hcu4tlj2', 'type': 'tool_call'}])
# ('invalid_tool_calls', [])
# ('usage_metadata', {'input_tokens': 369, 'output_tokens': 41, 'total_tokens': 410, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}})
response = agent.invoke(input=messages)
for chunk in response:
    print(chunk)

# 直接使用封装好的 create_react_agent
react_agent = create_agent(llm, tools=[getWeather, getCity], system_prompt=system_prompt)

messages = {"messages": [HumanMessage("今天天气怎么样")]}

response = react_agent.invoke(messages)
for chunk in response:
    print(chunk)
