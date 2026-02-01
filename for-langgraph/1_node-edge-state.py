# langgraph采用图的设计理念来实现智能体的工作流程，将复杂多变的agent流程通过抽象为节点（node）和边（edge）来表示
# 节点可以是“实体”的抽象，也可以是“行为”的抽象，比如我们可以将llm抽象为一个节点，也可以将tool抽象为一个节点
# 边可以理解为各个Node之间是否存在联系，联系的条件是什么，所以边分为普通边和条件边
# 状态可以理解为各个Node之间的数据传递媒介和结果存储媒介
# 通过node, edge, state的组合，可以实现各种复杂的agent流程，甚至可以无限套娃
import operator

# node 节点
# 可以理解为一个个独立的能力模块，每个模块实现一个功能，但是模块之间没有本质的联系，关注的是输入输出与内在能力

# edge 边（普通边和条件边）
# 将节点连接起来，形成一个完整的功能流程，关注的是节点之间的联系是什么，以及如何将这些联系组织起来

# state 状态
# 状态是节点在执行过程中的中间结果，关注的是节点执行过程中的中间结果是什么，以及如何保存和传递这些结果，相当于整个图的数据快照（运行态）

# command 命令
# 一般来说只有在node里边可以修改状态，通过edge边进行路由，如果你想同时修改状态以及设置跳转路由的话，可以使用command命令
## 不过注意的是，command命令只会覆盖条件边，如果node1 --普通边--> node2，这种仍然还是会走普通边

from langgraph.graph import StateGraph, MessagesState, START,END
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain.tools import tool
from pydantic import BaseModel
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode

load_dotenv()

# State状态：
# State是langgraph中对于数据运行时的一个抽象：
# 1、是短期记忆（即消息上下文）的存储媒介；
# 2、是node节点之间数据传递以及结果存储的媒介；
# 3、是状态机的状态，即状态机的当前状态；
# langgraph是基于状态作为数据流动，所以先创建一个状态，有两种方式:TypedDict（静态类型检查）或者Pydantic（动态类型检查）
class StateDemo(TypedDict):
    messages: list[AnyMessage]
    step: str


class StateDemo2(BaseModel):
    messages: list[AnyMessage]
    step: str


# Node本质是就是一个函数：输入state -> 执行函数逻辑 -> 输出state
# 1、LLM调用节点 调用大语言模型的Node
# 2、Tool调用节点 调用工具的Node
# 3、自定义调用节点 调用自定义逻辑的Node
# 这里我们先试用mock数据，假装调用了llm，并且llm返回了llm_node..，并且指定state的step为tool
def llm_node(state: StateDemo):
    state["messages"].append(AIMessage(content="llm_node.."))
    state["step"] = "tool"
    return state

def tool_node(state: StateDemo):
    state["messages"].append(AIMessage(content="tool_node"))
    state["step"] = "end"
    return state

# 我们现在有了node,state,那么就可以来创建第一个图了
# 创建一个图的构建器，需要传入状态
graph_builder = StateGraph(StateDemo)
# 声明node节点，添加node到图中
graph_builder.add_node("llm_node", llm_node)
graph_builder.add_node("tool_node", tool_node)

def router(state: StateDemo) -> str:
    return state["step"]

# 添加边，绑定或者叫关联node之间的联系
# 添加普通边，从START节点到llm_node节点，表示从START节点开始，调用llm_node节点
graph_builder.add_edge(START, "llm_node")
# 添加条件边，从llm_node节点到tool_node节点，表示从llm_node节点根据条件调用tool_node节点
graph_builder.add_conditional_edges("llm_node", router, {"llm": "llm_node", "tool": "tool_node", "end": "__end__"})
# 添加普通边，从tool_node节点到END节点，表示从tool_node节点结束
graph_builder.add_edge("tool_node", END)

# 编译图，生成一个可执行的图，compile会进行简单的静态校验，比如是否有游离的node,是否有类型错误等，编译结果才是一个可执行的图
graph = graph_builder.compile()
# 执行图，传入初始状态，返回最终状态
result = graph.invoke(input={"messages": [], "step": ""})
print(result)


# 好，了解了上边的流程之后，我们来真实的调用llm，实现一个react模型的agent
# 我们先创建一个llm，并且绑定上我们的工具

# 创建llm可以调用的函数
@tool
# 通过@tool装饰器，底层实现其实就是将函数注释转化为提示词，通过bind_tools函数将
# 函数与模型进行绑定，在真实调用llm时会自动将函数名和参数作为提示词传给llm，这样llm才知道要调用哪个函数
def getWeather(city: str) -> str:
    """获取天气
     :arg: city 城市
     :return: 天气情况
    """
    if city == "北京":
        return "北京天气晴，25度"
    elif city == "上海":
        return "上海天气多云，20度"
    else:
        return "其他城市天气情况未知"

system_prompt = """
    你是一个穿衣助手，擅长根据天气信息转换为具体、可执行的穿衣建议，服务对象是普通用户（非专业人士）
    
    你的核心目标：
    1、根据城市名称获取天气信息
    2、根据天气信息给出穿衣建议
    
    输出要求：
    1、必须根据用户输入进行回复，不能自己造轮子
    2、必须使用用户的语言进行回复
    3、不能使用夸张、情绪化的语气回复
    4、不要出现专业术语，使用简单易懂的语言进行回复
"""

# 初始化模型并将工具绑定到模型上，这里使用的是ChatOpenAI模型
# 像火山引擎，阿里百炼，DeepSeek，智普等平台都有免费额度，实在不行充个几块钱学习也值得
llm = ChatOpenAI(model="ark-code-latest", temperature=0)
llm = llm.bind_tools([getWeather])

# 我们可以先调用llm一下，看看效果
messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content="请给出穿衣建议，我在北京")
]
res = llm.invoke(messages)
# content='我来帮您获取天气信息并给出穿衣建议。首先让我获取您所在的城市名称。' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 23, 'prompt_tokens': 359, 'total_tokens': 382, 'completion_tokens_details': {'accepted_prediction_tokens': None, 'audio_tokens': None, 'reasoning_tokens': 0, 'rejected_prediction_tokens': None}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'glm-4-7', 'system_fingerprint': None, 'id': '02176991322575413c33c36521378d02b2c8389204fa8c39364f1', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019c170c-9097-7713-81ae-b3128445aaaa-0' tool_calls=[] invalid_tool_calls=[{'type': 'invalid_tool_call', 'id': 'call_0qbwslfbh74meh03n77ow6br', 'name': 'getCity', 'args': '{', 'error': 'Function getCity arguments:\n\n{\n\nare not valid JSON. Received JSONDecodeError Expecting property name enclosed in double quotes: line 1 column 2 (char 1)\nFor troubleshooting, visit: https://docs.langchain.com/oss/python/langchain/errors/OUTPUT_PARSING_FAILURE '}] usage_metadata={'input_tokens': 359, 'output_tokens': 23, 'total_tokens': 382, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}
print(res)
# 可以看到，llm已经成功调用，并且返回了结果，结果包含tool_call：getCity，说明llm想要调用getCity函数

# 好，我们现在已经有了llm和tool,接下来我们使用langgraph创建一个react模式的图
# 同样，我们先创建一个State
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]# langgraph建议不要直接在node中对state中进行修改
    # 而是通过Reducers进行修改
    current_step: str
    max_steps: int

# 创建一个llm node
def llm_node(state: AgentState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response], "current_step": "llm"}

# 定义tool node
tool_node = ToolNode([getWeather])

# 定义react模式的图
graph_builder = StateGraph(AgentState)
graph_builder.add_node("llm", llm_node)
graph_builder.add_node("tool", tool_node)
graph_builder.add_edge(START, "llm")

def router(state: AgentState) -> str:
    message = state['messages'][-1]
    print(f"当前消息内容: {message}")
    if message.tool_calls:
        return "tool"
    else:
        return "end"

# 这是react模式循环的关键：llm可以路由到tool和end -> toolNode进行反射函数调用，封装ToolMessage返回 -> 每次toolNode执行完必须回到llmNode,再次有llmNode
# 判断是否结束，这就达到了循环的效果
graph_builder.add_conditional_edges("llm", router, {"tool": "tool", "end": "__end__"})
graph_builder.add_edge("tool", "llm")
graph = graph_builder.compile()

messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content="请给出穿衣建议，我在上海")
]
result = graph.invoke(input={"messages": messages, "current_step": "", "max_steps": 10})
print(result)

#     START
#       |
#       v
#    +------+
#    | llm  | -----> END
#    +------+
#       |
#       | (根据router返回"tool")
#       v
#    +------+
#    | tool |
#    +------+
#       |
#       +-----> 回到llm节点形成循环






































