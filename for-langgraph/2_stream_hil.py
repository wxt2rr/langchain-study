# 在上节中我们已经学会了根据自己的业务场景定义实现简易的react模式的agent
# 但是对比我们常用的ChatGPT，Deepseek，豆包等API应用软件，发现有一些用户体验相关功能没有实现：
# 1、流式输出，我们当前的agent只有在node执行完进行流转时才会输出整段的内容，需要等很久才输出结果，体验不够好
# 2、人机环路，当涉及到危险的操作需要人进行审查在执行，现在是不具备的
# 3、持久化，当前agent一旦在过程中重启或者中断，那么整个状态就丢失了，只能重新来一遍

from typing import Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from langgraph.constants import START


# 使用interrupt函数基于人工输入的路由逻辑
class ApprovalState(TypedDict):
    topic:str
    proposed_action_details:str
    final_results:str

def propose_action(state:ApprovalState):
    return {
        "proposed_action_details": f"我将要进行 {state['topic']} 的操作"
    }

def human_action_node(state:ApprovalState) -> Command[Literal["yes_node", "no_node"]]:
    action_result = interrupt(
        {
            "question":"是否批准以下操作",
            "action_details": state["proposed_action_details"],
            "options": ["是", "否"]
        }
    )

    if action_result.get("user_action") == "是":
        return Command(goto="yes_node")
    else:
        return Command(goto="no_node")

def yes_node_action(state:ApprovalState):
    return {
        "final_results": "操作执行完成"
    }

def no_node_action(state:ApprovalState):
    return {
        "final_results": "操作被拒绝"
    }


graph_builder = StateGraph(ApprovalState)
graph_builder.add_node("propose_action", propose_action)
graph_builder.add_node("human_action_node", human_action_node)
graph_builder.add_node("yes_node", yes_node_action)
graph_builder.add_node("no_node", no_node_action)

graph_builder.add_edge(START, "propose_action")
graph_builder.add_edge("propose_action", "human_action_node")
graph_builder.add_edge("no_node", "human_action_node")

graph = graph_builder.compile(checkpointer=MemorySaver())

config = {
   "configurable": {"thread_id": "approval_thread"}
}

res = graph.invoke({"topic":"请假10天"}, config = config)
print(res)

# 模拟审核未通过
res = graph.invoke(Command(resume={"user_action":"否"}), config = config)
print(res)

# 模拟审核通过
res = graph.invoke(Command(resume={"user_action":"是"}), config = config)
print(res)




