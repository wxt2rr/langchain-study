import json
from typing import List

import requests

# 单agent示例，核心组成 LLM（思考能力） + TOOLS（干活能力）
# React模式： 思考 -> 行动 -> 观察 -> 思考，其实核心思想就是先想，想的差不多了就干，干中在根据实际情况不断纠正，直到问题解决
# Plan-Execute模式：思考 -> 计划 -> 执行 -> 结束

# 使用自定义的方式完成一个react模式的agent
# 核心问题：agent的职责与能力边界
# 1、系统提示词：规定agent、角色、职责、能力、背景信息
# 2、如何进行工具调用（规定llm返回格式）：本质还是通过llm进行”思考“，llm不可能直接进行工具（函数）调用，所以本质是规范llm返回的格式，程序根据llm的返回格式进行解析，进行真实的函数调用
# 3、如何与用户交互：一种是闷头干活，干完回复一个总结；一种是边干边说，调工具之前先说明调用工具的原因和目的，最后给总结

def get_address() -> str:
    """
    获取地址
    :param city:城市
    :return:地址
    """
    return '地址是北京'


# 普通函数
def get_weather(city: str) -> str:
    """
    获取天气
    :param city:城市
    :return:天气情况
    """
    return '天气晴，温度25度'


TOOL_FUN_MAP = {
    'get_weather': get_weather,
    'get_address': get_address,
}


def invoke_agent() -> List[dict[str, str]]:
    system_prompt = """
    你是贾维斯，一个天气智能助理，可以查天气，查地址
    
    你需要根据用户的问题，理解用户问题，判断是否需要调用工具，如果需要调用工具，则调用工具，否则直接返回结果
    
    你可以使用的工具：
    1. get_weather: 获取天气，参数：city
    2. get_address: 获取当前所在城市
    
    返回格式要求：任何情况下都需要以以下格式返回，不要添加其它符号，也不要将自己的思考直接输出，必须以以下结构返回给用户：
    {
        "thought": "思考过程，分析用户问题和解决方案",
        "action": {
            "name": "工具名称",
            "args": "工具参数"
            "result": "调用结果"
        },
        "final_answer": "最终答案（当不需要调用工具或问题解决时提供）",
    }
    """
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': '当地天气如何'},
    ]

    # loop循环，直到问题解决
    max_loop_count = 4
    current_loop_count = 0
    while True:
        if current_loop_count > max_loop_count:
            print("循环次数超过最大值，退出循环")
            break

        llm_call_result = call_llm(system_prompt, messages)
        print(f'模型返回结果 {llm_call_result}')
        if llm_call_result is None:
            print("模型返回结果为空，退出循环")
            break

        response = json.loads(llm_call_result)

        if 'final_answer' in response and response['final_answer']:
            messages.append({'role': 'assistant', 'content': response['final_answer']})
            break

        if 'action' in response and response['action']:
            result = execute_tool(response['action'])
            observation = {'action': {'name': response['action']['name'], 'result': result}}
            messages.append({'role': 'assistant', 'content': json.dumps(response)})
            messages.append({'role': 'assistant', 'content': json.dumps(observation)})

        if 'thought' in response and response['thought']:
            messages.append({'role': 'assistant', 'content': response['thought']})

        current_loop_count += 1

    return messages


# agent核心方法
def execute_tool(tool_data: dict):
    tool_name = tool_data.get('name')
    tool_args = tool_data.get('args')

    # 使用反射调用工具
    tool_method = TOOL_FUN_MAP[tool_name]
    if tool_args:
        if isinstance(tool_args, dict):
            result = tool_method(**tool_args)
        else:
            result = tool_method(tool_args)
    else:
        result = tool_method()
    return result


def call_llm(system_prompt: str, messages: List[dict[str, str]]) -> str:
    # 直接通过http调用llm
    base_url = 'http://127.0.0.1:8045/v1'
    api_key = 'claude-sonnet-4-5-thinking'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer {}'.format(api_key),
    }

    payload = {
        'system_prompt': system_prompt,
        'messages': messages,
        'model': 'claude-sonnet-4-5',
        "thinking": {
            "type": "enabled"
        }
    }

    print(f'llm模型请求参数: {payload}')
    response = requests.post(base_url + "/chat/completions", headers=headers, json=payload)
    print(f'llm模型请求结果: {response.text}')

    message = json.loads(response.content)['choices'][0]['message']['content']
    return message


if __name__ == '__main__':
    invoke_agent()
