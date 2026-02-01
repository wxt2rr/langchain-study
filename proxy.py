import json
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

app = FastAPI()

# --- 配置区 ---
# 目标服务商的 API 基础地址（例如 OpenAI 或 DeepSeek）
TARGET_URL = "https://api.deepseek.com"


# 如果你想代理真正的 OpenAI，改为 https://api.openai.com
# --------------

@app.post("/{path:path}")
async def proxy_api(path: str, request: Request):
    # 1. 获取并解析请求头
    headers = dict(request.headers)
    # 移除 Host 头，避免目标服务器因 Host 不匹配拒绝请求
    headers.pop("host", None)

    # 2. 获取请求体并打印
    body = await request.json()
    print("\n" + "=" * 50)
    print(f"🔔 [拦截请求] 路径: /{path}")
    print(f"📦 [请求参数]:\n{json.dumps(body, indent=2, ensure_ascii=False)}")
    print("=" * 50)

    # 3. 转发请求到真实服务器
    async def send_request():
        async with httpx.AsyncClient() as client:
            # 判断是否是流式输出
            is_stream = body.get("stream", False)

            # 发送请求
            resp = await client.post(
                f"{TARGET_URL}/{path}",
                json=body,
                headers=headers,
                timeout=60.0
            )

            # 4. 处理并记录响应
            if not is_stream:
                # 非流式：直接打印并返回
                resp_json = resp.json()
                print("\n" + "*" * 50)
                print(f"✅ [收到响应]:\n{json.dumps(resp_json, indent=2, ensure_ascii=False)}")
                print("*" * 50)
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    headers=dict(resp.headers)
                )
            else:
                # 流式：实时打印并透传（流式打印稍微复杂一点）
                print("\n" + "*" * 50)
                print("🌊 [收到流式响应]: (内容将实时透传)")
                print("*" * 50)
                return StreamingResponse(
                    resp.aiter_bytes(),
                    status_code=resp.status_code,
                    headers=dict(resp.headers)
                )

    return await send_request()


if __name__ == "__main__":
    import uvicorn

    # 启动在本地 8000 端口
    uvicorn.run(app, host="127.0.0.1", port=8000)
