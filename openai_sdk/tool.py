from openai import APIConnectionError, APITimeoutError, OpenAI
import json
import os

tools = [
    {
        "type": "function",
        "name": "get_current_time",
        "description": "Get the current date and time in a specific timezone",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone name, e.g. America/Vancouver",
                }
            },
            "required": ["timezone"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a specific city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
            "additionalProperties": False,
        },
    },
]


messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Use tools to get real-time information when needed.",
    },
    {"role": "user", "content": "What's the current time and weather in Vancouver?"},
]


def execute_tool(tool_name, tool_args):
    if tool_name == "get_current_time":
        # mock
        return '{"datetime": "2025-09-13T05:18:47", "day_of_week": "Saturday"}'
    elif tool_name == "get_weather":
        # mock
        return '{"temperature": 13.2, "unit": "celsius", "conditions": "clear", "humidity": 93}'
    else:
        return "Tool not found."


def run_conversation(client):
    while True:
        response = client.responses.create(
            model="deepseek-v4-pro", input=messages, tools=tools
        )
        # print(response.to_json())  # 打印响应
        messages.extend(item.model_dump(exclude_none=True) for item in response.output)

        tool_calls = [item for item in response.output if item.type == "function_call"]
        if not tool_calls:
            return

        for tool_call in tool_calls:
            tool_args = json.loads(tool_call.arguments)
            tool_result = execute_tool(tool_call.name, tool_args)
            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": tool_result,
                }
            )

        print(json.dumps(messages))  # 打印更新后的消息
        print("--------------------------------------------------")


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("请先设置环境变量 DEEPSEEK_API_KEY")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    try:
        run_conversation(client)
    except APITimeoutError:
        raise SystemExit("连接 DeepSeek API 超时，请检查网络或代理设置") from None
    except APIConnectionError as exc:
        raise SystemExit(f"无法连接 DeepSeek API：{exc}") from None


if __name__ == "__main__":
    main()
