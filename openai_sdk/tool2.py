import os

from openai import OpenAI
import json

api_key = os.environ.get("DEEPSEEK_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

tools = [
    {
        "type": "function",
        "name": "get_horoscope",
        "description": "Get today's horoscope for an astrological sign.",
        "parameters": {
            "type": "object",
            "properties": {
                "sign": {
                    "type": "string",
                    "description": "An astrological sign like Taurus or Aquarius",
                },
            },
            "required": ["sign"],
        },
    },
]


def get_horoscope(sign):
    return f"{sign}: Next Tuesday you will befriend a baby otter."


input_list = [{"role": "user", "content": "What is my horoscope? I am an Aquarius."}]
while True:
    response = client.responses.create(
        model="deepseek-v4-pro",
        tools=tools,
        input=input_list,
    )

    input_list.extend(item.model_dump(exclude_none=True) for item in response.output)
    print("\nCurrent input:" + json.dumps(input_list, indent=2))

    tool_calls = []
    for item in response.output:
        if item.type == "function_call":
            tool_calls.append(item)
    if not tool_calls:
        break

    for item in tool_calls:
        if item.name == "get_horoscope":
            sign = json.loads(item.arguments)["sign"]
            horoscope = get_horoscope(sign)
            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": horoscope,
                }
            )


print("Final output:")
print(response.model_dump_json(indent=2))
print("\n" + response.output_text)
