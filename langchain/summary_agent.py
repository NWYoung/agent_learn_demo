import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemPermission
from deepagents.middleware.unsupported_content import UnsupportedContentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.structured_output import ProviderStrategy
from langchain.tools import tool
from langchain_core.utils.uuid import uuid7
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

MODEL_NAME = "deepseek-v4-pro"
MODEL_BASE_URL = "https://api.deepseek.com"
PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = "/test.txt"
SUMMARY_FILE = "/backend/test_summary.json"
SYSTEM_PROMPT = """你是一名文本摘要助手。你的任务是准确、清晰地概括用户提供的文本，保留原文的核心主题、主要观点、关键事实和结论。

## 工作要求

- 用户直接提供文本时，直接基于该文本进行总结。
- 用户提供 URL 或要求总结网页内容时，先调用 `fetch_text_from_url` 获取原文，再总结工具返回的内容。
- 用户要求总结本地文件时，先调用 `read_file` 读取文件，再仅依据文件内容总结。
- 用户要求保存摘要时，使用 `write_file` 写入指定文件；只有写入成功后才能确认。
- 不得编造原文没有提到的信息、事实、数据或观点；遇到缺失、含糊或无法获取的内容时，明确说明。
- 默认使用中文回答；若用户指定其他语言，则按用户要求输出。
- 默认先给出一段简明摘要；文本较长或内容复杂时，再用要点列出关键内容。
- 用户指定摘要长度、受众或格式时，优先遵循这些要求。
- 区分原文事实与作者观点，不要把推测写成事实。
- 尽量控制在 100 字以内；内容复杂时可适当延长，但仍需简明扼要。"""


class Answer(BaseModel):
    """结构化的摘要结果。"""

    summary: str = Field(description="忠于原文的简明摘要")
    confidence: float = Field(
        ge=0,
        le=1,
        description="摘要准确性的置信度，范围为 0 到 1",
    )


@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch and return the text content from a URL."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw_text = response.read()
    except urllib.error.URLError as error:
        return f"Fetch failed: {error}"

    return raw_text.decode("utf-8", errors="replace")


def create_model() -> ChatOpenAI:
    """Create the DeepSeek chat model using the Responses API."""
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url=MODEL_BASE_URL,
        use_responses_api=True,
        output_version="responses/v1",
        reasoning={"effort": "high"},
    )


def create_filesystem_middleware() -> FilesystemMiddleware:
    """Expose read/write tools scoped to the demo's input and output files."""
    backend = FilesystemBackend(root_dir=PROJECT_DIR, virtual_mode=True)
    permissions = [
        FilesystemPermission(operations=["read"], paths=[INPUT_FILE]),
        FilesystemPermission(operations=["write"], paths=[SUMMARY_FILE]),
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/**"],
            mode="deny",
        ),
    ]
    return FilesystemMiddleware(
        backend=backend,
        tools=["read_file", "write_file"],
        permissions=permissions,
    )


def create_summary_agent(model: ChatOpenAI):
    """Build the summarization agent with filesystem access and write approval."""
    return create_agent(
        model=model,
        tools=[fetch_text_from_url],
        system_prompt=SYSTEM_PROMPT,
        response_format=ProviderStrategy(Answer),
        checkpointer=InMemorySaver(),
        middleware=[
            ModelRetryMiddleware(max_retries=3),
            ToolRetryMiddleware(max_retries=2),
            create_filesystem_middleware(),
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "write_file": {
                        "allowed_decisions": ["approve", "reject"],
                        "description": (
                            "请确认是否将摘要写入 backend/test_summary.json。"
                        ),
                    }
                }
            ),
            UnsupportedContentMiddleware(),
        ],
    )


def invoke_summary(agent: Any, prompt: str, config: dict[str, Any]) -> dict[str, Any]:
    """Send one user turn to the agent and return its state."""
    return agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
    )


def print_structured_response(result: dict[str, Any]) -> None:
    """Print an agent response as readable JSON."""
    answer = result["structured_response"]
    print(answer)


def run_web_rounds(agent: Any, config: dict[str, Any]) -> None:
    """Run the optional URL summary and follow-up rounds."""
    round1 = invoke_summary(
        agent,
        "总结 https://www.thepaper.cn/newsDetail_forward_34502009 的内容",
        config,
    )
    print_structured_response(round1)

    round2 = invoke_summary(agent, "其中提到的出土文物数量如何？", config)
    print_structured_response(round2)


def print_pending_write(interrupts: tuple[Any, ...]) -> None:
    """Show the tool call that is waiting for human approval."""
    print("等待人工确认的文件写入操作:")
    for interrupt in interrupts:
        request = getattr(interrupt, "value", interrupt)
        if hasattr(request, "model_dump"):
            request = request.model_dump(mode="json")
        print(json.dumps(request, ensure_ascii=False, indent=2, default=str))


def request_write_approval(
    agent: Any,
    result: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Ask for approval and resume the agent only when the write is approved."""
    interrupts = result.get("__interrupt__", ())
    if not interrupts:
        return result

    print_pending_write(interrupts)
    approval = (
        input("批准写入 backend/test_summary.json 吗？输入 y 批准，其他输入拒绝: ")
        .strip()
        .lower()
    )
    if approval not in {"y", "yes", "是"}:
        print("已拒绝写入；文件不会被创建或修改。")
        return None

    return agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,
    )


def run_local_file_round(agent: Any, config: dict[str, Any]) -> None:
    """Summarize test.txt, request approval, and save the JSON result."""
    result = invoke_summary(
        agent,
        (
            f"请调用 read_file 读取 {INPUT_FILE}，仅根据文件内容提炼摘要。"
            "然后调用 write_file 把 summary 和 confidence 写成 JSON，"
            f"保存到 {SUMMARY_FILE}。写入成功后返回 Answer 结构。"
        ),
        config,
    )
    result = request_write_approval(agent, result, config)
    if result is None:
        return

    answer = result.get("structured_response")
    if answer is None:
        interrupt_info = result.get("__interrupt__")
        raise RuntimeError(
            "round3 没有生成结构化摘要。"
            f"中断信息: {interrupt_info!r}; 返回字段: {list(result.keys())}"
        )

    write_results = [
        message
        for message in result.get("messages", [])
        if getattr(message, "name", None) == "write_file"
    ]
    if not write_results or getattr(write_results[-1], "status", None) == "error":
        raise RuntimeError("round3 摘要没有成功写入 backend/test_summary.json。")

    output_path = PROJECT_DIR / SUMMARY_FILE.lstrip("/")
    print(
        json.dumps(
            {
                "round": 3,
                "saved_to": str(output_path),
                "summary": answer.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    """Run the local-file summarization demo."""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY。")
    agent = create_summary_agent(create_model())
    config = {"configurable": {"thread_id": str(uuid7())}}
    run_web_rounds(agent, config)
    run_local_file_round(agent, config)


if __name__ == "__main__":
    main()
