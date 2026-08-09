from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from starter_client import (
    VOCAREUM_CLAUDE_BASE_URL,
    VOCAREUM_CLAUDE_MODEL,
    ChatSession,
    DataExtractor,
    Server,
)


class FakeMcpResult:
    def __init__(self, text: str):
        self.content = [SimpleNamespace(text=text)]


class FakeSession:
    def __init__(self, tools: list[Any] | None = None, failures: int = 0):
        self.tools = tools or []
        self.failures = failures
        self.calls: list[dict[str, Any]] = []

    async def list_tools(self) -> Any:
        return SimpleNamespace(tools=self.tools)

    async def call_tool(self, **kwargs: Any) -> FakeMcpResult:
        self.calls.append(kwargs)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("temporary failure")
        return FakeMcpResult("ok")


class FakeServer:
    def __init__(self, tools: list[dict[str, Any]], results: dict[str, str] | None = None):
        self.tools = tools
        self.results = results or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[dict[str, Any]]:
        return self.tools

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> FakeMcpResult:
        self.calls.append((name, arguments))
        return FakeMcpResult(self.results.get(name, "[]"))


class FakeClaude:
    def __init__(self, responses: list[Any]):
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.messages = self

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_chat_session_uses_vocareum_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, Any] = {}

    class FakeAnthropicClient:
        def __init__(self, **kwargs: Any):
            created.update(kwargs)

    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.setattr("starter_client.AsyncAnthropic", FakeAnthropicClient)

    ChatSession(
        {"sqlite": FakeServer([]), "llm_inference": FakeServer([]), "filesystem": FakeServer([])}
    )

    assert created["base_url"] == VOCAREUM_CLAUDE_BASE_URL
    assert VOCAREUM_CLAUDE_MODEL == "claude-sonnet-4-5-20250929"


def test_chat_session_replaces_the_retired_course_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    chat = ChatSession(
        {"sqlite": FakeServer([]), "llm_inference": FakeServer([]), "filesystem": FakeServer([])},
        FakeClaude([]),
    )

    assert chat.model == VOCAREUM_CLAUDE_MODEL


def response(*blocks: Any) -> Any:
    return SimpleNamespace(content=list(blocks))


def text_block(text: str) -> Any:
    return SimpleNamespace(type="text", text=text)


def tool_block(name: str, tool_id: str, tool_input: dict[str, Any]) -> Any:
    return SimpleNamespace(type="tool_use", name=name, id=tool_id, input=tool_input)


@pytest.mark.asyncio
async def test_list_tools_requires_session_and_maps_schema() -> None:
    server = Server("scraper", {"command": "python", "args": ["server.py"]})
    with pytest.raises(RuntimeError, match="has not been initialized"):
        await server.list_tools()

    server.session = FakeSession(
        [SimpleNamespace(name="scrape_websites", description="Scrape pages", inputSchema={"type": "object"})]
    )  # type: ignore[assignment]
    assert await server.list_tools() == [
        {"name": "scrape_websites", "description": "Scrape pages", "input_schema": {"type": "object"}}
    ]


@pytest.mark.asyncio
async def test_execute_tool_retries_with_required_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    server = Server("sqlite", {"command": "npx", "args": ["sqlite"]})
    fake_session = FakeSession(failures=2)
    server.session = fake_session  # type: ignore[assignment]

    async def no_wait(_: float) -> None:
        return None

    monkeypatch.setattr("starter_client.asyncio.sleep", no_wait)
    result = await server.execute_tool("write_query", {"query": "SELECT 1"})

    assert result.content[0].text == "ok"
    assert len(fake_session.calls) == 3
    assert fake_session.calls[-1]["read_timeout_seconds"].total_seconds() == 60


@pytest.mark.asyncio
async def test_data_extractor_writes_every_plan() -> None:
    sqlite = FakeServer([], {"write_query": "ok"})
    pricing_json = json.dumps(
        {
            "company_name": "CloudRift",
            "plans": [
                {
                    "plan_name": "DeepSeek V3",
                    "input_tokens": 0.25,
                    "output_tokens": 0.9,
                    "currency": "USD",
                    "billing_period": "per million tokens",
                    "features": ["serverless"],
                    "limitations": "regional availability",
                }
            ],
        }
    )
    claude = FakeClaude([response(text_block(pricing_json))])
    extractor = DataExtractor(claude, sqlite, "test_model")  # type: ignore[arg-type]

    await extractor.extract_and_store_data("saved pricing page", "compare prices")

    assert len(sqlite.calls) == 1
    query = sqlite.calls[0][1]["query"]
    assert "INSERT INTO pricing_plans (company_name, plan_name, input_tokens, output_tokens" in query
    assert "json" not in query.lower() or "serverless" in query
    assert "CloudRift" in query and "DeepSeek V3" in query


@pytest.mark.asyncio
async def test_chat_session_handles_text_only_and_full_tool_loop() -> None:
    sqlite = FakeServer([], {"write_query": "ok"})
    scraper = FakeServer(
        [{"name": "extract_scraped_info", "description": "Read saved source", "input_schema": {}}],
        {"extract_scraped_info": '{"provider_name": "cloudrift", "content": {"markdown": "price"}}'},
    )
    filesystem = FakeServer([])
    claude = FakeClaude(
        [
            response(tool_block("extract_scraped_info", "tool_1", {"identifier": "cloudrift"})),
            response(text_block("CloudRift has a saved DeepSeek V3 price.")),
        ]
    )
    chat = ChatSession(
        {"sqlite": sqlite, "llm_inference": scraper, "filesystem": filesystem}, claude
    )  # type: ignore[arg-type]
    await chat.prepare_tools()

    answer = await chat.process_query("What does CloudRift charge?")

    assert answer == "CloudRift has a saved DeepSeek V3 price."
    assert scraper.calls == [("extract_scraped_info", {"identifier": "cloudrift"})]
    assert len(claude.calls) == 2
    tool_results = claude.calls[1]["messages"][-1]["content"]
    assert tool_results[0]["tool_use_id"] == "tool_1"


@pytest.mark.asyncio
async def test_chat_session_returns_a_text_only_completion() -> None:
    claude = FakeClaude([response(text_block("DeepInfra lists $0.32 per million input tokens."))])
    chat = ChatSession(
        {"sqlite": FakeServer([]), "llm_inference": FakeServer([]), "filesystem": FakeServer([])},
        claude,
    )  # type: ignore[arg-type]
    await chat.prepare_tools()

    answer = await chat.process_query("What does DeepInfra charge for DeepSeek V3?")

    assert answer == "DeepInfra lists $0.32 per million input tokens."
    assert len(claude.calls) == 1


@pytest.mark.asyncio
async def test_chat_session_reuses_saved_data_for_a_deepseek_comparison() -> None:
    rows = json.dumps(
        [
            {
                "company_name": "DeepInfra",
                "plan_name": "DeepSeek-V3",
                "input_tokens": 0.32,
                "output_tokens": 0.89,
                "currency": "USD",
                "billing_period": "per million tokens",
            }
        ]
    )
    sqlite = FakeServer([], {"read_query": rows, "write_query": "ok"})
    claude = FakeClaude([])
    chat = ChatSession(
        {"sqlite": sqlite, "llm_inference": FakeServer([]), "filesystem": FakeServer([])}, claude
    )  # type: ignore[arg-type]
    await chat.prepare_tools()

    answer = await chat.process_query("Compare CloudRift AI and DeepInfra's costs for DeepSeek V3")

    assert "CloudRift: no matching DeepSeek V3 price is stored." in answer
    assert "DeepInfra: USD $0.32 input and $0.89 output per million tokens." in answer
    assert len(claude.calls) == 0
    assert any(name == "read_query" for name, _ in sqlite.calls)


@pytest.mark.asyncio
async def test_mocked_scrape_to_database_to_answer_workflow() -> None:
    sqlite = FakeServer([], {"write_query": "ok"})
    scraper = FakeServer(
        [
            {"name": "scrape_websites", "description": "Scrape provider pages", "input_schema": {}},
            {"name": "extract_scraped_info", "description": "Read saved sources", "input_schema": {}},
        ],
        {
            "scrape_websites": "['deepinfra']",
            "extract_scraped_info": json.dumps(
                {
                    "provider_name": "deepinfra",
                    "content": {"markdown": "DeepSeek V3 input $0.32, output $0.89."},
                }
            ),
        },
    )
    pricing_data = json.dumps(
        {
            "company_name": "DeepInfra",
            "plans": [
                {
                    "plan_name": "DeepSeek V3",
                    "input_tokens": 0.32,
                    "output_tokens": 0.89,
                    "currency": "USD",
                    "billing_period": "per million tokens",
                    "features": ["serverless"],
                    "limitations": "None stated",
                }
            ],
        }
    )
    claude = FakeClaude(
        [
            response(
                tool_block(
                    "scrape_websites",
                    "tool_1",
                    {"websites": {"deepinfra": "https://deepinfra.com/pricing"}},
                )
            ),
            response(text_block(pricing_data)),
            response(text_block("DeepInfra charges $0.32 input and $0.89 output per million tokens.")),
        ]
    )
    chat = ChatSession(
        {"sqlite": sqlite, "llm_inference": scraper, "filesystem": FakeServer([])}, claude
    )  # type: ignore[arg-type]
    await chat.prepare_tools()

    answer = await chat.process_query("Scrape DeepInfra, then find the DeepSeek V3 price.")

    assert answer == "DeepInfra charges $0.32 input and $0.89 output per million tokens."
    assert [name for name, _ in scraper.calls] == ["scrape_websites", "extract_scraped_info"]
    write_queries = [arguments["query"] for name, arguments in sqlite.calls if name == "write_query"]
    assert any("DeepInfra" in query and "DeepSeek V3" in query for query in write_queries)
    assert len(claude.calls) == 3


@pytest.mark.asyncio
async def test_show_stored_data_prints_required_rows(capsys: pytest.CaptureFixture[str]) -> None:
    rows = "[{'company_name': 'DeepInfra', 'plan_name': 'DeepSeek V3', " \
           "'input_tokens': 0.27, 'output_tokens': 1.1, 'currency': 'USD'}]"
    sqlite = FakeServer([], {"read_query": rows, "write_query": "ok"})
    chat = ChatSession(
        {"sqlite": sqlite, "llm_inference": FakeServer([]), "filesystem": FakeServer([])},
        FakeClaude([]),
    )  # type: ignore[arg-type]

    await chat.show_stored_data()

    output = capsys.readouterr().out
    assert "Recently Stored Data:" in output
    assert "DeepInfra: DeepSeek V3" in output
    assert output.rstrip().endswith("=" * 50)
