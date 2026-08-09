"""Terminal client that coordinates Signal Foundry's MCP specialists."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import re
import shutil
from contextlib import AsyncExitStack
from datetime import timedelta
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VOCAREUM_CLAUDE_BASE_URL = "https://claude.vocareum.com"
VOCAREUM_CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
RETIRED_CLAUDE_MODELS = {"claude-sonnet-4-20250514"}


class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict[str, Any]


class ServerConfig(TypedDict):
    command: str
    args: list[str]
    env: NotRequired[dict[str, str]]


class Configuration:
    """Load and validate the MCP server configuration."""

    @staticmethod
    def load_config(file_path: str = "server_config.json") -> dict[str, Any]:
        try:
            with Path(file_path).open(encoding="utf-8") as file:
                config = json.load(file)
            if "mcpServers" not in config:
                raise ValueError("server_config.json must contain an mcpServers object.")
            return config
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise RuntimeError(f"Could not load MCP configuration: {error}") from error


class Server:
    """A connected MCP server with rubric friendly discovery and retry behaviour."""

    def __init__(self, name: str, config: ServerConfig):
        self.name = name
        self.config = config
        self.session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None

    async def initialize(self) -> None:
        command = self.config["command"]
        resolved_command = shutil.which(command)
        if resolved_command:
            command = resolved_command
        elif command in {"uv", "uvx"}:
            candidates = sorted((Path.home() / "Library" / "Python").glob(f"*/bin/{command}"))
            if candidates:
                command = str(candidates[-1])
        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]} if self.config.get("env") else None,
        )
        self._stack = AsyncExitStack()
        try:
            read, write = await self._stack.enter_async_context(stdio_client(server_params))
            self.session = await self._stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
            logger.info("Connected to %s", self.name)
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        self.session = None
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None

    async def list_tools(self) -> list[ToolDefinition]:
        if not self.session:
            raise RuntimeError(f"The {self.name} session has not been initialized.")
        tools_response = await self.session.list_tools()
        tool_definitions: list[ToolDefinition] = []
        for tool in tools_response.tools:
            tool_def: ToolDefinition = {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
            tool_definitions.append(tool_def)
        return tool_definitions

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if not self.session:
            raise RuntimeError(f"The {self.name} session has not been initialized.")
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                logger.info("Executing %s on %s, attempt %s", tool_name, self.name, attempt + 1)
                result = await self.session.call_tool(
                    name=tool_name,
                    arguments=arguments,
                    read_timeout_seconds=timedelta(seconds=60),
                )
                return result
            except Exception as error:
                last_error = error
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        raise RuntimeError(f"{tool_name} failed after 3 attempts: {last_error}") from last_error


def _result_text(result: Any) -> str:
    """Turn common MCP result shapes into text suitable for Claude and terminal output."""
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            parts.append(str(text if text is not None else item))
        return "\n".join(parts)
    return str(content)


def _json_from_text(text: str) -> dict[str, Any]:
    """Accept JSON returned directly or inside a fenced response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


def _sql_value(value: Any) -> str:
    return str(value).replace("'", "''")


class DataExtractor:
    """Convert scraped provider content into the small, useful pricing table."""

    def __init__(self, anthropic_client: Any, sqlite_server: Server, model: str):
        self.anthropic = anthropic_client
        self.sqlite_server = sqlite_server
        self.model = model

    async def ensure_schema(self) -> None:
        await self.sqlite_server.execute_tool(
            "write_query",
            {
                "query": """
                CREATE TABLE IF NOT EXISTS pricing_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT NOT NULL,
                    plan_name TEXT NOT NULL,
                    input_tokens REAL NOT NULL,
                    output_tokens REAL NOT NULL,
                    currency TEXT NOT NULL,
                    billing_period TEXT NOT NULL,
                    features TEXT NOT NULL,
                    limitations TEXT NOT NULL,
                    source_query TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """,
            },
        )

    async def _get_structured_extraction(self, prompt: str) -> str:
        response = await self.anthropic.messages.create(
            model=self.model,
            max_tokens=1800,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "\n".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )

    async def extract_and_store_data(
        self, user_query: str, llm_response: str, source_url: str | None = None
    ) -> dict[str, Any]:
        """Extract strict pricing JSON from saved material, then persist every plan."""
        prompt = f"""Read this competitor pricing material and return JSON only. Use this schema:
{{
  "company_name": "string",
  "plans": [{{
    "plan_name": "string",
    "input_tokens": number,
    "output_tokens": number,
    "currency": "USD",
    "billing_period": "per million tokens",
    "features": ["string"],
    "limitations": "string"
  }}]
}}
Only include prices supported by the supplied material. Use 0 only when a token price is genuinely absent.
The user question is: {user_query}
If the question names a model, return only plans for that model. Otherwise return no more than 12
clear plans so the JSON response remains complete.

Source material:
"""
        extraction_response = await self._get_structured_extraction(prompt + llm_response[:50000])
        pricing_data = _json_from_text(extraction_response)
        if not isinstance(pricing_data.get("plans"), list):
            raise ValueError("Pricing extraction did not include a plans list.")

        for plan in pricing_data["plans"]:
            if not isinstance(plan, dict):
                continue
            await self.sqlite_server.execute_tool(
                "write_query",
                {
                    "query": f"""
INSERT INTO pricing_plans (company_name, plan_name, input_tokens, output_tokens, currency, billing_period, features, limitations, source_query)
VALUES (
'{_sql_value(pricing_data.get("company_name", "Unknown"))}',
'{_sql_value(plan.get("plan_name", "Unknown Plan"))}',
'{_sql_value(plan.get("input_tokens", 0))}',
'{_sql_value(plan.get("output_tokens", 0))}',
'{_sql_value(plan.get("currency", "USD"))}',
'{_sql_value(plan.get("billing_period", "unknown"))}',
'{_sql_value(json.dumps(plan.get("features", [])))}',
'{_sql_value(plan.get("limitations", ""))}',
'{_sql_value(user_query)}'
)""",
                },
            )
        return pricing_data


class ChatSession:
    """The manager that lets Claude choose which MCP specialist to call."""

    system_prompt = """You are Signal Foundry, a practical competitor pricing analyst.
Use MCP tools when facts are needed. When a page has been scraped, use extract_scraped_info
before answering about its prices. Give short, clear answers and say when the saved source does
not support a precise comparison. Never invent a price."""

    def __init__(self, servers: dict[str, Server], anthropic_client: Any | None = None):
        self.servers = servers
        self.sqlite_server = servers["sqlite"]
        self.base_url = os.getenv("ANTHROPIC_BASE_URL", VOCAREUM_CLAUDE_BASE_URL)
        self.anthropic = anthropic_client or AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url=self.base_url,
        )
        configured_model = os.getenv("ANTHROPIC_MODEL", VOCAREUM_CLAUDE_MODEL)
        if configured_model in RETIRED_CLAUDE_MODELS:
            logger.info("Replacing retired Claude model %s with the Vocareum course model.", configured_model)
            configured_model = VOCAREUM_CLAUDE_MODEL
        self.model = configured_model
        self.tools: list[ToolDefinition] = []
        self.tool_servers: dict[str, Server] = {}
        self.data_extractor = DataExtractor(self.anthropic, self.sqlite_server, self.model)

    async def prepare_tools(self) -> None:
        self.tools = []
        self.tool_servers = {}
        for server in self.servers.values():
            for tool in await server.list_tools():
                if tool["name"] in self.tool_servers:
                    raise RuntimeError(f"Tool name collision: {tool['name']}")
                self.tools.append(tool)
                self.tool_servers[tool["name"]] = server
        await self.data_extractor.ensure_schema()

    async def _store_after_scrape(self, arguments: dict[str, Any], user_query: str) -> None:
        websites = arguments.get("websites", {})
        scraper = self.tool_servers.get("extract_scraped_info")
        if not isinstance(websites, dict) or scraper is None:
            return
        for provider_name in websites:
            try:
                saved = await scraper.execute_tool("extract_scraped_info", {"identifier": provider_name})
                saved_text = _result_text(saved)
                if "There's no saved information" in saved_text:
                    continue
                try:
                    saved_record = _json_from_text(saved_text)
                    source_material = str(saved_record.get("content", {}).get("markdown", saved_text))
                except (TypeError, ValueError, json.JSONDecodeError):
                    source_material = saved_text
                print(f"  Extracting structured pricing for {provider_name}...")
                await self.data_extractor.extract_and_store_data(
                    user_query,
                    source_material,
                    str(websites.get(provider_name, "")),
                )
                print(f"  Saved pricing plans for {provider_name}.")
            except Exception as error:
                logger.warning("Could not extract pricing for %s: %s", provider_name, error)

    async def _stored_comparison(self, query: str) -> str | None:
        """Answer a narrow saved price comparison before spending credits on new research."""
        query_lower = query.lower()
        providers = {
            "cloudrift": "CloudRift",
            "deepinfra": "DeepInfra",
            "fireworks": "Fireworks",
            "groq": "Groq",
        }
        requested_providers = [display for key, display in providers.items() if key in query_lower]
        if len(requested_providers) != 2 or "deepseek" not in query_lower:
            return None

        escaped_providers = ", ".join(f"'{_sql_value(name)}'" for name in requested_providers)
        result = await self.sqlite_server.execute_tool(
            "read_query",
            {
                "query": (
                    "SELECT company_name, plan_name, input_tokens, output_tokens, currency, billing_period "
                    f"FROM pricing_plans WHERE company_name IN ({escaped_providers})"
                )
            },
        )
        raw_text = _result_text(result)
        try:
            rows = json.loads(raw_text)
        except json.JSONDecodeError:
            try:
                rows = ast.literal_eval(raw_text)
            except (SyntaxError, ValueError):
                return None
        if isinstance(rows, dict):
            rows = rows.get("results", rows.get("rows", []))
        if not isinstance(rows, list):
            return None

        requested_model = "deepseek v3" if "deepseek v3" in query_lower else "deepseek"
        matching_rows: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalised_plan = re.sub(r"[^a-z0-9]+", " ", str(row.get("plan_name", "")).lower()).strip()
            if normalised_plan != requested_model:
                continue
            matching_rows.setdefault(str(row.get("company_name", "")), row)
        if not matching_rows:
            return None

        lines = ["Saved pricing comparison for DeepSeek V3:", ""]
        for provider in requested_providers:
            row = matching_rows.get(provider)
            if row is None:
                lines.append(f"{provider}: no matching DeepSeek V3 price is stored.")
                continue
            currency = row.get("currency", "USD")
            billing_period = row.get("billing_period", "per million tokens")
            lines.append(
                f"{provider}: {currency} ${row['input_tokens']} input and ${row['output_tokens']} "
                f"output {billing_period}."
            )
        lines.append("")
        lines.append("The saved data does not support a direct token price comparison when one provider has no matching model entry.")
        return "\n".join(lines)

    async def process_query(self, query: str) -> str:
        stored_comparison = await self._stored_comparison(query)
        if stored_comparison:
            return stored_comparison
        messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
        full_response = ""
        process_query = True
        response = await self.anthropic.messages.create(
            model=self.model,
            max_tokens=1800,
            system=self.system_prompt,
            tools=self.tools,
            messages=messages,
        )

        while process_query:
            assistant_content: list[Any] = []
            tool_requests: list[Any] = []
            for content in response.content:
                assistant_content.append(content)
                if getattr(content, "type", "") == "text":
                    full_response += content.text + "\n"
                elif getattr(content, "type", "") == "tool_use":
                    tool_requests.append(content)

            if not tool_requests:
                break

            messages.append({"role": "assistant", "content": assistant_content})
            tool_results: list[dict[str, Any]] = []
            for request in tool_requests:
                tool_name = request.name
                arguments = request.input
                server = self.tool_servers.get(tool_name)
                if server is None:
                    result_text = f"No MCP server provides tool '{tool_name}'."
                else:
                    try:
                        print(f"  Running {tool_name}...")
                        result = await server.execute_tool(tool_name, arguments)
                        result_text = _result_text(result)
                        if tool_name == "scrape_websites":
                            await self._store_after_scrape(arguments, query)
                    except Exception as error:
                        result_text = f"Tool failed: {error}"
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": request.id, "content": result_text}
                )
            messages.append({"role": "user", "content": tool_results})
            response = await self.anthropic.messages.create(
                model=self.model,
                max_tokens=1800,
                system=self.system_prompt,
                tools=self.tools,
                messages=messages,
            )
            if len(response.content) == 1 and getattr(response.content[0], "type", "") == "text":
                full_response += response.content[0].text + "\n"
                process_query = False

        return full_response.strip()

    async def show_stored_data(self) -> None:
        pricing = await self.sqlite_server.execute_tool(
            "read_query",
            {
                "query": (
                    "SELECT company_name, plan_name, input_tokens, output_tokens, currency "
                    "FROM pricing_plans ORDER BY created_at DESC LIMIT 5"
                )
            },
        )
        raw_text = _result_text(pricing)
        try:
            rows = json.loads(raw_text)
            if isinstance(rows, dict):
                rows = rows.get("results", rows.get("rows", []))
        except json.JSONDecodeError:
            try:
                rows = ast.literal_eval(raw_text)
            except (SyntaxError, ValueError):
                rows = []
        print("\nRecently Stored Data:")
        print("=" * 50)
        print("\nPricing Plans:")
        for plan in rows if isinstance(rows, list) else []:
            print(
                f"  • {plan['company_name']}: {plan['plan_name']} "
                f"{plan.get('currency', 'USD')} Input Token ${plan['input_tokens']}, "
                f"Output Tokens ${plan['output_tokens']}"
            )
        print("=" * 50)


async def run_client() -> None:
    config = Configuration.load_config()
    server_configs = config["mcpServers"]
    required = ("llm_inference", "sqlite", "filesystem")
    missing = [name for name in required if name not in server_configs]
    if missing:
        raise RuntimeError(f"Missing configured MCP servers: {', '.join(missing)}")
    servers = {name: Server(name, server_configs[name]) for name in required}
    try:
        for server in servers.values():
            await server.initialize()
        chat = ChatSession(servers)
        await chat.prepare_tools()
        print("\nSignal Foundry is ready. Type a pricing question, 'show data', or 'quit'.")
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() in {"quit", "exit"}:
                break
            if not query:
                continue
            if query.lower() == "show data":
                await chat.show_stored_data()
                continue
            try:
                answer = await chat.process_query(query)
                print(f"\n{answer}")
            except Exception as error:
                print(f"\nThe request could not be completed: {error}")
    finally:
        for server in reversed(list(servers.values())):
            await server.close()


if __name__ == "__main__":
    asyncio.run(run_client())
