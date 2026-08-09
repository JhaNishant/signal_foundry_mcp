# Signal Foundry MCP

Signal Foundry watches competitor pricing pages so product teams can spend less time copying numbers between tabs. Ask a plain language question, let the MCP tools collect the source, and keep useful pricing plans in SQLite for the next question.

## What it does

* Scrapes pricing pages through Firecrawl and saves markdown and HTML.
* Records source details so saved pages can be found by provider, URL, or domain.
* Uses Claude through Vocareum to decide which MCP tool should run and to explain results clearly.
* Extracts supported pricing details into SQLite for fast follow up questions.
* Shows recent plans with `show data`.

## How the pieces work

```text
Question in terminal
        ↓
Claude client and MCP tool loop
        ↓
Scraper server  →  saved page files and metadata
        ↓
Pricing extractor  →  SQLite MCP server
        ↓
Clear comparison answer
```

Think of the system as a small research desk. The scraper gathers the page, the database keeps the useful notes, and Claude turns those notes into an answer.

## Setup

1. Install Python 3.11 or later, Node.js, and `uv`.
2. Copy `.env.example` to `.env`, then add the Vocareum key and Firecrawl key.
   The supplied defaults route Claude through Vocareum. Leave them unchanged unless
   Udacity provides a different endpoint or model.
3. Install dependencies.

```bash
uv sync --all-groups
```

4. Start the client.

```bash
uv run python starter_client.py
```

The client starts the custom scraper, SQLite, and filesystem MCP servers from `server_config.json`.

## Example prompts

```text
scrape these sites: {'cloudrift': 'https://www.cloudrift.ai/inference', 'deepinfra': 'https://deepinfra.com/pricing', 'fireworks': 'https://fireworks.ai/pricing#serverless-pricing', 'groq': 'https://groq.com/pricing'}

Compare cloudrift ai and deepinfra's costs for deepseek v3

show data
```

## Tests

```bash
uv run pytest
```

The test suite checks scraper persistence, retrieval, MCP retries, database writes, Claude tool use, terminal data display, and a mocked scrape to answer workflow.

## Limits

Pricing pages change often and providers may use different billing units. Signal Foundry reports only what a saved source supports and keeps the source URL with each scrape. Live runs need valid API keys and may use API credits.
