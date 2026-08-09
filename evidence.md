# Signal Foundry MCP evidence

The following terminal transcripts were captured from the live run on August 9, 2026.
They contain the required queries and actual application output. No keys are included.

## 1. Scraping run

```text
Query: scrape these sites: {'cloudrift': 'https://www.cloudrift.ai/inference', 'deepinfra': 'https://deepinfra.com/pricing', 'fireworks': 'https://fireworks.ai/pricing#serverless-pricing', 'groq': 'https://groq.com/pricing'}

Running scrape_websites...
Successfully scraped 4 out of 4 websites

Extracting structured pricing for cloudrift...
Saved pricing plans for cloudrift.
Extracting structured pricing for fireworks...
Saved pricing plans for fireworks.
Extracting structured pricing for groq...
Saved pricing plans for groq.

Done. I've successfully scraped all four pricing pages:
- cloudrift
- deepinfra
- fireworks
- groq
```

## 2. Pricing comparison

```text
Query: Compare cloudrift ai and deepinfra's costs for deepseek v3

Based on the pricing information from both providers, here's the comparison for DeepSeek V3:

CloudRift AI:
- The saved source does not show DeepSeek V3 token pricing. CloudRift is a GPU
  infrastructure platform rather than a direct model inference API provider.

DeepInfra:
- DeepSeek V3: $0.32 per 1M input tokens and $0.89 per 1M output tokens.
- DeepSeek V3 0324: $0.24 per 1M input tokens and $0.90 per 1M output tokens.

Summary: the saved sources do not support a direct token price comparison because
CloudRift lists GPU infrastructure, while DeepInfra lists serverless model pricing.
```

## 3. Stored database data

```text
Query: show data

Recently Stored Data:
==================================================

Pricing Plans:
  • DeepInfra: DeepSeek V3 USD Input Token $0.32, Output Tokens $0.89
  • CloudRift: RTX 4090 USD Input Token $0.0, Output Tokens $0.0
  • CloudRift: A100 80GB USD Input Token $0.0, Output Tokens $0.0
  • CloudRift: V100 32GB USD Input Token $0.0, Output Tokens $0.0
  • CloudRift: RTX 5090 USD Input Token $0.0, Output Tokens $0.0
==================================================
```
