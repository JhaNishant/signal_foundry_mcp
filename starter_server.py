"""Signal Foundry's custom MCP scraper server."""

import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

try:
    from firecrawl import Firecrawl
except ImportError:  # Older course environments expose this class name.
    Firecrawl = None  # type: ignore[assignment,misc]

try:
    from firecrawl import FirecrawlApp
except ImportError:
    FirecrawlApp = None  # type: ignore[assignment,misc]

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRAPE_DIR = Path(os.getenv("SCRAPE_DIR", "scraped_content"))
METADATA_FILE = Path(os.getenv("SCRAPED_METADATA_FILE", str(SCRAPE_DIR / "scraped_metadata.json")))
CACHE_TTL_HOURS = int(os.getenv("SCRAPE_CACHE_TTL_HOURS", "24"))
SCRAPE_FORMATS = ("markdown", "html")

mcp = FastMCP("Signal Foundry Scraper")
_firecrawl_client: Any | None = None


def _load_metadata() -> dict[str, dict[str, Any]]:
    """Load previously saved provider records without treating a blank file as an error."""
    try:
        raw = METADATA_FILE.read_text(encoding="utf-8").strip()
        return json.loads(raw) if raw else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_metadata(metadata: dict[str, dict[str, Any]]) -> None:
    METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def _get_firecrawl_client(api_key: str | None = None) -> Any:
    global _firecrawl_client
    if _firecrawl_client is not None:
        return _firecrawl_client

    api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise RuntimeError("FIRECRAWL_API_KEY is missing. Add it to the local .env file.")
    if Firecrawl is not None:
        _firecrawl_client = Firecrawl(api_key=api_key)
    elif FirecrawlApp is not None:
        _firecrawl_client = FirecrawlApp(api_key=api_key)
    else:
        raise RuntimeError("The firecrawl package is not installed.")
    return _firecrawl_client


def _as_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        result = result.model_dump()
    elif hasattr(result, "dict"):
        result = result.dict()
    if not isinstance(result, dict):
        raise TypeError("Firecrawl returned an unsupported response.")
    data = result.get("data")
    return data if isinstance(data, dict) and any(key in data for key in SCRAPE_FORMATS) else result


def _is_fresh(metadata: dict[str, Any], url: str) -> bool:
    if metadata.get("url") != url:
        return False
    try:
        scraped_at = datetime.fromisoformat(str(metadata["scraped_at"]))
        if scraped_at.tzinfo is None:
            scraped_at = scraped_at.replace(tzinfo=UTC)
    except (KeyError, TypeError, ValueError):
        return False
    if scraped_at < datetime.now(UTC) - timedelta(hours=CACHE_TTL_HOURS):
        return False
    content_files = metadata.get("content_files", {})
    return bool(content_files) and all((SCRAPE_DIR / name).is_file() for name in content_files.values())


def _content_from_result(result: dict[str, Any], format_type: str) -> str:
    value = result.get(format_type, "")
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return str(value or "")


def scrape_websites_impl(
    websites: dict[str, str],
    formats: list[str] | None = None,
    api_key: str | None = None,
    force: bool = False,
) -> list[str]:
    """Scrape provider pages, preserve their content, and track every successful result."""
    SCRAPE_DIR.mkdir(parents=True, exist_ok=True)
    scraped_metadata = _load_metadata()
    successful_scrapes: list[str] = []
    app = _get_firecrawl_client(api_key)
    requested_formats = tuple(formats or SCRAPE_FORMATS)

    for provider_name, url in websites.items():
        existing = scraped_metadata.get(provider_name, {})
        if not force and _is_fresh(existing, url):
            logger.info("Using fresh saved content for %s", provider_name)
            successful_scrapes.append(provider_name)
            continue

        logger.info("Scraping %s: %s", provider_name, url)
        for attempt in range(3):
            try:
                scrape_result = _as_dict(app.scrape(url, formats=list(requested_formats)))
                success = scrape_result.get("success", True)
                if success is False:
                    raise RuntimeError(str(scrape_result.get("error", "Firecrawl reported a failed scrape.")))

                content_files: dict[str, str] = {}
                for format_type in requested_formats:
                    content = _content_from_result(scrape_result, format_type)
                    if not content:
                        continue
                    filename = f"{provider_name}_{format_type}.txt"
                    (SCRAPE_DIR / filename).write_text(content, encoding="utf-8")
                    content_files[format_type] = filename

                if not content_files:
                    raise RuntimeError("Firecrawl returned no markdown or HTML content.")

                page_metadata = scrape_result.get("metadata", {})
                if not isinstance(page_metadata, dict):
                    page_metadata = {}
                scraped_metadata[provider_name] = {
                    "provider_name": provider_name,
                    "url": url,
                    "domain": urlparse(url).netloc.lower(),
                    "scraped_at": datetime.now(UTC).isoformat(),
                    "formats": list(content_files),
                    "success": "true",
                    "content_files": content_files,
                    "title": page_metadata.get("title", scrape_result.get("title", "")),
                    "description": page_metadata.get(
                        "description", scrape_result.get("description", "")
                    ),
                }
                successful_scrapes.append(provider_name)
                break
            except Exception as error:  # Firecrawl errors must not stop the remaining providers.
                if attempt == 2:
                    logger.warning("Could not scrape %s: %s", provider_name, error)
                else:
                    delay = 2**attempt
                    logger.info("Retrying %s in %s second(s)", provider_name, delay)
                    time.sleep(delay)

    _write_metadata(scraped_metadata)
    logger.info("Successfully scraped %s out of %s websites", len(successful_scrapes), len(websites))
    return successful_scrapes


def extract_scraped_info_impl(identifier: str) -> str:
    """Return saved provider metadata and every available content format."""
    try:
        scraped_metadata = _load_metadata()
        needle = identifier.strip().lower()
        for provider_name, metadata in scraped_metadata.items():
            candidates = (provider_name, metadata.get("url", ""), metadata.get("domain", ""))
            if needle not in {str(candidate).lower() for candidate in candidates}:
                continue

            result = metadata.copy()
            result["content"] = {}
            for format_type, filename in metadata.get("content_files", {}).items():
                path = SCRAPE_DIR / filename
                if path.is_file():
                    result["content"][format_type] = path.read_text(encoding="utf-8")
            return json.dumps(result, indent=2, ensure_ascii=False)
    except (OSError, TypeError, ValueError) as error:
        logger.warning("Could not load saved content for %s: %s", identifier, error)
    return f"There's no saved information related to identifier '{identifier}'."


@mcp.tool()
def scrape_websites(
    websites: dict[str, str],
    formats: list[str] = ["markdown", "html"],
    api_key: Optional[str] = None,
    force: bool = False,
) -> list[str]:
    """Scrape provider URLs and save markdown and HTML for later analysis."""
    return scrape_websites_impl(websites, formats, api_key, force)


@mcp.tool()
def extract_scraped_info(identifier: str) -> str:
    """Load saved content using a provider name, URL, or domain."""
    return extract_scraped_info_impl(identifier)


if __name__ == "__main__":
    mcp.run()
