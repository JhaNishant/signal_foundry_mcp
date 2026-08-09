from __future__ import annotations

import json
from pathlib import Path

import pytest

import starter_server as server


class FakeScrapeResult:
    def __init__(self, payload: dict):
        self.payload = payload

    def model_dump(self) -> dict:
        return self.payload


class FakeFirecrawl:
    def __init__(self, responses: dict[str, object]):
        self.responses = responses
        self.calls: list[str] = []

    def scrape(self, url: str, formats: list[str]) -> FakeScrapeResult:
        self.calls.append(url)
        response = self.responses[url]
        if isinstance(response, Exception):
            raise response
        return FakeScrapeResult(response)  # type: ignore[arg-type]


@pytest.fixture()
def isolated_storage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(server, "SCRAPE_DIR", tmp_path / "scraped_content")
    monkeypatch.setattr(server, "METADATA_FILE", tmp_path / "scraped_metadata.json")
    monkeypatch.setattr(server, "CACHE_TTL_HOURS", 24)
    monkeypatch.setattr(server.time, "sleep", lambda _: None)
    monkeypatch.setattr(server, "_firecrawl_client", None)


def test_scrape_persists_each_format_and_metadata(
    isolated_storage: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    cloudrift = "https://www.cloudrift.ai/inference"
    deepinfra = "https://deepinfra.com/pricing"
    fake = FakeFirecrawl(
        {
            cloudrift: {
                "success": True,
                "markdown": "# CloudRift pricing",
                "html": "<h1>CloudRift pricing</h1>",
                "metadata": {"title": "CloudRift", "description": "GPU pricing"},
            },
            deepinfra: {
                "success": True,
                "markdown": "# DeepInfra pricing",
                "html": "<h1>DeepInfra pricing</h1>",
                "metadata": {"title": "DeepInfra", "description": "Model pricing"},
            },
        }
    )
    monkeypatch.setattr(server, "_firecrawl_client", fake)

    successful = server.scrape_websites_impl({"cloudrift": cloudrift, "deepinfra": deepinfra})

    assert successful == ["cloudrift", "deepinfra"]
    metadata = json.loads(server.METADATA_FILE.read_text())
    assert set(metadata) == {"cloudrift", "deepinfra"}
    for provider, url in (("cloudrift", cloudrift), ("deepinfra", deepinfra)):
        record = metadata[provider]
        assert record["provider_name"] == provider
        assert record["url"] == url
        assert record["domain"] == url.split("/")[2]
        assert record["scraped_at"]
        assert record["title"]
        assert record["description"]
        assert set(record["content_files"]) == {"markdown", "html"}
        for filename in record["content_files"].values():
            assert (server.SCRAPE_DIR / filename).is_file()

    server.scrape_websites_impl({"cloudrift": cloudrift})
    assert fake.calls.count(cloudrift) == 1


def test_scrape_failure_does_not_block_other_sites(
    isolated_storage: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    failing_url = "https://example.com/fails"
    working_url = "https://example.com/works"
    fake = FakeFirecrawl(
        {
            failing_url: RuntimeError("temporary upstream failure"),
            working_url: {
                "success": True,
                "markdown": "working page",
                "html": "<p>working page</p>",
                "metadata": {},
            },
        }
    )
    monkeypatch.setattr(server, "_firecrawl_client", fake)

    successful = server.scrape_websites_impl({"fails": failing_url, "works": working_url})

    assert successful == ["works"]
    metadata = json.loads(server.METADATA_FILE.read_text())
    assert set(metadata) == {"works"}
    assert fake.calls.count(failing_url) == 3


@pytest.mark.parametrize("identifier", ["cloudrift", "https://www.cloudrift.ai/inference", "www.cloudrift.ai"])
def test_extract_matches_name_url_and_domain(
    isolated_storage: None, monkeypatch: pytest.MonkeyPatch, identifier: str
) -> None:
    monkeypatch.setattr(
        server,
        "_firecrawl_client",
        FakeFirecrawl(
            {
                "https://www.cloudrift.ai/inference": {
                    "success": True,
                    "markdown": "DeepSeek V3 input cost",
                    "html": "<p>DeepSeek V3 input cost</p>",
                    "metadata": {},
                }
            }
        ),
    )
    server.scrape_websites_impl({"cloudrift": "https://www.cloudrift.ai/inference"})

    extracted = json.loads(server.extract_scraped_info_impl(identifier))

    assert extracted["provider_name"] == "cloudrift"
    assert extracted["content"]["markdown"] == "DeepSeek V3 input cost"
    assert "DeepSeek V3" in extracted["content"]["html"]


def test_extract_missing_identifier_returns_plain_message(isolated_storage: None) -> None:
    assert server.extract_scraped_info_impl("unknown") == (
        "There's no saved information related to identifier 'unknown'."
    )
