"""
Web search tool for the agentic loop (API process only).

Providers (pick one with env vars):
  TENSORLINK_SEARXNG_URL      base URL of a SearXNG instance with the JSON
                              format enabled (settings.yml: search.formats)
  TENSORLINK_BRAVE_API_KEY    Brave Search API key
  TENSORLINK_SEARCH_PROVIDER  optional "searxng" | "brave" to force one when
                              both are configured

Search only talks to the configured provider; it never fetches result URLs, so
there is no SSRF surface yet. A future fetch_url tool will need a guard.
"""

import html
import logging
import os
import re
import time
from collections import OrderedDict, deque
from typing import Dict, List, Optional, Tuple

import httpx

from tensorlink.api.tools.base import Tool, ToolResult

log = logging.getLogger(__name__)

SEARCH_TIMEOUT = float(os.environ.get("TENSORLINK_SEARCH_TIMEOUT", 8))
SEARCH_RPM = int(os.environ.get("TENSORLINK_SEARCH_RPM", 60))
CACHE_TTL = 300
CACHE_MAX = 256
MAX_RESULTS = 5
MAX_QUERY_CHARS = 300
TITLE_CHARS = 120
SNIPPET_CHARS = 300

_TAG_RE = re.compile(r"<[^>]+>")


class SearchError(Exception):
    """A user-presentable failure. Never carries URLs, keys or provider internals."""


def _clean(text: Optional[str], limit: int) -> str:
    text = html.unescape(_TAG_RE.sub("", text or ""))
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _normalize(raw: List[Tuple[str, str, str]], limit: int) -> List[Dict[str, str]]:
    """Clean, drop non-http(s) and duplicate URLs, cap the count."""
    seen, out = set(), []
    for title, url, snippet in raw:
        url = (url or "").strip()
        if not url.lower().startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        out.append(
            {
                "title": _clean(title, TITLE_CHARS) or url,
                "url": url,
                "snippet": _clean(snippet, SNIPPET_CHARS),
            }
        )
        if len(out) >= limit:
            break
    return out


class SearxngProvider:
    name = "searxng"

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def search(self, client: httpx.AsyncClient, query: str, limit: int):
        resp = await client.get(
            f"{self.base_url}/search",
            params={"q": query, "format": "json", "safesearch": 1},
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return _normalize(
            [
                (r.get("title"), r.get("url"), r.get("content"))
                for r in results
                if isinstance(r, dict)
            ],
            limit,
        )


class BraveProvider:
    name = "brave"
    URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def search(self, client: httpx.AsyncClient, query: str, limit: int):
        resp = await client.get(
            self.URL,
            params={"q": query, "count": limit},
            headers={
                "X-Subscription-Token": self.api_key,
                "Accept": "application/json",
            },
        )
        resp.raise_for_status()
        results = (resp.json().get("web") or {}).get("results", [])
        return _normalize(
            [
                (r.get("title"), r.get("url"), r.get("description"))
                for r in results
                if isinstance(r, dict)
            ],
            limit,
        )


def provider_from_env():
    name = os.environ.get("TENSORLINK_SEARCH_PROVIDER", "").strip().lower()
    searxng_url = os.environ.get("TENSORLINK_SEARXNG_URL", "").strip()
    brave_key = os.environ.get("TENSORLINK_BRAVE_API_KEY", "").strip()
    if name in ("", "searxng") and searxng_url:
        return SearxngProvider(searxng_url)
    if name in ("", "brave") and brave_key:
        return BraveProvider(brave_key)
    return None


class WebSearch:
    """Provider + TTL cache + sliding-window rate limit + lazy shared client."""

    def __init__(self, provider=None, client: Optional[httpx.AsyncClient] = None):
        self.provider = provider
        self._client = client
        self._cache: "OrderedDict[tuple, tuple]" = OrderedDict()
        self._calls: deque = deque()

    @property
    def configured(self) -> bool:
        return self.provider is not None

    def _http(self) -> httpx.AsyncClient:
        # Created lazily so it binds to the API server's event loop.
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=SEARCH_TIMEOUT,
                follow_redirects=False,
                headers={"User-Agent": "tensorlink-websearch/1.0"},
            )
        return self._client

    def _allow(self) -> bool:
        now = time.monotonic()
        while self._calls and now - self._calls[0] > 60:
            self._calls.popleft()
        if len(self._calls) >= SEARCH_RPM:
            return False
        self._calls.append(now)
        return True

    async def search(
        self, query: str, limit: int = MAX_RESULTS
    ) -> List[Dict[str, str]]:
        if not self.configured:
            raise SearchError("web search is not configured")

        key = (self.provider.name, query.lower(), limit)
        hit = self._cache.get(key)
        if hit and hit[0] > time.monotonic():
            self._cache.move_to_end(key)
            return hit[1]

        if not self._allow():
            raise SearchError("search rate limit reached, try again shortly")

        try:
            results = await self.provider.search(self._http(), query, limit)
        except (httpx.HTTPError, ValueError) as e:
            # Details (URL, status) go to the log, not to the model or client.
            log.warning("web search provider error (%s): %s", self.provider.name, e)
            raise SearchError("the search provider is unavailable") from e

        self._cache[key] = (time.monotonic() + CACHE_TTL, results)
        while len(self._cache) > CACHE_MAX:
            self._cache.popitem(last=False)
        return results

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def make_web_search_tool(search: WebSearch) -> Tool:
    async def run(arguments) -> ToolResult:
        query = arguments.get("query") if isinstance(arguments, dict) else None
        if not isinstance(query, str) or not query.strip():
            return ToolResult(error="missing 'query' argument")
        query = " ".join(query.split())[:MAX_QUERY_CHARS]
        try:
            return ToolResult(items=await search.search(query))
        except SearchError as e:
            return ToolResult(error=str(e))

    return Tool(
        name="web_search",
        signature="web_search(query)",
        description=(
            "search the web for current or factual information; returns titles, "
            "URLs and snippets"
        ),
        example_arguments={"query": "<standalone search query>"},
        run=run,
        default_arguments=lambda message: {"query": " ".join(message.split())[:200]},
    )
