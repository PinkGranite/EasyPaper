"""
Paper Search Tool for academic literature retrieval.

Supports Google Scholar (SerpAPI), Semantic Scholar, and arXiv
for searching academic papers and exporting BibTeX entries.
"""

import asyncio
import re
import time as _time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import httpx

from .base import WriterTool, ToolResult


class SemanticScholarClient:
    """
    Async client for the Semantic Scholar Academic Graph API.

    - **Description**:
        - Searches for papers by query string.
        - Returns structured paper data with BibTeX generation.
        - Rate limits: 100 requests/5min (anonymous), higher with API key.
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    FIELDS = "paperId,title,authors,year,abstract,venue,citationCount,externalIds,publicationTypes,journal"

    # Maximum number of retries on 429 (rate limit)
    MAX_RETRIES = 3
    # Base wait time in seconds (exponential backoff: 2s, 4s, 8s)
    RETRY_BASE_WAIT = 2.0

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self.api_key = api_key
        self.timeout = timeout
        self._rate_limited = False  # Set to True when 429 is hit

    @property
    def is_rate_limited(self) -> bool:
        """Whether the last request was rate-limited (429)."""
        return self._rate_limited

    async def search(
        self,
        query: str,
        max_results: int = 5,
        year_range: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for papers on Semantic Scholar with retry on 429.

        - **Description**:
            - On HTTP 429, retries up to MAX_RETRIES times with exponential
              backoff (2s, 4s, 8s). Sets is_rate_limited flag so callers
              can fall back to arXiv.

        - **Args**:
            - `query` (str): Search query string.
            - `max_results` (int): Maximum number of results.
            - `year_range` (str, optional): Year range filter, e.g. "2020-2025".

        - **Returns**:
            - `List[dict]`: List of paper dicts with standardized fields.
        """
        self._rate_limited = False
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": self.FIELDS,
        }
        if year_range:
            params["year"] = year_range

        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                    response = await client.get(
                        f"{self.BASE_URL}/paper/search",
                        params=params,
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()

                # Success
                papers = []
                for item in data.get("data", []):
                    paper = self._normalize_paper(item)
                    if paper:
                        papers.append(paper)
                return papers

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 429 and attempt < self.MAX_RETRIES:
                    wait = self.RETRY_BASE_WAIT * (2 ** (attempt - 1))
                    print(f"[SemanticScholar] 429 rate limited, retry {attempt}/{self.MAX_RETRIES} "
                          f"in {wait:.0f}s...")
                    await asyncio.sleep(wait)
                    last_error = e
                    continue
                else:
                    if status == 429:
                        self._rate_limited = True
                        print(f"[SemanticScholar] 429 rate limited, all {self.MAX_RETRIES} retries exhausted")
                    else:
                        print(f"[SemanticScholar] HTTP error: {status}")
                    return []
            except httpx.TimeoutException:
                print("[SemanticScholar] Request timed out")
                return []
            except Exception as e:
                print(f"[SemanticScholar] Error: {e}")
                return []

        # Should not reach here, but just in case
        print(f"[SemanticScholar] All retries failed: {last_error}")
        self._rate_limited = True
        return []

    def _normalize_paper(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize a Semantic Scholar paper result to standard format."""
        title = item.get("title")
        if not title:
            return None

        authors_list = item.get("authors", [])
        authors = [a.get("name", "") for a in authors_list if a.get("name")]

        year = item.get("year")
        external_ids = item.get("externalIds", {}) or {}
        doi = external_ids.get("DOI")
        arxiv_id = external_ids.get("ArXiv")
        journal_info = item.get("journal", {}) or {}

        paper = {
            "source": "semantic_scholar",
            "paper_id": item.get("paperId", ""),
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": item.get("abstract", ""),
            "venue": item.get("venue", "") or journal_info.get("name", ""),
            "citation_count": item.get("citationCount", 0),
            "doi": doi,
            "arxiv_id": arxiv_id,
            "bibtex_key": self._generate_bibtex_key(authors, year, title),
        }

        paper["bibtex"] = self._generate_bibtex(paper)
        return paper

    def _generate_bibtex_key(
        self,
        authors: List[str],
        year: Optional[int],
        title: str,
    ) -> str:
        """Generate a BibTeX citation key from paper metadata."""
        # Use first author's last name
        if authors:
            first_author = authors[0]
            # Extract last name (handle "First Last" and "Last, First")
            if "," in first_author:
                last_name = first_author.split(",")[0].strip()
            else:
                parts = first_author.split()
                last_name = parts[-1] if parts else "unknown"
        else:
            last_name = "unknown"

        # Clean last name for BibTeX key
        last_name = re.sub(r'[^a-zA-Z]', '', last_name).lower()

        year_str = str(year) if year else "nd"

        # First significant word from title
        title_words = [w for w in title.split() if len(w) > 3 and w.isalpha()]
        title_word = title_words[0].lower() if title_words else "paper"

        return f"{last_name}{year_str}{title_word}"

    def _generate_bibtex(self, paper: Dict[str, Any]) -> str:
        """Generate a BibTeX entry string from paper data."""
        key = paper["bibtex_key"]
        title = paper.get("title", "")
        authors = " and ".join(paper.get("authors", []))
        year = paper.get("year", "")
        venue = paper.get("venue", "")
        doi = paper.get("doi", "")
        arxiv_id = paper.get("arxiv_id", "")

        # Determine entry type
        entry_type = "article"
        if venue and any(kw in venue.lower() for kw in ["conference", "proceedings", "workshop", "symposium"]):
            entry_type = "inproceedings"
        elif arxiv_id and not venue:
            entry_type = "article"

        lines = [f"@{entry_type}{{{key},"]
        lines.append(f"  title = {{{title}}},")
        if authors:
            lines.append(f"  author = {{{authors}}},")
        if year:
            lines.append(f"  year = {{{year}}},")

        if entry_type == "inproceedings" and venue:
            lines.append(f"  booktitle = {{{venue}}},")
        elif venue:
            lines.append(f"  journal = {{{venue}}},")

        if doi:
            lines.append(f"  doi = {{{doi}}},")
        if arxiv_id:
            lines.append(f"  eprint = {{{arxiv_id}}},")
            lines.append(f"  archivePrefix = {{arXiv}},")

        lines.append("}")
        return "\n".join(lines)


class GoogleScholarClient:
    """
    Async client for Google Scholar results through SerpAPI.

    - **Description**:
        - Searches Google Scholar via SerpAPI.
        - Normalizes results into the common paper schema.
        - Generates BibTeX entries for downstream tooling.
    """

    BASE_URL = "https://serpapi.com/search.json"

    def __init__(self, api_key: Optional[str] = None, timeout: int = 10):
        self.api_key = api_key
        self.timeout = timeout
        self._rate_limited = False

    @property
    def is_rate_limited(self) -> bool:
        """Whether the last request was rate-limited (429)."""
        return self._rate_limited

    async def search(
        self,
        query: str,
        max_results: int = 5,
        year_range: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for papers on Google Scholar via SerpAPI.

        - **Description**:
            - Calls SerpAPI `engine=google_scholar`.
            - Returns normalized paper entries.
            - Applies optional post-filtering by year range.

        - **Args**:
            - `query` (str): Search query string.
            - `max_results` (int): Maximum number of results.
            - `year_range` (str, optional): Year range filter, e.g. "2020-2025".

        - **Returns**:
            - `List[dict]`: List of paper dicts with standardized fields.
        """
        self._rate_limited = False

        if not self.api_key:
            print("[GoogleScholar] Missing SerpAPI key")
            return []

        params = {
            "engine": "google_scholar",
            "q": query,
            "num": min(max_results, 20),
            "api_key": self.api_key,
            "hl": "en",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException:
            print("[GoogleScholar] Request timed out")
            return []
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 429:
                self._rate_limited = True
            print(f"[GoogleScholar] HTTP error: {status}")
            return []
        except Exception as e:
            print(f"[GoogleScholar] Error: {e}")
            return []

        if data.get("error"):
            err = str(data["error"])
            if "rate" in err.lower() or "429" in err:
                self._rate_limited = True
            print(f"[GoogleScholar] API error: {err}")
            return []

        papers = []
        for item in data.get("organic_results", []):
            paper = self._normalize_paper(item)
            if not paper:
                continue
            if year_range and not self._in_year_range(paper.get("year"), year_range):
                continue
            papers.append(paper)

        return papers[:max_results]

    def _normalize_paper(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize a Google Scholar result to standard format."""
        title = item.get("title")
        if not title:
            return None

        publication_info = item.get("publication_info", {}) or {}
        summary_text = publication_info.get("summary", "") or ""
        snippet = item.get("snippet", "") or ""
        link = item.get("link", "") or ""
        result_id = item.get("result_id", "") or link

        authors = self._extract_authors(publication_info)
        year = self._extract_year(summary_text, snippet)
        citation_count = self._extract_citation_count(item)
        doi = self._extract_doi(link, snippet)
        arxiv_id = self._extract_arxiv_id(link, snippet)
        venue = item.get("publication", "") or self._extract_venue(summary_text)

        paper = {
            "source": "google_scholar",
            "paper_id": result_id,
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": snippet,
            "venue": venue,
            "citation_count": citation_count,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "bibtex_key": self._generate_bibtex_key(authors, year, title),
        }
        paper["bibtex"] = self._generate_bibtex(paper)
        return paper

    def _extract_authors(self, publication_info: Dict[str, Any]) -> List[str]:
        """Extract author names from publication info."""
        authors_data = publication_info.get("authors", []) or []
        if authors_data:
            return [a.get("name", "").strip() for a in authors_data if a.get("name")]

        summary = publication_info.get("summary", "") or ""
        if " - " in summary:
            head = summary.split(" - ", 1)[0]
            return [part.strip() for part in head.split(",") if part.strip()]
        return []

    def _extract_year(self, summary_text: str, snippet: str) -> Optional[int]:
        """Extract a publication year from text fields."""
        for text in [summary_text, snippet]:
            if not text:
                continue
            years = re.findall(r"\b(19\d{2}|20\d{2})\b", text)
            if years:
                return int(years[-1])
        return None

    def _extract_venue(self, summary_text: str) -> str:
        """Extract venue hint from summary text."""
        if not summary_text:
            return ""
        parts = [p.strip() for p in summary_text.split(" - ") if p.strip()]
        if len(parts) >= 2:
            return parts[-1]
        return ""

    def _extract_citation_count(self, item: Dict[str, Any]) -> Optional[int]:
        """Extract citation count from inline links."""
        cited_by = item.get("inline_links", {}).get("cited_by", {}) or {}
        if isinstance(cited_by, list):
            for inline in cited_by:
                if isinstance(inline, dict):
                    total = inline.get("total")
                    if isinstance(total, int):
                        return total
            return None
        total = cited_by.get("total")
        if isinstance(total, int):
            return total
        return None

    def _extract_doi(self, link: str, snippet: str) -> Optional[str]:
        """Extract DOI from link or snippet text."""
        for text in [link, snippet]:
            if not text:
                continue
            match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", text, re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def _extract_arxiv_id(self, link: str, snippet: str) -> Optional[str]:
        """Extract arXiv ID from link or snippet text."""
        for text in [link, snippet]:
            if not text:
                continue
            match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", text)
            if match:
                return match.group(1)
        return None

    def _in_year_range(self, year: Optional[int], year_range: str) -> bool:
        """Return whether year is inside range like START-END."""
        if year is None:
            return False
        try:
            parts = year_range.split("-")
            start_year = int(parts[0])
            end_year = int(parts[1]) if len(parts) > 1 else 9999
            return start_year <= year <= end_year
        except (ValueError, IndexError):
            return True

    def _generate_bibtex_key(
        self,
        authors: List[str],
        year: Optional[int],
        title: str,
    ) -> str:
        """Generate a BibTeX citation key from paper metadata."""
        if authors:
            first_author = authors[0]
            if "," in first_author:
                last_name = first_author.split(",")[0].strip()
            else:
                parts = first_author.split()
                last_name = parts[-1] if parts else "unknown"
        else:
            last_name = "unknown"

        last_name = re.sub(r'[^a-zA-Z]', '', last_name).lower()
        year_str = str(year) if year else "nd"
        title_words = [w for w in title.split() if len(w) > 3 and w.isalpha()]
        title_word = title_words[0].lower() if title_words else "paper"
        return f"{last_name}{year_str}{title_word}"

    def _generate_bibtex(self, paper: Dict[str, Any]) -> str:
        """Generate a BibTeX entry string from paper data."""
        key = paper["bibtex_key"]
        title = paper.get("title", "")
        authors = " and ".join(paper.get("authors", []))
        year = paper.get("year", "")
        venue = paper.get("venue", "")
        doi = paper.get("doi", "")
        arxiv_id = paper.get("arxiv_id", "")

        entry_type = "article"
        if venue and any(kw in venue.lower() for kw in ["conference", "proceedings", "workshop", "symposium"]):
            entry_type = "inproceedings"

        lines = [f"@{entry_type}{{{key},"]
        lines.append(f"  title = {{{title}}},")
        if authors:
            lines.append(f"  author = {{{authors}}},")
        if year:
            lines.append(f"  year = {{{year}}},")
        if entry_type == "inproceedings" and venue:
            lines.append(f"  booktitle = {{{venue}}},")
        elif venue:
            lines.append(f"  journal = {{{venue}}},")
        if doi:
            lines.append(f"  doi = {{{doi}}},")
        if arxiv_id:
            lines.append(f"  eprint = {{{arxiv_id}}},")
            lines.append("  archivePrefix = {arXiv},")
        lines.append("}")
        return "\n".join(lines)


class ArxivClient:
    """
    Async client for the arXiv API.

    - **Description**:
        - Searches for papers on arXiv using the Atom feed API.
        - Parses XML responses and generates BibTeX entries.
    """

    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    async def search(
        self,
        query: str,
        max_results: int = 5,
        year_range: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for papers on arXiv.

        - **Args**:
            - `query` (str): Search query string.
            - `max_results` (int): Maximum number of results.
            - `year_range` (str, optional): Year range (used for post-filtering).

        - **Returns**:
            - `List[dict]`: List of paper dicts with standardized fields.
        """
        # Build arXiv query (httpx handles URL encoding automatically)
        search_query = f"all:{query}"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": min(max_results * 2, 50),  # Fetch extra for year filtering
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            headers = {"User-Agent": "EasyPaper/1.0 (academic research tool)"}
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                response = await client.get(self.BASE_URL, params=params, headers=headers)
                response.raise_for_status()
                xml_content = response.text
        except httpx.TimeoutException:
            print("[arXiv] Request timed out")
            return []
        except httpx.HTTPStatusError as e:
            print(f"[arXiv] HTTP error: {e.response.status_code}")
            if e.response.status_code == 429:
                # Trigger circuit-breaker via PaperSearchTool class variable
                PaperSearchTool._arxiv_cooldown_until = _time.time() + 600
                print("[arXiv] Rate limited (429), cooldown set for 10 minutes")
            return []
        except Exception as e:
            print(f"[arXiv] Error: {e}")
            return []

        papers = self._parse_atom_feed(xml_content)

        # Apply year filtering if specified
        if year_range:
            papers = self._filter_by_year(papers, year_range)

        return papers[:max_results]

    def _parse_atom_feed(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse arXiv Atom feed XML into paper dicts."""
        papers = []
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            print(f"[arXiv] XML parse error: {e}")
            return []

        # Namespace handling for Atom feed
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        for entry in root.findall("atom:entry", ns):
            paper = self._parse_entry(entry, ns)
            if paper:
                papers.append(paper)

        return papers

    def _parse_entry(
        self,
        entry: ET.Element,
        ns: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        """Parse a single Atom entry into a paper dict."""
        title_el = entry.find("atom:title", ns)
        title = title_el.text.strip().replace("\n", " ") if title_el is not None and title_el.text else None
        if not title:
            return None

        # Authors
        authors = []
        for author_el in entry.findall("atom:author", ns):
            name_el = author_el.find("atom:name", ns)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        # Published date -> year
        published_el = entry.find("atom:published", ns)
        year = None
        if published_el is not None and published_el.text:
            year = int(published_el.text[:4])

        # Abstract
        summary_el = entry.find("atom:summary", ns)
        abstract = ""
        if summary_el is not None and summary_el.text:
            abstract = summary_el.text.strip().replace("\n", " ")

        # arXiv ID from the entry id URL
        id_el = entry.find("atom:id", ns)
        arxiv_id = ""
        if id_el is not None and id_el.text:
            # Extract ID from URL like http://arxiv.org/abs/2301.12345v1
            arxiv_url = id_el.text.strip()
            match = re.search(r'abs/(.+?)(?:v\d+)?$', arxiv_url)
            if match:
                arxiv_id = match.group(1)

        # DOI (if available)
        doi_el = entry.find("arxiv:doi", ns)
        doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None

        paper = {
            "source": "arxiv",
            "paper_id": arxiv_id,
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "venue": "arXiv preprint",
            "citation_count": None,  # arXiv doesn't provide this
            "doi": doi,
            "arxiv_id": arxiv_id,
            "bibtex_key": self._generate_bibtex_key(authors, year, title),
        }

        paper["bibtex"] = self._generate_bibtex(paper)
        return paper

    def _generate_bibtex_key(
        self,
        authors: List[str],
        year: Optional[int],
        title: str,
    ) -> str:
        """Generate a BibTeX citation key."""
        if authors:
            first_author = authors[0]
            parts = first_author.split()
            last_name = parts[-1] if parts else "unknown"
        else:
            last_name = "unknown"

        last_name = re.sub(r'[^a-zA-Z]', '', last_name).lower()
        year_str = str(year) if year else "nd"

        title_words = [w for w in title.split() if len(w) > 3 and w.isalpha()]
        title_word = title_words[0].lower() if title_words else "paper"

        return f"{last_name}{year_str}{title_word}"

    def _generate_bibtex(self, paper: Dict[str, Any]) -> str:
        """Generate a BibTeX entry string."""
        key = paper["bibtex_key"]
        title = paper.get("title", "")
        authors = " and ".join(paper.get("authors", []))
        year = paper.get("year", "")
        arxiv_id = paper.get("arxiv_id", "")
        doi = paper.get("doi", "")

        lines = [f"@article{{{key},"]
        lines.append(f"  title = {{{title}}},")
        if authors:
            lines.append(f"  author = {{{authors}}},")
        if year:
            lines.append(f"  year = {{{year}}},")
        lines.append(f"  journal = {{arXiv preprint arXiv:{arxiv_id}}},")
        if arxiv_id:
            lines.append(f"  eprint = {{{arxiv_id}}},")
            lines.append(f"  archivePrefix = {{arXiv}},")
        if doi:
            lines.append(f"  doi = {{{doi}}},")
        lines.append("}")
        return "\n".join(lines)

    def _filter_by_year(
        self,
        papers: List[Dict[str, Any]],
        year_range: str,
    ) -> List[Dict[str, Any]]:
        """Filter papers by year range string like '2020-2025'."""
        try:
            parts = year_range.split("-")
            start_year = int(parts[0])
            end_year = int(parts[1]) if len(parts) > 1 else 9999
        except (ValueError, IndexError):
            return papers

        return [
            p for p in papers
            if p.get("year") and start_year <= p["year"] <= end_year
        ]


class PaperSearchTool(WriterTool):
    """
    Tool for searching academic papers and generating BibTeX entries.

    - **Description**:
        - Searches Google Scholar, Semantic Scholar, and arXiv.
        - Returns structured paper metadata with ready-to-use BibTeX.
        - Supports year range filtering and result count limits.
        - Default mode is "google_scholar" for broader retrieval.
        - "auto" mode runs all sources in parallel with aggressive arXiv
          timeout to avoid blocking.
        - arXiv has a circuit-breaker: after a 429 error, arXiv calls are
          skipped for a cooldown period to avoid wasting time.
    """

    # Circuit-breaker: skip arXiv for this many seconds after a 429
    _arxiv_cooldown_until: float = 0.0

    def __init__(
        self,
        serpapi_api_key: Optional[str] = None,
        semantic_scholar_api_key: Optional[str] = None,
        timeout: int = 10,
    ):
        self._gs_client = GoogleScholarClient(
            api_key=serpapi_api_key,
            timeout=timeout,
        )
        self._ss_client = SemanticScholarClient(
            api_key=semantic_scholar_api_key,
            timeout=timeout,
        )
        # arXiv gets a shorter timeout to avoid blocking
        self._arxiv_client = ArxivClient(timeout=min(timeout, 5))

    @property
    def name(self) -> str:
        return "search_papers"

    @property
    def description(self) -> str:
        return (
            "Search academic papers by query keywords. Returns paper metadata "
            "(title, authors, year, abstract, venue, citation count) and "
            "ready-to-use BibTeX entries. Uses Google Scholar by default. "
            "Use this tool when you need to find relevant references for a "
            "topic or verify the existence of cited works."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords or a natural language query describing the papers to find."
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        "Maximum number of papers to return. "
                        "The caller should choose this based on current writing needs."
                    ),
                    "minimum": 1
                },
                "year_range": {
                    "type": "string",
                    "description": "Optional year range filter, e.g. '2020-2025'. Format: 'START-END'."
                },
                "source": {
                    "type": "string",
                    "description": (
                        "Search source. 'google_scholar' (default, broad coverage), "
                        "'semantic_scholar' (structured metadata), 'arxiv' (preprints only), "
                        "or 'auto' (all in parallel)."
                    ),
                    "enum": ["google_scholar", "semantic_scholar", "arxiv", "auto"],
                    "default": "google_scholar"
                }
            },
            "required": ["query", "max_results"]
        }

    def _merge_unique_by_title(
        self,
        target: List[Dict[str, Any]],
        seen_titles: set,
        papers: List[Dict[str, Any]],
    ) -> int:
        """
        Merge papers into target list with title-based deduplication.

        - **Args**:
            - `target` (List[dict]): Destination papers list.
            - `seen_titles` (set): Lowercased title set for deduplication.
            - `papers` (List[dict]): Candidate papers to merge.

        - **Returns**:
            - `int`: Number of newly added papers.
        """
        added = 0
        for paper in papers:
            title = str(paper.get("title", "")).strip().lower()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            target.append(paper)
            added += 1
        return added

    def _is_arxiv_available(self) -> bool:
        """
        Check if arXiv is available (not in cooldown after 429).

        - **Returns**:
            - `bool`: True if arXiv can be called, False if in cooldown.
        """
        if _time.time() < PaperSearchTool._arxiv_cooldown_until:
            remaining = int(PaperSearchTool._arxiv_cooldown_until - _time.time())
            print(f"[Tool:search_papers] arXiv in cooldown ({remaining}s remaining), skipping")
            return False
        return True

    def _set_arxiv_cooldown(self, seconds: int = 600) -> None:
        """
        Set arXiv cooldown after a 429 error.

        - **Args**:
            - `seconds` (int): Cooldown duration. Default 600s (10 minutes).
        """
        PaperSearchTool._arxiv_cooldown_until = _time.time() + seconds
        print(f"[Tool:search_papers] arXiv cooldown set for {seconds}s")

    async def _search_arxiv_safe(
        self,
        query: str,
        max_results: int,
        year_range: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Search arXiv with circuit-breaker and fast timeout.
        - **Description**:
            - Skips the call entirely if arXiv is in cooldown.
            - Sets cooldown on 429 errors to avoid repeated failures.

        - **Returns**:
            - `List[dict]`: Papers found, or empty list on failure.
        """
        if not self._is_arxiv_available():
            return []

        papers = await self._arxiv_client.search(
            query=query,
            max_results=max_results,
            year_range=year_range,
        )

        # The ArxivClient returns [] on errors; we check if it was a 429
        # by hooking into the client. Since we can't easily detect 429 from
        # the return value, we do a direct check here.
        return papers

    async def execute(
        self,
        query: str,
        max_results: Optional[int] = None,
        year_range: Optional[str] = None,
        source: str = "google_scholar",
        **kwargs,
    ) -> ToolResult:
        """
        Search for academic papers with multi-source fallback.

        - **Description**:
            - If source is "google_scholar" and no SerpAPI key is configured,
              automatically falls back to Semantic Scholar (then arXiv on 429).
            - If source is "semantic_scholar" and the request is rate-limited
              (429 after retries), automatically falls back to arXiv.
            - If source is "auto", runs all sources in parallel.

        - **Args**:
            - `query` (str): Search query.
            - `max_results` (int, optional): Max papers to return.
            - `year_range` (str, optional): Year range filter.
            - `source` (str): Search source. Default "google_scholar".

        - **Returns**:
            - `ToolResult` with data containing:
              - `papers`: List of paper metadata dicts.
              - `bibtex`: Combined BibTeX string for all found papers.
              - `total_found`: Number of papers found.
        """
        if max_results is None:
            return ToolResult(
                success=False,
                message="Parameter 'max_results' is required and must be >= 1.",
                data={"papers": [], "bibtex": "", "total_found": 0},
            )
        if max_results < 1:
            return ToolResult(
                success=False,
                message="Parameter 'max_results' must be >= 1.",
                data={"papers": [], "bibtex": "", "total_found": 0},
            )

        max_res = max_results
        print(f"[Tool:search_papers] Searching '{query}' (max={max_res}, "
              f"source={source}, years={year_range or 'any'})...")

        all_papers: List[Dict[str, Any]] = []
        seen_titles: set = set()
        message_notes: List[str] = []

        if source == "auto":
            # Run all sources in parallel to minimize total wait time
            gs_task = asyncio.create_task(
                self._gs_client.search(query=query, max_results=max_res, year_range=year_range)
            )
            ss_task = asyncio.create_task(
                self._ss_client.search(query=query, max_results=max_res, year_range=year_range)
            )
            arxiv_task = asyncio.create_task(
                self._search_arxiv_safe(query=query, max_results=max_res, year_range=year_range)
            )

            gs_papers, ss_papers, arxiv_papers = await asyncio.gather(gs_task, ss_task, arxiv_task)

            gs_added = self._merge_unique_by_title(all_papers, seen_titles, gs_papers)
            ss_added = self._merge_unique_by_title(all_papers, seen_titles, ss_papers)
            arxiv_added = self._merge_unique_by_title(all_papers, seen_titles, arxiv_papers)

            print(f"[Tool:search_papers] Google Scholar: {len(gs_papers)} results ({gs_added} new)")
            if ss_papers or ss_added:
                print(f"[Tool:search_papers] Semantic Scholar: {len(ss_papers)} results ({ss_added} new)")
            if arxiv_papers or arxiv_added:
                print(f"[Tool:search_papers] arXiv: {len(arxiv_papers)} results ({arxiv_added} new)")

            if not self._gs_client.api_key:
                message_notes.append("SerpAPI key missing; Google Scholar may return 0 results.")

        elif source == "google_scholar":
            gs_papers = await self._gs_client.search(
                query=query, max_results=max_res, year_range=year_range,
            )
            gs_added = self._merge_unique_by_title(all_papers, seen_titles, gs_papers)
            print(f"[Tool:search_papers] Google Scholar: {len(gs_papers)} results ({gs_added} new)")

            # Fallback for missing key or request failure/empty result.
            if not gs_papers:
                if not self._gs_client.api_key:
                    message_notes.append("SerpAPI key missing; fell back to Semantic Scholar.")
                elif self._gs_client.is_rate_limited:
                    message_notes.append("Google Scholar rate-limited; fell back to Semantic Scholar.")
                else:
                    message_notes.append("Google Scholar returned no results; fell back to Semantic Scholar.")

                ss_papers = await self._ss_client.search(
                    query=query, max_results=max_res, year_range=year_range,
                )
                ss_added = self._merge_unique_by_title(all_papers, seen_titles, ss_papers)
                print(f"[Tool:search_papers] Semantic Scholar fallback: {len(ss_papers)} results ({ss_added} new)")

                if not ss_papers and self._ss_client.is_rate_limited:
                    arxiv_papers = await self._search_arxiv_safe(
                        query=query, max_results=max_res, year_range=year_range,
                    )
                    arxiv_added = self._merge_unique_by_title(all_papers, seen_titles, arxiv_papers)
                    print(f"[Tool:search_papers] arXiv fallback: {len(arxiv_papers)} results ({arxiv_added} new)")

        elif source == "semantic_scholar":
            ss_papers = await self._ss_client.search(
                query=query, max_results=max_res, year_range=year_range,
            )
            ss_added = self._merge_unique_by_title(all_papers, seen_titles, ss_papers)
            print(f"[Tool:search_papers] Semantic Scholar: {len(ss_papers)} results ({ss_added} new)")

            # Fallback: if Semantic Scholar was rate-limited, try arXiv
            if not ss_papers and self._ss_client.is_rate_limited:
                print("[Tool:search_papers] Semantic Scholar rate-limited, falling back to arXiv...")
                arxiv_papers = await self._search_arxiv_safe(
                    query=query, max_results=max_res, year_range=year_range,
                )
                arxiv_added = self._merge_unique_by_title(all_papers, seen_titles, arxiv_papers)
                print(f"[Tool:search_papers] arXiv fallback: {len(arxiv_papers)} results ({arxiv_added} new)")

        elif source == "arxiv":
            arxiv_papers = await self._search_arxiv_safe(
                query=query, max_results=max_res, year_range=year_range,
            )
            arxiv_added = self._merge_unique_by_title(all_papers, seen_titles, arxiv_papers)
            if arxiv_papers or arxiv_added:
                print(f"[Tool:search_papers] arXiv: {len(arxiv_papers)} results ({arxiv_added} new)")
        else:
            return ToolResult(
                success=False,
                message=(
                    "Parameter 'source' must be one of "
                    "['google_scholar', 'semantic_scholar', 'arxiv', 'auto']."
                ),
                data={"papers": [], "bibtex": "", "total_found": 0},
            )

        # Trim to max_results
        all_papers = all_papers[:max_res]

        # Deduplicate BibTeX keys
        used_keys: set = set()
        for paper in all_papers:
            key = paper["bibtex_key"]
            if key in used_keys:
                suffix = 2
                while f"{key}{chr(96 + suffix)}" in used_keys:
                    suffix += 1
                new_key = f"{key}{chr(96 + suffix)}"
                paper["bibtex_key"] = new_key
                paper["bibtex"] = paper["bibtex"].replace(
                    f"{{{key},", f"{{{new_key},"
                )
            used_keys.add(paper["bibtex_key"])

        # Build combined BibTeX
        bibtex_entries = [p["bibtex"] for p in all_papers if p.get("bibtex")]
        combined_bibtex = "\n\n".join(bibtex_entries)

        # Build summary for each paper (for LLM consumption and ref_pool merging)
        paper_summaries = []
        for p in all_papers:
            summary = {
                "bibtex_key": p["bibtex_key"],
                "title": p["title"],
                "authors": p["authors"][:5],  # Limit to 5 authors for brevity
                "year": p["year"],
                "venue": p["venue"],
                "citation_count": p.get("citation_count"),
                "abstract": (p.get("abstract", "")[:300] + "...")
                            if p.get("abstract") and len(p.get("abstract", "")) > 300
                            else p.get("abstract", ""),
                "bibtex": p.get("bibtex", ""),  # Include for ref_pool merging
            }
            paper_summaries.append(summary)

        data = {
            "papers": paper_summaries,
            "bibtex": combined_bibtex,
            "total_found": len(all_papers),
        }

        if all_papers:
            message = (f"Found {len(all_papers)} paper(s) for '{query}'. "
                       f"BibTeX keys: {[p['bibtex_key'] for p in all_papers]}")
            if message_notes:
                message += f" Notes: {' '.join(message_notes)}"
            print(f"[Tool:search_papers] OK: {len(all_papers)} papers found")
        else:
            message = f"No papers found for '{query}'."
            if message_notes:
                message += f" Notes: {' '.join(message_notes)}"
            print(f"[Tool:search_papers] No results")

        return ToolResult(
            success=True,
            data=data,
            message=message,
        )
