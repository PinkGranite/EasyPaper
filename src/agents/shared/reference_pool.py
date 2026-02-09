"""
ReferencePool - Persistent reference management across paper generation phases.

- **Description**:
    - Manages a growing pool of academic references used during paper generation.
    - Separates core references (user-provided, immutable) from discovered
      references (found via search_papers during writing).
    - Provides real-time valid_citation_keys as the pool grows.
    - Generates the final .bib file content from all accumulated references.
"""

import json
import re
from typing import Any, Dict, List, Optional, Set


class ReferencePool:
    """
    Persistent reference pool that accumulates citations across generation phases.

    - **Description**:
        - Initialized with the user's core references (BibTeX strings).
        - During content generation, search_papers may discover new papers.
        - After validation (two-layer: LLM judgment + system cross-reference),
          new papers are added via add_discovered().
        - valid_citation_keys grows in real-time, so subsequent sections and
          mini_review always see the full, up-to-date reference set.
        - to_bibtex() produces the complete .bib content for final output.

    - **Args**:
        - `initial_bibtex_list` (List[str]): User-provided BibTeX entry strings.
    """

    def __init__(self, initial_bibtex_list: List[str]):
        self._core_refs: List[Dict[str, Any]] = self._parse_bibtex_list(
            initial_bibtex_list
        )
        self._discovered_refs: List[Dict[str, Any]] = []
        self._all_keys: Set[str] = {r["ref_id"] for r in self._core_refs if r.get("ref_id")}

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def valid_citation_keys(self) -> Set[str]:
        """
        All currently valid citation keys (core + discovered).

        - **Returns**:
            - `Set[str]`: The complete set of valid BibTeX citation keys.
        """
        return set(self._all_keys)

    @property
    def core_refs(self) -> List[Dict[str, Any]]:
        """
        User-provided core references (read-only copy).

        - **Returns**:
            - `List[Dict]`: The core reference list.
        """
        return list(self._core_refs)

    @property
    def discovered_refs(self) -> List[Dict[str, Any]]:
        """
        References discovered during writing via search_papers (read-only copy).

        - **Returns**:
            - `List[Dict]`: The discovered reference list.
        """
        return list(self._discovered_refs)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_all_refs(self) -> List[Dict[str, Any]]:
        """
        Get all references (core + discovered).

        - **Returns**:
            - `List[Dict]`: Combined reference list for prompt building and
              bib generation.
        """
        return self._core_refs + self._discovered_refs

    def has_key(self, ref_id: str) -> bool:
        """
        Check if a citation key exists in the pool.

        - **Args**:
            - `ref_id` (str): The citation key to check.

        - **Returns**:
            - `bool`: True if the key is in the pool.
        """
        return ref_id in self._all_keys

    def get_ref(self, ref_id: str) -> Optional[Dict[str, Any]]:
        """
        Look up a reference by its citation key.

        - **Args**:
            - `ref_id` (str): The citation key.

        - **Returns**:
            - `Optional[Dict]`: The reference dict, or None if not found.
        """
        for ref in self._core_refs + self._discovered_refs:
            if ref.get("ref_id") == ref_id:
                return ref
        return None

    def add_discovered(
        self,
        ref_id: str,
        bibtex: str,
        source: str = "search",
    ) -> bool:
        """
        Add a validated discovered paper to the pool.

        - **Description**:
            - Skips duplicates (returns False if ref_id already exists).
            - Parses the BibTeX to extract metadata.
            - Tracks provenance via `source` field.

        - **Args**:
            - `ref_id` (str): The BibTeX citation key.
            - `bibtex` (str): The full BibTeX entry string.
            - `source` (str): Provenance label (default "search").

        - **Returns**:
            - `bool`: True if added, False if duplicate.
        """
        if ref_id in self._all_keys:
            return False
        parsed = self._parse_single_bibtex(bibtex, fallback_id=ref_id)
        parsed["source"] = source
        self._discovered_refs.append(parsed)
        self._all_keys.add(ref_id)
        return True

    def to_bibtex(self) -> str:
        """
        Generate complete .bib file content from all references.

        - **Returns**:
            - `str`: Combined BibTeX string for the entire pool.
        """
        bib_entries = []
        for ref in self.get_all_refs():
            if ref.get("bibtex"):
                bib_entries.append(ref["bibtex"])
            else:
                # Fallback: generate a minimal entry
                ref_id = ref.get("ref_id", "unknown")
                title = ref.get("title", "Unknown Title")
                authors = ref.get("authors", "Unknown Author")
                year = ref.get("year", 2024)
                entry = (
                    f"@article{{{ref_id},\n"
                    f"  title = {{{title}}},\n"
                    f"  author = {{{authors}}},\n"
                    f"  year = {{{year}}},\n"
                    f"}}"
                )
                bib_entries.append(entry)
        return "\n\n".join(bib_entries)

    def summary(self) -> str:
        """
        Human-readable summary for logging.

        - **Returns**:
            - `str`: Summary string like "core=5, discovered=3, total_keys=8".
        """
        return (
            f"core={len(self._core_refs)}, "
            f"discovered={len(self._discovered_refs)}, "
            f"total_keys={len(self._all_keys)}"
        )

    # ------------------------------------------------------------------
    # Static / class-level helpers for post-ReAct validation
    # ------------------------------------------------------------------

    @staticmethod
    def extract_cite_keys(content: str) -> Set[str]:
        """
        Extract all citation keys from LaTeX content.

        - **Description**:
            - Finds all \\cite{key1, key2, ...} patterns and returns the
              individual keys.

        - **Args**:
            - `content` (str): LaTeX content string.

        - **Returns**:
            - `Set[str]`: Set of citation keys found in the content.
        """
        keys: Set[str] = set()
        for match in re.finditer(r"\\cite\{([^}]+)\}", content):
            for key in match.group(1).split(","):
                stripped = key.strip()
                if stripped:
                    keys.add(stripped)
        return keys

    @staticmethod
    def extract_search_results_from_history(
        msg_history: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Extract BibTeX entries from search_papers tool results in message history.

        - **Description**:
            - Scans messages with role='tool' for search_papers results.
            - Parses the JSON content to find bibtex_key -> bibtex mappings.

        - **Args**:
            - `msg_history` (List[Dict]): Message history from react_loop.

        - **Returns**:
            - `Dict[str, str]`: Mapping of bibtex_key to full BibTeX string.
        """
        results: Dict[str, str] = {}
        for msg in msg_history:
            if msg.get("role") != "tool":
                continue
            content_str = msg.get("content", "")
            if not content_str:
                continue
            try:
                data = json.loads(content_str)
            except (json.JSONDecodeError, TypeError):
                continue

            # ToolResult format: {"success": ..., "data": {"papers": [...], "bibtex": "..."}, ...}
            tool_data = data.get("data", {})
            if not isinstance(tool_data, dict):
                continue
            papers = tool_data.get("papers", [])
            combined_bibtex = tool_data.get("bibtex", "")

            if not papers:
                continue

            # Parse individual BibTeX entries from the combined string
            bibtex_map = ReferencePool._split_bibtex_entries(combined_bibtex)

            # Map bibtex_key from paper summaries to their BibTeX strings
            for paper in papers:
                bkey = paper.get("bibtex_key", "")
                if bkey and bkey in bibtex_map:
                    results[bkey] = bibtex_map[bkey]
                elif bkey and bkey not in results:
                    # Try to find by key in the combined string
                    entry = ReferencePool._find_bibtex_entry(combined_bibtex, bkey)
                    if entry:
                        results[bkey] = entry

        return results

    @staticmethod
    def remove_citation(content: str, key: str) -> str:
        """
        Remove a specific citation key from LaTeX content.

        - **Description**:
            - Handles single-key \\cite{key} → removes entire command.
            - Handles multi-key \\cite{a, key, b} → removes just that key.

        - **Args**:
            - `content` (str): LaTeX content.
            - `key` (str): Citation key to remove.

        - **Returns**:
            - `str`: Content with the citation key removed.
        """
        escaped_key = re.escape(key)

        # Pattern 1: sole key in \cite{key} → remove the whole \cite{}
        content = re.sub(
            rf"\\cite\{{\s*{escaped_key}\s*\}}",
            "",
            content,
        )

        # Pattern 2: key among others → remove just that key
        # \cite{a, key, b} → \cite{a, b}
        content = re.sub(
            rf",\s*{escaped_key}",
            "",
            content,
        )
        content = re.sub(
            rf"{escaped_key}\s*,\s*",
            "",
            content,
        )

        # Clean up empty cites and trailing whitespace
        content = re.sub(r"\\cite\{\s*\}", "", content)
        content = re.sub(r"  +", " ", content)
        content = re.sub(r" +([.,;:])", r"\1", content)

        return content

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_bibtex_list(self, bibtex_list: List[str]) -> List[Dict[str, Any]]:
        """
        Parse a list of BibTeX entry strings into structured dicts.

        - **Args**:
            - `bibtex_list` (List[str]): Raw BibTeX strings.

        - **Returns**:
            - `List[Dict]`: Parsed reference dicts with ref_id, title,
              authors, year, bibtex fields.
        """
        parsed = []
        for bibtex in bibtex_list:
            parsed.append(
                self._parse_single_bibtex(bibtex, fallback_id=f"ref_{len(parsed) + 1}")
            )
        return parsed

    @staticmethod
    def _parse_single_bibtex(bibtex: str, fallback_id: str = "unknown") -> Dict[str, Any]:
        """
        Parse a single BibTeX entry string into a structured dict.

        - **Args**:
            - `bibtex` (str): Raw BibTeX string.
            - `fallback_id` (str): Fallback ref_id if parsing fails.

        - **Returns**:
            - `Dict[str, Any]`: Parsed reference dict.
        """
        try:
            ref_id_match = re.search(r"@\w+{([^,]+),", bibtex)
            title_match = re.search(
                r"title\s*=\s*[{\"]([^}\"]+)[}\"]", bibtex, re.IGNORECASE
            )
            author_match = re.search(
                r"author\s*=\s*[{\"]([^}\"]+)[}\"]", bibtex, re.IGNORECASE
            )
            year_match = re.search(
                r"year\s*=\s*[{\"]?(\d{4})[}\"]?", bibtex, re.IGNORECASE
            )

            return {
                "ref_id": ref_id_match.group(1).strip() if ref_id_match else fallback_id,
                "title": title_match.group(1) if title_match else "",
                "authors": author_match.group(1) if author_match else "",
                "year": int(year_match.group(1)) if year_match else None,
                "bibtex": bibtex,
            }
        except Exception:
            return {
                "ref_id": fallback_id,
                "bibtex": bibtex,
            }

    @staticmethod
    def _split_bibtex_entries(combined: str) -> Dict[str, str]:
        """
        Split a combined BibTeX string into individual entries keyed by ref_id.

        - **Args**:
            - `combined` (str): Multi-entry BibTeX string.

        - **Returns**:
            - `Dict[str, str]`: Mapping from citation key to entry string.
        """
        entries: Dict[str, str] = {}
        # Split on @type{ patterns
        parts = re.split(r"(?=@\w+\{)", combined)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            key_match = re.search(r"@\w+{([^,]+),", part)
            if key_match:
                entries[key_match.group(1).strip()] = part
        return entries

    @staticmethod
    def _find_bibtex_entry(combined: str, key: str) -> Optional[str]:
        """
        Find a specific BibTeX entry by key in a combined BibTeX string.

        - **Args**:
            - `combined` (str): Multi-entry BibTeX string.
            - `key` (str): Citation key to find.

        - **Returns**:
            - `Optional[str]`: The BibTeX entry or None if not found.
        """
        entries = ReferencePool._split_bibtex_entries(combined)
        return entries.get(key)
