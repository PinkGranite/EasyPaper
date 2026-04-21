"""
LLM-based claim-evidence matching for DAG construction.

This complements the heuristic DAGBuilder by using LLM reasoning
to determine semantic relationships between claims and evidence.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClaimEvidenceMatch:
    """A single claim-evidence match with confidence and reasoning."""
    claim_id: str
    evidence_id: str
    confidence: float  # 0-1
    reasoning: str


class LLMDAGBuilder:
    """
    Uses LLM to analyze semantic relationships between claims and evidence,
    producing matches that supplement the heuristic DAG builder.
    """

    def __init__(self, llm_client: Any, model_name: Optional[str] = None):
        """
        Initialize with an LLM client and model name.

        - **Args**:
            - `llm_client`: Chat completions client (e.g., OpenAI-compatible).
              Must have `chat.completions.create` method.
            - `model_name` (str, optional): Model name to use for LLM calls.
              If not provided, must be passed to ``find_claim_evidence_matches``.
        """
        self._llm = llm_client
        self._model_name = model_name

    async def find_claim_evidence_matches(
        self,
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
        paper_topic: str,
        model_name: Optional[str] = None,
        temperature: float = 0.3,
    ) -> List[ClaimEvidenceMatch]:
        """
        For each claim, find the most relevant evidence using LLM semantic understanding.

        - **Args**:
            - `claims`: List of claim nodes with statement, section_scope, priority
            - `evidence`: List of evidence nodes with content, source_path, node_type
            - `paper_topic`: The research topic for contextual relevance
            - `model_name`: Model to use for LLM calls. If not provided, uses the
              model from the LLM client passed at construction time.
            - `temperature`: Temperature for LLM sampling

        - **Returns**:
            - `List[ClaimEvidenceMatch]`: List of matches with confidence scores
        """
        if not claims or not evidence:
            return []

        prompt = self._build_matching_prompt(claims, evidence, paper_topic)
        response = await self._call_llm(prompt, model_name, temperature)
        return self._parse_matches(response)

    def _build_matching_prompt(
        self,
        claims: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
        topic: str,
    ) -> str:
        """
        Build the LLM prompt for claim-evidence matching.
        """
        claims_text = "\n".join([
            f"- Claim {c.get('node_id', c.get('claim_id', '?'))}: "
            f"{c.get('statement', c.get('content', ''))} "
            f"(scope: {c.get('section_scope', [])}, priority: {c.get('priority', 'P1')})"
            for c in claims
        ])

        evidence_text = "\n".join([
            f"- Evidence {e.get('node_id', e.get('evidence_id', '?'))}: "
            f"{e.get('content', '')[:200]}... "
            f"(type: {e.get('node_type', 'unknown')}, source: {e.get('source_path', '')})"
            for e in evidence
        ])

        return f"""Given this research topic: {topic}

Analyze which evidence supports which claims. Consider:
1. Semantic relevance - does the evidence actually prove/support the claim?
2. Section scope - is the evidence appropriate for where the claim will be used?
3. Priority - P0 claims are most important and need the best evidence.
4. Evidence type - code evidence supports method claims, literature supports background/related work.

Claims:
{claims_text}

Evidence:
{evidence_text}

Return JSON with a "matches" array. Each match should include:
- "claim_id": the claim identifier
- "evidence_id": the evidence identifier
- "confidence": score from 0.0 to 1.0 (how strongly this evidence supports this claim)
- "reasoning": brief explanation of why this evidence supports this claim

Only include matches with confidence >= 0.5. Format:
{{"matches": [{{"claim_id": "...", "evidence_id": "...", "confidence": 0.0-1.0, "reasoning": "..."}}]}}
"""

    async def _call_llm(
        self,
        prompt: str,
        model_name: Optional[str],
        temperature: float,
    ) -> str:
        """
        Make an LLM call and return the response text.

        - **Args**:
            - `model_name`: Model name. If None, uses the model from the LLM client
              passed at construction time.
        """
        system_prompt = """You are a research assistant that matches evidence to claims in academic papers.
Be precise and only match evidence that genuinely supports the claim.
Output valid JSON only."""

        model = model_name or self._model_name
        if not model:
            raise ValueError(
                "model_name must be passed to __init__ or to find_claim_evidence_matches"
            )

        try:
            response = await self._llm.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            text = response.choices[0].message.content.strip()
            logger.debug("LLM match response: %s", text[:500])
            return text
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return '{"matches": []}'

    def _parse_matches(self, response: str) -> List[ClaimEvidenceMatch]:
        """
        Parse LLM response into ClaimEvidenceMatch objects.
        """
        matches = []

        # Try to extract JSON from the response
        json_str = self._extract_json(response)
        if not json_str:
            logger.warning("No JSON found in LLM response")
            return matches

        try:
            data = json.loads(json_str)
            raw_matches = data.get("matches", [])

            for m in raw_matches:
                try:
                    matches.append(ClaimEvidenceMatch(
                        claim_id=m["claim_id"],
                        evidence_id=m["evidence_id"],
                        confidence=float(m["confidence"]),
                        reasoning=m.get("reasoning", ""),
                    ))
                except (KeyError, ValueError) as e:
                    logger.warning("Skipping invalid match: %s, error: %s", m, e)
                    continue

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON from LLM response: %s", e)

        return matches

    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract JSON from LLM response text.
        Handles cases where the JSON is wrapped in markdown code blocks.
        """
        # Try direct parse first
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code blocks
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{.*\})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                candidate = match.group(1)
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue

        return None
