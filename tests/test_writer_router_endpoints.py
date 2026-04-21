"""
Endpoint-level compatibility tests for the public writer router.
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agents.writer_agent.router import create_writer_router


class _FakeWriterAgent:
    async def run(
        self,
        system_prompt: str,
        user_prompt: str,
        section_type: str = "introduction",
        citation_format: str = "cite",
        constraints=None,
        **kwargs,
    ):
        return {
            "generated_content": f"Generated {section_type} content.",
            "citation_ids": ["smith2024"],
            "figure_ids": ["fig:demo"],
            "table_ids": ["tab:demo"],
            "invalid_citations_removed": [],
            "review_history": [
                {
                    "passed": True,
                    "issues": [],
                    "warnings": [],
                    "invalid_citations": [],
                    "word_count": 3,
                    "target_words": None,
                    "key_point_coverage": 1.0,
                }
            ],
            "writer_response_section": [],
            "writer_response_paragraph": [],
            "paragraph_units": [],
            "iteration": 1,
            "review_result": {"passed": True},
        }


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(create_writer_router(_FakeWriterAgent()))
    return TestClient(app)


def test_writer_generate_endpoint_compat():
    client = _make_client()
    response = client.post(
        "/agent/writer/generate",
        json={
            "request_id": "req-1",
            "payload": {
                "system_prompt": "sys",
                "user_prompt": "Write intro",
                "section_type": "introduction",
                "citation_format": "cite",
                "constraints": [],
            },
            "valid_citation_keys": ["smith2024"],
            "key_points": ["motivation"],
            "max_iterations": 1,
            "enable_review": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["result"]["section_type"] == "introduction"
    assert data["result"]["latex_content"] == "Generated introduction content."
    assert data["result"]["citation_ids"] == ["smith2024"]


def test_writer_write_section_endpoint_compat():
    client = _make_client()
    response = client.post(
        "/agent/writer/write-section?validate=false",
        json={
            "request_id": "req-2",
            "section_type": "method",
            "section_title": "Method",
            "user_prompt": "Describe the method.",
            "constraints": {
                "citation_format": "cite",
                "additional_instructions": [],
            },
            "resources": {
                "references": [],
                "figures": [],
                "tables": [],
                "equations": [],
            },
            "argument": {
                "thesis": "Method summary",
                "main_points": [],
                "background_context": [],
            },
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["section_type"] == "method"
    assert data["latex_content"] == "Generated method content."


def test_writer_assemble_paper_endpoint_compat():
    client = _make_client()
    response = client.post(
        "/agent/writer/assemble-paper",
        json={
            "paper_title": "Demo Paper",
            "output_dir": "/tmp/easypaper-writer-router-test",
            "compile_pdf": False,
            "sections": [
                {
                    "section_type": "introduction",
                    "enabled": True,
                    "payload": {
                        "request_id": "req-3",
                        "section_type": "introduction",
                        "section_title": "Introduction",
                        "user_prompt": "Write intro",
                        "constraints": {
                            "citation_format": "cite",
                            "additional_instructions": [],
                        },
                        "resources": {
                            "references": [],
                            "figures": [],
                            "tables": [],
                            "equations": [],
                        },
                        "argument": {
                            "thesis": "Intro thesis",
                            "main_points": [],
                            "background_context": [],
                        },
                    },
                }
            ],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["paper_title"] == "Demo Paper"
    assert len(data["sections_status"]) == 1
    assert data["sections_status"][0]["status"] == "ok"
