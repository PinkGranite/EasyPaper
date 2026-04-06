"""
Simple test for ParseAgent functionality.
Tests with a real PDF file and checks for JSON output.
"""
import pytest
import asyncio
import json
import sys
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.schema import ModelConfig
from agents.parse_agent.parse_agent import ParseAgent


class TestParseAgentSimple:
    """Simple test suite for ParseAgent"""

    @pytest.fixture
    def config(self):
        """Create a ModelConfig for testing"""
        return ModelConfig(
            model_name="openai/gpt-4o-mini",
            api_key="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",
            base_url="https://openrouter.ai/api/v1"
        )

    def test_agent_initialization(self, config):
        """Test that ParseAgent initializes correctly"""
        agent = ParseAgent(config)
        assert agent.name == "paper_parser"
        assert agent.description == "Research paper understanding and parsing agent"
        assert agent.model_name == config.model_name

    @pytest.mark.asyncio
    async def test_parse_pdf_output_format(self, config):
        """Test that ParseAgent can parse a PDF and output JSON format"""
        # You need to provide a real PDF file for this test
        pdf_path = "/Users/pinkgranite/code_repo/local/academicHub/agentsys/test_pdf.pdf"

        # Check if test file exists
        if not Path(pdf_path).exists():
            pytest.skip(f"Test PDF file not found at: {pdf_path}")

        agent = ParseAgent(config)

        try:
            # Run the agent
            result = await agent.run(file_path=pdf_path)

            # Check that we got a result
            assert result is not None
            assert "understand_result" in result

            # Check that the understand_result is a dictionary
            understand_result = result["understand_result"]
            assert isinstance(understand_result, dict)

            # Check for expected JSON fields
            expected_fields = [
                "summary",
                "research_background",
                "research_question",
                "research_hypothesis",
                "methods",
                "results",
                "key_findings"
            ]

            for field in expected_fields:
                assert field in understand_result, f"Missing field: {field}"

            # Print the result for manual inspection
            print("\n=== ParseAgent Test Results ===")
            print(f"Successfully parsed PDF: {pdf_path}")
            print("\nParsed JSON output:")
            print(json.dumps(understand_result, indent=2))

        except Exception as e:
            pytest.fail(f"ParseAgent failed to process PDF: {e}")

    @pytest.mark.asyncio
    async def test_agent_properties(self, config):
        """Test agent properties and metadata"""
        agent = ParseAgent(config)

        # Test basic properties
        assert agent.name == "paper_parser"
        assert agent.description == "Research paper understanding and parsing agent"

        # Test router property
        from fastapi import APIRouter
        assert hasattr(agent, 'router')

        # Test endpoints_info
        endpoints = agent.endpoints_info
        assert isinstance(endpoints, list)
        assert len(endpoints) > 0


if __name__ == "__main__":
    # Run a simple test directly
    print("Running simple ParseAgent test...")

    config = ModelConfig(
        model_name="openai/gpt-4o-mini",
        api_key="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",
        base_url="https://openrouter.ai/api/v1"
    )

    pdf_path = "./test_pdf.pdf"

    if not Path(pdf_path).exists():
        print(f"Please place a PDF file at: {pdf_path}")
        print("Then run this test again.")
    else:
        async def run_test():
            agent = ParseAgent(config)
            try:
                result = await agent.run(file_path=pdf_path)
                print("✅ Success! ParseAgent processed the PDF.")
                if 'understand_result' in result:
                    print("JSON output format detected:")
                    print(json.dumps(result['understand_result'], indent=2))
            except Exception as e:
                print(f"❌ Error: {e}")

        asyncio.run(run_test())