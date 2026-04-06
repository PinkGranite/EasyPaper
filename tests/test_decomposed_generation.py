"""
Tests for the decomposed (paragraph-level) generation pipeline.

Validates that MetaDataAgent._generate_section_decomposed correctly
calls WriterAgent.generate_paragraph with the right parameter names.
"""
import inspect
import pytest


class TestGenerateParagraphSignature:
    """Verify the call-site in _generate_section_decomposed matches
    WriterAgent.generate_paragraph's actual signature."""

    def test_generate_paragraph_accepts_valid_refs_kwarg(self):
        """WriterAgent.generate_paragraph must accept 'valid_refs' as a keyword."""
        from src.agents.writer_agent.writer_agent import WriterAgent
        sig = inspect.signature(WriterAgent.generate_paragraph)
        assert "valid_refs" in sig.parameters, (
            f"generate_paragraph missing 'valid_refs'; params={list(sig.parameters)}"
        )

    def test_caller_does_not_pass_unexpected_kwargs(self):
        """_generate_section_decomposed must NOT pass kwargs that
        generate_paragraph does not accept (e.g. memory, peers,
        valid_citation_keys)."""
        from src.agents.writer_agent.writer_agent import WriterAgent
        sig = inspect.signature(WriterAgent.generate_paragraph)
        accepted = set(sig.parameters.keys()) - {"self"}

        # These are the kwargs that _generate_section_decomposed passes.
        # If any of them are not in the accepted set, the call will TypeError.
        caller_kwargs = {
            "paragraph_prompt",
            "section_type",
            "valid_refs",
            "paragraph_index",
            "claim_id",
        }
        unexpected = caller_kwargs - accepted
        assert not unexpected, (
            f"Caller passes kwargs not accepted by generate_paragraph: {unexpected}"
        )

    def test_caller_does_not_use_valid_citation_keys_for_writer(self):
        """Ensure _generate_section_decomposed uses 'valid_refs', not 
        'valid_citation_keys' when calling generate_paragraph or
        generate_from_template."""
        import ast, textwrap
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent._generate_section_decomposed)
        tree = ast.parse(textwrap.dedent(source))

        writer_methods = {"generate_paragraph", "generate_from_template"}
        bad_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                attr = getattr(node.func, "attr", "")
                if attr in writer_methods:
                    for kw in node.keywords:
                        if kw.arg == "valid_citation_keys":
                            bad_calls.append(f"{attr}() uses 'valid_citation_keys'")

        assert not bad_calls, (
            "_generate_section_decomposed passes 'valid_citation_keys' "
            f"to writer methods — should be 'valid_refs': {bad_calls}"
        )

    def test_caller_does_not_pass_memory_or_peers(self):
        """Ensure _generate_section_decomposed does NOT pass 'memory' or 
        'peers' kwargs to generate_paragraph (they are not accepted)."""
        import ast, textwrap
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent._generate_section_decomposed)
        tree = ast.parse(textwrap.dedent(source))

        # Find calls to generate_paragraph and check for forbidden kwargs
        forbidden = {"memory", "peers"}
        found_forbidden = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                attr = getattr(node.func, "attr", "")
                if attr == "generate_paragraph":
                    for kw in node.keywords:
                        if kw.arg in forbidden:
                            found_forbidden.add(kw.arg)

        assert not found_forbidden, (
            f"_generate_section_decomposed passes forbidden kwargs to "
            f"generate_paragraph: {found_forbidden}"
        )
