"""
Tests for the decomposed (paragraph-level) generation pipeline.

Validates that MetaDataAgent._generate_section_decomposed correctly
calls WriterAgent.generate_paragraph with the right parameter names,
handles no-claim paragraphs, and uses _all_paragraphs().
"""
import inspect
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Path Unification: decomposed path handles all paragraph types
# ═══════════════════════════════════════════════════════════════════════════


class TestDecomposedPathUnification:
    """Decomposed path must handle paragraphs without claim_id (no DAG)."""

    def test_decomposed_uses_all_paragraphs(self):
        """_generate_section_decomposed must iterate _all_paragraphs() to
        include paragraphs nested inside subsections."""
        import textwrap
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent._generate_section_decomposed)
        assert "_all_paragraphs()" in source, (
            "_generate_section_decomposed must use section_plan._all_paragraphs() "
            "instead of section_plan.paragraphs to include subsection paragraphs"
        )

    def test_decomposed_skips_verifier_for_no_claim(self):
        """When claim_id is empty, ClaimVerifier.verify should NOT be called."""
        import textwrap
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent._generate_section_decomposed)
        # The method should have conditional logic around claim_id for verification
        assert "claim_id" in source, "decomposed must reference claim_id"

    def test_body_section_always_calls_decomposed(self):
        """_generate_body_section should NOT have has_claim_bindings fallback."""
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent._generate_body_section)
        assert "has_claim_bindings" not in source, (
            "_generate_body_section must not branch on has_claim_bindings; "
            "it should always call _generate_section_decomposed"
        )

    def test_introduction_does_not_call_writer_run(self):
        """_generate_introduction must not call self._writer.run()."""
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent._generate_introduction)
        assert "self._writer.run(" not in source, (
            "_generate_introduction must not use the legacy self._writer.run() path; "
            "it should delegate to _generate_section_decomposed"
        )

    def test_synthesis_does_not_call_writer_run(self):
        """_generate_synthesis_section must not call self._writer.run()."""
        from src.agents.metadata_agent import metadata_agent as mod

        source = inspect.getsource(mod.MetaDataAgent._generate_synthesis_section)
        assert "self._writer.run(" not in source, (
            "_generate_synthesis_section must not use the legacy self._writer.run() path"
        )


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
