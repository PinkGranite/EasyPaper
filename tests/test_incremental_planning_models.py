"""
Tests for incremental section planning model extensions and planner steps.
- **Description**:
    - Phase 1: Verify SectionPlan.mission/key_content and
      SubSectionPlan.mission/key_themes/depends_on/transition_from_previous fields.
    - Phase 2: Verify Step 1 prompt expansion and parsing.
    - Phase 3+: Tests for new planner steps are added incrementally.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.planner_agent.models import (
    SectionPlan,
    SubSectionPlan,
    ParagraphPlan,
    PaperPlan,
    PlanRequest,
)
from src.agents.planner_agent.planner_agent import (
    PlannerAgent,
    STEP1_STRUCTURE_USER,
)


# =========================================================================
# Phase 1: Model field extensions
# =========================================================================


class TestSectionPlanMissionFields:
    """SectionPlan should carry mission and key_content from Step 1."""

    def test_mission_default_empty(self):
        sp = SectionPlan(section_type="method")
        assert sp.mission == ""

    def test_key_content_default_empty_list(self):
        sp = SectionPlan(section_type="method")
        assert sp.key_content == []

    def test_mission_set(self):
        sp = SectionPlan(
            section_type="method",
            mission="Describe the proposed architecture and training procedure.",
        )
        assert sp.mission == "Describe the proposed architecture and training procedure."

    def test_key_content_set(self):
        sp = SectionPlan(
            section_type="method",
            key_content=["model architecture", "loss function", "training details"],
        )
        assert sp.key_content == ["model architecture", "loss function", "training details"]

    def test_serialization_roundtrip(self):
        sp = SectionPlan(
            section_type="experiment",
            section_title="Experiments",
            mission="Validate the hypothesis with ablation studies.",
            key_content=["dataset selection", "baselines", "ablation protocol"],
        )
        data = sp.model_dump()
        assert data["mission"] == "Validate the hypothesis with ablation studies."
        assert data["key_content"] == ["dataset selection", "baselines", "ablation protocol"]
        restored = SectionPlan(**data)
        assert restored.mission == sp.mission
        assert restored.key_content == sp.key_content

    def test_backward_compat_no_mission(self):
        """Old serialized data without mission/key_content should still work."""
        old_data = {
            "section_type": "introduction",
            "section_title": "Introduction",
            "paragraphs": [],
        }
        sp = SectionPlan(**old_data)
        assert sp.mission == ""
        assert sp.key_content == []


class TestSubSectionPlanNewFields:
    """SubSectionPlan should carry mission, key_themes, depends_on, transition."""

    def test_defaults(self):
        sub = SubSectionPlan(title="Self-Attention")
        assert sub.mission == ""
        assert sub.key_themes == []
        assert sub.depends_on == []
        assert sub.transition_from_previous == ""

    def test_full_construction(self):
        sub = SubSectionPlan(
            title="Self-Attention Mechanism",
            mission="Explain the core self-attention computation.",
            key_themes=["query-key-value", "scaled dot-product", "multi-head"],
            depends_on=["Model Overview"],
            transition_from_previous="Building on the overall architecture...",
        )
        assert sub.mission == "Explain the core self-attention computation."
        assert len(sub.key_themes) == 3
        assert sub.depends_on == ["Model Overview"]
        assert sub.transition_from_previous.startswith("Building")

    def test_serialization_roundtrip(self):
        sub = SubSectionPlan(
            title="Training Procedure",
            mission="Detail the two-stage training.",
            key_themes=["pre-training", "fine-tuning"],
            depends_on=["Model Architecture"],
            transition_from_previous="After defining the architecture, we describe...",
            paragraphs=[ParagraphPlan(key_point="Pre-training on large corpus")],
        )
        data = sub.model_dump()
        assert data["mission"] == "Detail the two-stage training."
        assert data["key_themes"] == ["pre-training", "fine-tuning"]
        assert data["depends_on"] == ["Model Architecture"]
        assert "After defining" in data["transition_from_previous"]
        restored = SubSectionPlan(**data)
        assert restored.mission == sub.mission
        assert restored.key_themes == sub.key_themes
        assert len(restored.paragraphs) == 1

    def test_backward_compat_no_new_fields(self):
        old_data = {
            "title": "Results",
            "paragraphs": [],
        }
        sub = SubSectionPlan(**old_data)
        assert sub.mission == ""
        assert sub.key_themes == []
        assert sub.depends_on == []
        assert sub.transition_from_previous == ""


class TestPaperPlanWithNewFields:
    """PaperPlan should correctly contain sections with new fields."""

    def test_section_with_mission_in_paper_plan(self):
        plan = PaperPlan(
            title="Test Paper",
            sections=[
                SectionPlan(
                    section_type="method",
                    section_title="Method",
                    mission="Present the proposed method.",
                    key_content=["architecture", "training"],
                    subsections=[
                        SubSectionPlan(
                            title="Architecture",
                            mission="Describe model components.",
                            key_themes=["encoder", "decoder"],
                            depends_on=[],
                        ),
                        SubSectionPlan(
                            title="Training",
                            mission="Describe training procedure.",
                            key_themes=["loss", "optimizer"],
                            depends_on=["Architecture"],
                            transition_from_previous="Given the architecture...",
                        ),
                    ],
                ),
            ],
        )
        method = plan.get_section("method")
        assert method is not None
        assert method.mission == "Present the proposed method."
        assert len(method.subsections) == 2
        assert method.subsections[1].depends_on == ["Architecture"]


# =========================================================================
# Phase 2: Step 1 Expansion (prompt + parsing)
# =========================================================================


@pytest.fixture
def planner():
    """Create a PlannerAgent with mocked config (no real LLM needed)."""
    config = MagicMock()
    config.model_name = "test-model"
    config.api_key = "test-key"
    config.base_url = "http://localhost:9999"
    agent = PlannerAgent.__new__(PlannerAgent)
    agent.config = config
    agent.model_name = config.model_name
    agent.vlm_service = None
    agent._last_plan = None
    return agent


class TestStep1PromptContainsMissionSchema:
    """Step 1 prompt template should request mission/key_content per section."""

    def test_prompt_template_contains_mission(self):
        assert "mission" in STEP1_STRUCTURE_USER

    def test_prompt_template_contains_key_content(self):
        assert "key_content" in STEP1_STRUCTURE_USER


class TestStep1ParsingStoresMission:
    """Step 1 parsing should extract mission/key_content and store on SectionPlan."""

    @pytest.mark.asyncio
    async def test_step1_stores_mission_and_key_content(self, planner):
        """When LLM returns mission/key_content, they should be stored on SectionPlan."""
        mock_step1_response = {
            "paper_type": "empirical",
            "contributions": ["Proposed BLIP-2"],
            "narrative_style": "technical",
            "sections": [
                {
                    "section_type": "abstract",
                    "section_title": "Abstract",
                    "mission": "Summarize the paper contributions.",
                    "key_content": ["BLIP-2 overview", "key results"],
                },
                {
                    "section_type": "introduction",
                    "section_title": "Introduction",
                    "mission": "Motivate the problem and present contributions.",
                    "key_content": ["vision-language gap", "Q-Former proposal", "efficiency gains"],
                },
                {
                    "section_type": "method",
                    "section_title": "Method",
                    "mission": "Describe the Q-Former architecture and training.",
                    "key_content": ["architecture overview", "two-stage training"],
                },
                {
                    "section_type": "experiment",
                    "section_title": "Experiments",
                    "mission": "Validate across VQA, captioning, and retrieval.",
                    "key_content": ["dataset selection", "baselines", "ablations"],
                },
                {
                    "section_type": "conclusion",
                    "section_title": "Conclusion",
                    "mission": "Summarize findings.",
                    "key_content": ["main results", "future work"],
                },
            ],
            "structure_rationale": "Standard empirical structure.",
            "abstract_focus": "Efficiency + performance.",
        }

        mock_step2_response = {
            "total_target": 30,
            "rationale": "Standard",
            "section_allocation": {},
        }

        mock_step3_response = {
            "paragraphs": [
                {"key_point": "Test paragraph", "approx_sentences": 4, "role": "evidence"},
            ],
            "topic_clusters": [],
            "sectioning_recommended": False,
        }

        call_count = {"n": 0}

        async def mock_llm_json_call(system, user, label, **kwargs):
            call_count["n"] += 1
            if "step1" in label:
                return mock_step1_response
            elif "step2" in label:
                return mock_step2_response
            else:
                return mock_step3_response

        planner._llm_json_call = mock_llm_json_call
        planner._assign_figures_to_sections = MagicMock(return_value={})

        request = PlanRequest(
            title="BLIP-2",
            idea_hypothesis="Bridge vision and language via Q-Former.",
            method="Q-Former architecture with frozen encoders.",
            data="COCO, VQA v2, etc.",
            experiments="Zero-shot VQA, captioning, retrieval.",
            references=["ref1", "ref2"],
        )

        plan = await planner.create_plan(request)
        assert plan is not None

        method_section = plan.get_section("method")
        assert method_section is not None
        assert method_section.mission == "Describe the Q-Former architecture and training."
        assert method_section.key_content == ["architecture overview", "two-stage training"]

        intro_section = plan.get_section("introduction")
        assert intro_section is not None
        assert intro_section.mission == "Motivate the problem and present contributions."
        assert "Q-Former proposal" in intro_section.key_content

    @pytest.mark.asyncio
    async def test_step1_missing_mission_falls_back(self, planner):
        """When LLM omits mission/key_content, defaults should apply."""
        mock_step1_response = {
            "paper_type": "empirical",
            "contributions": ["C1"],
            "narrative_style": "technical",
            "sections": [
                {"section_type": "abstract", "section_title": "Abstract"},
                {"section_type": "introduction", "section_title": "Introduction"},
                {"section_type": "method", "section_title": "Method"},
                {"section_type": "conclusion", "section_title": "Conclusion"},
            ],
            "structure_rationale": "Minimal.",
            "abstract_focus": "Core.",
        }

        mock_step2_response = {"total_target": 10, "section_allocation": {}}
        mock_step3_response = {
            "paragraphs": [{"key_point": "p1", "approx_sentences": 3}],
            "topic_clusters": [],
            "sectioning_recommended": False,
        }

        async def mock_llm_json_call(system, user, label, **kwargs):
            if "step1" in label:
                return mock_step1_response
            elif "step2" in label:
                return mock_step2_response
            else:
                return mock_step3_response

        planner._llm_json_call = mock_llm_json_call
        planner._assign_figures_to_sections = MagicMock(return_value={})

        request = PlanRequest(
            title="Test",
            idea_hypothesis="hypothesis",
            method="method",
            data="data",
            experiments="experiments",
        )
        plan = await planner.create_plan(request)
        assert plan is not None

        method_section = plan.get_section("method")
        assert method_section is not None
        assert method_section.mission == ""
        assert method_section.key_content == []


# =========================================================================
# Phase 3: Step 4 -- _decide_section_structure()
# =========================================================================


class TestDecideSectionStructure:
    """_decide_section_structure() should use mission/key_content (not raw metadata)."""

    @pytest.mark.asyncio
    async def test_simple_section_no_subsections(self, planner):
        """Section with short mission and few key_content => no subsections."""
        section = SectionPlan(
            section_type="introduction",
            section_title="Introduction",
            mission="Motivate the problem and present contributions.",
            key_content=["background", "gap"],
        )

        async def mock_llm(system, user, label, **kwargs):
            return {
                "needs_subsections": False,
                "reasoning": "Short section, no need for subsections.",
            }

        planner._llm_json_call = mock_llm

        result = await planner._decide_section_structure(
            section=section,
            paper_type="empirical",
            contributions=["C1"],
            venue="ICML 2025",
            word_budget=800,
            prior_sections_summary="",
        )
        assert result["needs_subsections"] is False
        assert "reasoning" in result

    @pytest.mark.asyncio
    async def test_complex_section_with_subsections(self, planner):
        """Section with complex mission and many key_content => subsections."""
        section = SectionPlan(
            section_type="method",
            section_title="Method",
            mission="Describe the Q-Former architecture and two-stage training procedure.",
            key_content=[
                "Q-Former architecture overview",
                "Learnable queries",
                "Self-attention mechanism",
                "Cross-attention to frozen image encoder",
                "Stage 1: representation learning",
                "Stage 2: generative learning",
            ],
        )

        async def mock_llm(system, user, label, **kwargs):
            return {
                "needs_subsections": True,
                "reasoning": "Complex architecture + multi-stage training warrants subsections.",
                "subsections": [
                    {
                        "title": "Q-Former Architecture",
                        "mission": "Present the learnable query mechanism.",
                        "key_themes": ["learnable queries", "self-attention", "cross-attention"],
                        "depends_on": [],
                        "approx_paragraphs": 3,
                    },
                    {
                        "title": "Two-Stage Training",
                        "mission": "Detail representation and generative learning stages.",
                        "key_themes": ["stage 1", "stage 2"],
                        "depends_on": ["Q-Former Architecture"],
                        "approx_paragraphs": 3,
                    },
                ],
                "cross_subsection_transitions": [
                    "Building on the architecture, we now describe the training procedure.",
                ],
            }

        planner._llm_json_call = mock_llm

        result = await planner._decide_section_structure(
            section=section,
            paper_type="empirical",
            contributions=["Proposed BLIP-2"],
            venue="ICML 2025",
            word_budget=1500,
            prior_sections_summary="introduction: 5 paras, flat",
        )
        assert result["needs_subsections"] is True
        assert len(result["subsections"]) == 2
        assert result["subsections"][0]["title"] == "Q-Former Architecture"
        assert result["subsections"][1]["depends_on"] == ["Q-Former Architecture"]

    @pytest.mark.asyncio
    async def test_prior_sections_summary_accumulated(self, planner):
        """Prompt should contain prior_sections_summary for cumulative context."""
        captured_prompts = []

        async def mock_llm(system, user, label, **kwargs):
            captured_prompts.append(user)
            return {"needs_subsections": False, "reasoning": "Simple."}

        planner._llm_json_call = mock_llm

        section = SectionPlan(
            section_type="experiment",
            section_title="Experiments",
            mission="Validate across tasks.",
            key_content=["VQA", "captioning"],
        )

        await planner._decide_section_structure(
            section=section,
            paper_type="empirical",
            contributions=["C1"],
            venue="NeurIPS",
            word_budget=1000,
            prior_sections_summary="introduction: 4 paras, flat; method: 6 paras, 2 subsections",
        )

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        assert "introduction: 4 paras, flat" in prompt
        assert "method: 6 paras, 2 subsections" in prompt

    @pytest.mark.asyncio
    async def test_prompt_does_not_contain_raw_metadata(self, planner):
        """Step 4 prompt should NOT contain raw idea_hypothesis/method/data/experiments."""
        captured_prompts = []

        async def mock_llm(system, user, label, **kwargs):
            captured_prompts.append(user)
            return {"needs_subsections": False, "reasoning": "Simple."}

        planner._llm_json_call = mock_llm

        section = SectionPlan(
            section_type="method",
            section_title="Method",
            mission="Describe the architecture.",
            key_content=["encoder design"],
        )

        await planner._decide_section_structure(
            section=section,
            paper_type="empirical",
            contributions=["C1"],
            venue="ICML",
            word_budget=1000,
            prior_sections_summary="",
        )

        prompt = captured_prompts[0]
        assert "idea_hypothesis" not in prompt.lower() or "idea/hypothesis" not in prompt.lower()
        assert "Describe the architecture." in prompt
        assert "encoder design" in prompt


# =========================================================================
# Phase 4: Step 5a -- _plan_flat_paragraphs()
# =========================================================================


class TestPlanFlatParagraphs:
    """_plan_flat_paragraphs() generates ParagraphPlans using mission/key_content."""

    @pytest.mark.asyncio
    async def test_returns_valid_paragraph_plans(self, planner):
        section = SectionPlan(
            section_type="introduction",
            section_title="Introduction",
            mission="Motivate the problem and present contributions.",
            key_content=["vision-language gap", "Q-Former proposal", "efficiency gains"],
        )

        async def mock_llm(system, user, label, **kwargs):
            return {
                "paragraphs": [
                    {
                        "key_point": "Vision-language models face a modality gap.",
                        "supporting_points": ["prior models need end-to-end training"],
                        "approx_sentences": 5,
                        "role": "topic",
                    },
                    {
                        "key_point": "We propose Q-Former to bridge this gap.",
                        "supporting_points": ["lightweight", "efficient"],
                        "approx_sentences": 4,
                        "role": "evidence",
                    },
                    {
                        "key_point": "Our contributions include...",
                        "supporting_points": ["architecture", "two-stage training"],
                        "approx_sentences": 3,
                        "role": "conclusion",
                    },
                ],
            }

        planner._llm_json_call = mock_llm

        paragraphs = await planner._plan_flat_paragraphs(
            section=section,
            word_budget=800,
            reference_keys=["ref1", "ref2"],
            prior_key_points="",
        )
        assert len(paragraphs) == 3
        assert all(isinstance(p, ParagraphPlan) for p in paragraphs)
        assert paragraphs[0].key_point == "Vision-language models face a modality gap."
        assert paragraphs[1].approx_sentences == 4

    @pytest.mark.asyncio
    async def test_prompt_uses_mission_not_raw_metadata(self, planner):
        captured_prompts = []

        async def mock_llm(system, user, label, **kwargs):
            captured_prompts.append(user)
            return {"paragraphs": [{"key_point": "p1", "approx_sentences": 3}]}

        planner._llm_json_call = mock_llm

        section = SectionPlan(
            section_type="method",
            section_title="Method",
            mission="Describe the architecture.",
            key_content=["encoder design", "decoder design"],
        )

        await planner._plan_flat_paragraphs(
            section=section,
            word_budget=1000,
            reference_keys=["ref1"],
            prior_key_points="introduction covered background and gap",
        )

        prompt = captured_prompts[0]
        assert "Describe the architecture." in prompt
        assert "encoder design" in prompt
        assert "introduction covered background" in prompt


# =========================================================================
# Phase 5: Step 5b -- _plan_subsection_paragraphs()
# =========================================================================


class TestPlanSubsectionParagraphs:
    """_plan_subsection_paragraphs() generates paragraph plans per subsection with cumulative key_points."""

    @pytest.mark.asyncio
    async def test_cumulative_key_points_across_subsections(self, planner):
        """sub_2's prompt should include key_points from sub_0 and sub_1."""
        captured_prompts = []

        call_idx = {"n": 0}

        async def mock_llm(system, user, label, **kwargs):
            captured_prompts.append(user)
            idx = call_idx["n"]
            call_idx["n"] += 1
            responses = [
                {
                    "paragraphs": [
                        {"key_point": "Architecture overview", "approx_sentences": 4},
                        {"key_point": "Learnable queries", "approx_sentences": 4},
                    ],
                    "subsection_key_points": ["Q-Former uses learnable queries with cross-attention"],
                },
                {
                    "paragraphs": [
                        {"key_point": "Stage 1 representation learning", "approx_sentences": 5},
                    ],
                    "subsection_key_points": ["Stage 1 trains on image-text pairs"],
                },
                {
                    "paragraphs": [
                        {"key_point": "Stage 2 generative learning", "approx_sentences": 4},
                    ],
                    "subsection_key_points": ["Stage 2 connects to LLM decoder"],
                },
            ]
            return responses[idx]

        planner._llm_json_call = mock_llm

        section = SectionPlan(
            section_type="method",
            section_title="Method",
            mission="Describe Q-Former and two-stage training.",
            key_content=["architecture", "stage 1", "stage 2"],
        )

        subsection_structure = {
            "needs_subsections": True,
            "subsections": [
                {
                    "title": "Q-Former Architecture",
                    "mission": "Present the learnable query mechanism.",
                    "key_themes": ["queries", "cross-attention"],
                    "depends_on": [],
                    "approx_paragraphs": 2,
                },
                {
                    "title": "Stage 1: Representation Learning",
                    "mission": "Detail representation learning stage.",
                    "key_themes": ["image-text contrastive", "matching"],
                    "depends_on": ["Q-Former Architecture"],
                    "approx_paragraphs": 1,
                },
                {
                    "title": "Stage 2: Generative Learning",
                    "mission": "Detail generative learning stage.",
                    "key_themes": ["LLM connection", "generation"],
                    "depends_on": ["Stage 1: Representation Learning"],
                    "approx_paragraphs": 1,
                },
            ],
            "cross_subsection_transitions": [
                "Building on the architecture, we describe stage 1.",
                "After representation learning, we detail stage 2.",
            ],
        }

        result = await planner._plan_subsection_paragraphs(
            section=section,
            subsection_structure=subsection_structure,
            reference_keys=["ref1", "ref2"],
        )

        assert len(result) == 3
        assert all(isinstance(s, SubSectionPlan) for s in result)
        assert result[0].title == "Q-Former Architecture"
        assert len(result[0].paragraphs) == 2

        assert len(captured_prompts) == 3
        assert "Q-Former uses learnable queries" in captured_prompts[1]
        assert "Q-Former uses learnable queries" in captured_prompts[2]
        assert "Stage 1 trains on image-text pairs" in captured_prompts[2]

    @pytest.mark.asyncio
    async def test_cross_subsection_transition_passed(self, planner):
        """Subsections after the first should receive transition guidance."""
        captured_prompts = []

        async def mock_llm(system, user, label, **kwargs):
            captured_prompts.append(user)
            return {
                "paragraphs": [{"key_point": "p", "approx_sentences": 3}],
                "subsection_key_points": ["key point"],
            }

        planner._llm_json_call = mock_llm

        section = SectionPlan(
            section_type="method",
            section_title="Method",
            mission="Test.",
            key_content=["a", "b"],
        )

        subsection_structure = {
            "needs_subsections": True,
            "subsections": [
                {"title": "Sub A", "mission": "A", "key_themes": [], "depends_on": [], "approx_paragraphs": 1},
                {"title": "Sub B", "mission": "B", "key_themes": [], "depends_on": ["Sub A"], "approx_paragraphs": 1},
            ],
            "cross_subsection_transitions": [
                "Transitioning from A to B.",
            ],
        }

        await planner._plan_subsection_paragraphs(
            section=section,
            subsection_structure=subsection_structure,
            reference_keys=[],
        )

        assert "Transitioning from A to B" in captured_prompts[1]

    @pytest.mark.asyncio
    async def test_output_subsection_plans_have_metadata(self, planner):
        """Returned SubSectionPlans should carry mission/key_themes/depends_on."""
        async def mock_llm(system, user, label, **kwargs):
            return {
                "paragraphs": [{"key_point": "point", "approx_sentences": 3}],
                "subsection_key_points": ["summary"],
            }

        planner._llm_json_call = mock_llm

        section = SectionPlan(
            section_type="experiment",
            section_title="Experiments",
            mission="Validate results.",
            key_content=["VQA", "captioning"],
        )

        subsection_structure = {
            "needs_subsections": True,
            "subsections": [
                {
                    "title": "VQA Evaluation",
                    "mission": "Evaluate on VQA benchmarks.",
                    "key_themes": ["VQAv2", "OK-VQA"],
                    "depends_on": [],
                    "approx_paragraphs": 2,
                },
            ],
            "cross_subsection_transitions": [],
        }

        result = await planner._plan_subsection_paragraphs(
            section=section,
            subsection_structure=subsection_structure,
            reference_keys=[],
        )

        assert result[0].mission == "Evaluate on VQA benchmarks."
        assert result[0].key_themes == ["VQAv2", "OK-VQA"]
        assert result[0].depends_on == []


# =========================================================================
# Phase 6: Integration -- Rewired create_plan()
# =========================================================================


class TestCreatePlanIntegration:
    """create_plan() should use Steps 4/5a/5b instead of old Step 3."""

    @pytest.mark.asyncio
    async def test_section_with_subsections_flows_through_step4_5b(self, planner):
        """Section where Step 4 decides subsections should go through 5b."""
        call_labels = []

        async def mock_llm(system, user, label, **kwargs):
            call_labels.append(label)
            if "step1" in label:
                return {
                    "paper_type": "empirical",
                    "contributions": ["C1"],
                    "narrative_style": "technical",
                    "sections": [
                        {"section_type": "abstract", "section_title": "Abstract",
                         "mission": "Summarize.", "key_content": ["summary"]},
                        {"section_type": "introduction", "section_title": "Introduction",
                         "mission": "Motivate.", "key_content": ["background"]},
                        {"section_type": "method", "section_title": "Method",
                         "mission": "Describe architecture and training.",
                         "key_content": ["arch", "stage 1", "stage 2", "modules", "losses", "evaluation"]},
                        {"section_type": "conclusion", "section_title": "Conclusion",
                         "mission": "Conclude.", "key_content": ["summary"]},
                    ],
                    "structure_rationale": "Standard.",
                    "abstract_focus": "Core.",
                }
            elif "step2" in label:
                return {"total_target": 20, "section_allocation": {}}
            elif "element_assignment" in label:
                return {}
            elif "step4" in label:
                if "method" in label:
                    return {
                        "needs_subsections": True,
                        "reasoning": "Complex content.",
                        "subsections": [
                            {"title": "Architecture", "mission": "Describe arch.",
                             "key_themes": ["encoder"], "depends_on": [], "approx_paragraphs": 2},
                            {"title": "Training", "mission": "Describe training.",
                             "key_themes": ["stages"], "depends_on": ["Architecture"], "approx_paragraphs": 2},
                        ],
                        "cross_subsection_transitions": ["After architecture, training."],
                    }
                return {"needs_subsections": False, "reasoning": "Simple."}
            elif "step5a" in label:
                return {
                    "paragraphs": [
                        {"key_point": "p1", "approx_sentences": 4},
                        {"key_point": "p2", "approx_sentences": 3},
                    ],
                }
            elif "step5b" in label:
                return {
                    "paragraphs": [
                        {"key_point": "sub_p", "approx_sentences": 4},
                    ],
                    "subsection_key_points": ["key summary"],
                }
            return {}

        planner._llm_json_call = mock_llm
        planner._assign_figures_to_sections = MagicMock(return_value={})

        request = PlanRequest(
            title="Test Paper",
            idea_hypothesis="hypothesis",
            method="method",
            data="data",
            experiments="experiments",
        )

        plan = await planner.create_plan(request)

        method = plan.get_section("method")
        assert method is not None
        assert len(method.subsections) == 2
        assert method.subsections[0].title == "Architecture"
        assert method.subsections[0].mission == "Describe arch."
        assert len(method.subsections[0].paragraphs) >= 1

        assert any("step4" in l for l in call_labels)
        assert any("step5b" in l for l in call_labels)

    @pytest.mark.asyncio
    async def test_flat_section_flows_through_step5a(self, planner):
        """Section where Step 4 decides no subsections should go through 5a."""
        call_labels = []

        async def mock_llm(system, user, label, **kwargs):
            call_labels.append(label)
            if "step1" in label:
                return {
                    "paper_type": "empirical",
                    "contributions": ["C1"],
                    "narrative_style": "technical",
                    "sections": [
                        {"section_type": "abstract", "section_title": "Abstract",
                         "mission": "Summarize.", "key_content": ["summary"]},
                        {"section_type": "introduction", "section_title": "Introduction",
                         "mission": "Motivate.", "key_content": ["background"]},
                        {"section_type": "conclusion", "section_title": "Conclusion",
                         "mission": "Conclude.", "key_content": ["summary"]},
                    ],
                    "structure_rationale": "Minimal.",
                    "abstract_focus": "Core.",
                }
            elif "step2" in label:
                return {"total_target": 10, "section_allocation": {}}
            elif "element_assignment" in label:
                return {}
            elif "step4" in label:
                return {"needs_subsections": False, "reasoning": "Simple."}
            elif "step5a" in label:
                return {
                    "paragraphs": [
                        {"key_point": "intro_p1", "approx_sentences": 4},
                        {"key_point": "intro_p2", "approx_sentences": 3},
                    ],
                }
            return {}

        planner._llm_json_call = mock_llm
        planner._assign_figures_to_sections = MagicMock(return_value={})

        request = PlanRequest(
            title="Test",
            idea_hypothesis="h",
            method="m",
            data="d",
            experiments="e",
        )

        plan = await planner.create_plan(request)

        intro = plan.get_section("introduction")
        assert intro is not None
        assert len(intro.subsections) == 0
        assert len(intro.paragraphs) >= 1

        assert any("step5a" in l for l in call_labels)
        assert not any("step3" in l for l in call_labels)

    @pytest.mark.asyncio
    async def test_old_step3_prompt_not_used(self, planner):
        """The old STEP3_SECTION_USER prompt should NOT be used anymore."""
        captured_prompts = []

        async def mock_llm(system, user, label, **kwargs):
            captured_prompts.append((label, user))
            if "step1" in label:
                return {
                    "paper_type": "empirical",
                    "contributions": ["C1"],
                    "narrative_style": "technical",
                    "sections": [
                        {"section_type": "abstract", "section_title": "Abstract",
                         "mission": "S.", "key_content": ["s"]},
                        {"section_type": "introduction", "section_title": "Introduction",
                         "mission": "M.", "key_content": ["b"]},
                        {"section_type": "conclusion", "section_title": "Conclusion",
                         "mission": "C.", "key_content": ["c"]},
                    ],
                    "structure_rationale": ".",
                    "abstract_focus": ".",
                }
            elif "step2" in label:
                return {"total_target": 5, "section_allocation": {}}
            elif "element_assignment" in label:
                return {}
            elif "step4" in label:
                return {"needs_subsections": False, "reasoning": "."}
            elif "step5a" in label:
                return {"paragraphs": [{"key_point": "p", "approx_sentences": 3}]}
            return {}

        planner._llm_json_call = mock_llm
        planner._assign_figures_to_sections = MagicMock(return_value={})

        request = PlanRequest(
            title="T", idea_hypothesis="h", method="m", data="d", experiments="e",
        )
        await planner.create_plan(request)

        labels = [l for l, _ in captured_prompts]
        assert not any("step3_" in l for l in labels), (
            f"Old step3 labels found: {[l for l in labels if 'step3' in l]}"
        )
