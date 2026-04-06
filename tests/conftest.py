"""
Shared fixtures for EasyPaper test suite.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.shared.session_memory import SessionMemory


SAMPLE_INTRO_LATEX = r"""
We study the problem of tax policy and firm-level innovation in emerging markets.
Innovation is widely regarded as a key driver of economic growth \cite{romer1990endogenous}.
Recent studies have highlighted the importance of fiscal incentives in promoting
research and development activities \cite{bloom2019toolkit}.
Our work contributes to this literature by examining the causal effect of R\&D
tax credits on firm productivity using a novel panel dataset.
We find that firms receiving tax credits increase their patent output by 15\%
relative to matched controls, with heterogeneous effects across firm size and sector.
These findings have important implications for policymakers designing innovation
support programs in developing economies.
The remainder of this paper is organized as follows. Section 2 reviews related work.
Section 3 describes our methodology. Section 4 presents the data and experimental
design. Section 5 discusses results, and Section 6 concludes.
""".strip()

SAMPLE_METHOD_LATEX = r"""
\subsection{Identification Strategy}
We employ a difference-in-differences (DiD) design exploiting the staggered
adoption of R\&D tax credit programs across provinces.
Let $Y_{it}$ denote the innovation output of firm $i$ in year $t$.
Our baseline specification is:
\begin{equation}
Y_{it} = \alpha + \beta \cdot \text{TaxCredit}_{it} + X_{it}'\gamma + \mu_i + \delta_t + \varepsilon_{it}
\end{equation}
where $\text{TaxCredit}_{it}$ is an indicator for whether firm $i$ received the
tax credit in year $t$, $X_{it}$ is a vector of time-varying controls,
$\mu_i$ and $\delta_t$ are firm and year fixed effects respectively.
""".strip()

SAMPLE_BIBTEX_ENTRIES = [
    r"""@article{romer1990endogenous,
  title={Endogenous technological change},
  author={Romer, Paul M},
  journal={Journal of Political Economy},
  volume={98},
  number={5},
  pages={71--102},
  year={1990}
}""",
    r"""@article{bloom2019toolkit,
  title={A toolkit of policies to promote innovation},
  author={Bloom, Nicholas and Van Reenen, John and Williams, Heidi},
  journal={Journal of Economic Perspectives},
  volume={33},
  number={3},
  pages={163--184},
  year={2019}
}""",
    r"""@article{hall2010patents,
  title={The choice between formal and informal intellectual property},
  author={Hall, Bronwyn and Helmers, Christian and Rogers, Mark and Sena, Vania},
  journal={Journal of Economic Literature},
  year={2010}
}""",
]


@pytest.fixture
def sample_intro_latex() -> str:
    """Sample introduction LaTeX content (~150 words)."""
    return SAMPLE_INTRO_LATEX


@pytest.fixture
def sample_method_latex() -> str:
    """Sample method section LaTeX content (~100 words)."""
    return SAMPLE_METHOD_LATEX


@pytest.fixture
def sample_bibtex_entries():
    """List of BibTeX entry strings."""
    return list(SAMPLE_BIBTEX_ENTRIES)


@pytest.fixture
def sample_session_memory(sample_intro_latex) -> SessionMemory:
    """Pre-populated SessionMemory with intro section."""
    mem = SessionMemory()
    mem.update_section("introduction", sample_intro_latex)
    mem.contributions = [
        "We propose a causal identification strategy for R&D tax credits",
        "Novel panel dataset spanning 10 years and 5000 firms",
    ]
    return mem


@pytest.fixture
def empty_session_memory() -> SessionMemory:
    """Empty SessionMemory for baseline tests."""
    return SessionMemory()


@pytest.fixture
def mock_llm_response():
    """Factory for mock LLM chat completion responses."""
    def _make(content: str = "mock response", usage=None):
        choice = MagicMock()
        choice.message.content = content
        choice.message.tool_calls = None
        choice.message._thinking = None
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = usage or MagicMock(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        return resp
    return _make


@pytest.fixture
def mock_llm_client(mock_llm_response):
    """Mocked LLMClient that returns canned responses."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=mock_llm_response("Generated LaTeX content.")
    )
    return client


@pytest.fixture
def sample_core_refs():
    """Two sample core reference dicts for CoreRefAnalyzer tests."""
    return [
        {
            "ref_id": "smith2020",
            "title": "Deep Learning for Vision",
            "authors": "Smith, A and Doe, B",
            "year": 2020,
            "abstract": "We propose a deep architecture for image classification.",
            "bibtex": '@article{smith2020, title={Deep Learning for Vision}, author={Smith, A and Doe, B}, year={2020}}',
        },
        {
            "ref_id": "jones2021",
            "title": "Robustness in Neural Networks",
            "authors": "Jones, C",
            "year": 2021,
            "abstract": "Adversarial training improves robustness.",
            "bibtex": '@article{jones2021, title={Robustness in Neural Networks}, author={Jones, C}, year={2021}}',
        },
    ]
