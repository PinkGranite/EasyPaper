"""
Load EasyPaper submodules without importing src.agents.__init__ (heavy deps).
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"


def ensure_src_stubs() -> None:
    """Minimal package tree for metadata_agent + shared."""
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    if "src" not in sys.modules:
        m = types.ModuleType("src")
        m.__path__ = [str(_SRC)]  # type: ignore[attr-defined]
        sys.modules["src"] = m
    if "src.agents" not in sys.modules:
        m = types.ModuleType("src.agents")
        m.__path__ = [str(_SRC / "agents")]  # type: ignore[attr-defined]
        sys.modules["src.agents"] = m
    for sub in ("metadata_agent", "shared"):
        name = f"src.agents.{sub}"
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [str(_SRC / "agents" / sub)]  # type: ignore[attr-defined]
            sys.modules[name] = m


def load_metadata_models():
    ensure_src_stubs()
    path = _SRC / "agents" / "metadata_agent" / "models.py"
    name = "src.agents.metadata_agent.models"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_core_ref_analyzer():
    load_metadata_models()
    ensure_src_stubs()
    path = _SRC / "agents" / "shared" / "core_ref_analyzer.py"
    name = "src.agents.shared.core_ref_analyzer"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_research_context_builder():
    load_metadata_models()
    ensure_src_stubs()
    path = _SRC / "agents" / "shared" / "research_context_builder.py"
    name = "src.agents.shared.research_context_builder"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_reference_assignment():
    ensure_src_stubs()
    path = _SRC / "agents" / "shared" / "reference_assignment.py"
    name = "src.agents.shared.reference_assignment"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_planner_models():
    ensure_src_stubs()
    path = _SRC / "agents" / "planner_agent" / "models.py"
    name = "src.agents.planner_agent.models"
    if name in sys.modules:
        return sys.modules[name]
    if "src.agents.planner_agent" not in sys.modules:
        m = types.ModuleType("src.agents.planner_agent")
        m.__path__ = [str(_SRC / "agents" / "planner_agent")]  # type: ignore[attr-defined]
        sys.modules["src.agents.planner_agent"] = m
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
