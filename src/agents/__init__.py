from typing import Dict, Optional, TYPE_CHECKING
from .base import BaseAgent
from .parse_agent import ParseAgent
from .template_agent import TemplateParserAgent
from .commander_agent import CommanderAgent
from .writer_agent import WriterAgent
from .typesetter_agent import TypesetterAgent
from .metadata_agent import MetaDataAgent
from .reviewer_agent import ReviewerAgent
from .planner_agent import PlannerAgent
from .vlm_review_agent import VLMReviewAgent
from .shared.vlm_service import VLMService
from config.schema import AgentConfig, VLMServiceConfig

if TYPE_CHECKING:
    from ..skills.registry import SkillRegistry

AGENT_DICT = {
    "paper_parser": ParseAgent,
    "template_parser": TemplateParserAgent,
    "commander": CommanderAgent,
    "writer": WriterAgent,
    "typesetter": TypesetterAgent,
    "metadata": MetaDataAgent,
    "reviewer": ReviewerAgent,
    "planner": PlannerAgent,
    "vlm_review": VLMReviewAgent,
}

def initialize_agents(
    agents: list[AgentConfig],
    skill_registry: Optional["SkillRegistry"] = None,
    global_tools_config=None,
    vlm_service_config: Optional[VLMServiceConfig] = None,
) -> dict[str, BaseAgent]:
    """
    Initialize agents from configuration.

    - **Args**:
        - `agents` (list[AgentConfig]): Agent configuration list
        - `skill_registry` (SkillRegistry, optional): Global skill registry
          passed to ReviewerAgent and MetaDataAgent
        - `global_tools_config` (ToolsConfig, optional): Global tools config
          from AppConfig.tools, used as default for ReAct-enabled agents
        - `vlm_service_config` (VLMServiceConfig, optional): Shared VLM service
          configuration for Planner and VLMReviewAgent
    """
    REACT_AGENTS = {"metadata", "writer"}

    # Create shared VLM service if configured
    vlm_service: Optional[VLMService] = None
    if vlm_service_config and vlm_service_config.enabled:
        vlm_service = VLMService(
            provider=vlm_service_config.provider,
            model=vlm_service_config.model,
            api_key=vlm_service_config.api_key,
            base_url=vlm_service_config.base_url,
        )

    agent_dict = {}
    for agent_config in agents:
        if agent_config.name in AGENT_DICT:
            agent_class = AGENT_DICT[agent_config.name]

            if agent_config.name == "vlm_review" and agent_config.vlm_review_config:
                agent_instance = agent_class(
                    model_config=agent_config.model,
                    vlm_review_config=agent_config.vlm_review_config,
                    vlm_service=vlm_service,
                )
            elif agent_config.name == "planner":
                agent_instance = agent_class(
                    config=agent_config.model,
                    vlm_service=vlm_service,
                )
            elif agent_config.name == "reviewer" and skill_registry is not None:
                agent_instance = agent_class(
                    config=agent_config.model,
                    skill_registry=skill_registry,
                )
            elif agent_config.name in REACT_AGENTS:
                tools_cfg = agent_config.tools_config or global_tools_config
                agent_instance = agent_class(
                    config=agent_config.model,
                    tools_config=tools_cfg,
                )
            else:
                agent_instance = agent_class(agent_config.model)

            # Attach skill_registry to MetaDataAgent after construction
            if agent_config.name == "metadata" and skill_registry is not None:
                agent_instance._skill_registry = skill_registry

            agent_dict[agent_instance.name] = agent_instance
        else:
            raise ValueError(f"Agent {agent_config.name} not found")

    # Inject peer references so MetaDataAgent can call agents directly
    metadata_agent = agent_dict.get("metadata")
    if metadata_agent is not None and hasattr(metadata_agent, "set_peers"):
        metadata_agent.set_peers(agent_dict)

    return agent_dict

def register_agent_routers(app, agents: Dict[str, BaseAgent]):
    """Register agent routers with FastAPI app"""
    for agent_name, agent_instance in agents.items():
        app.include_router(
            agent_instance.router,
            tags=[agent_name],
            prefix=""
        )

__all__ = ["initialize_agents", "register_agent_routers", "BaseAgent", "ParseAgent", "TemplateParserAgent", "CommanderAgent", "WriterAgent", "TypesetterAgent", "MetaDataAgent", "ReviewerAgent", "PlannerAgent", "VLMReviewAgent"]