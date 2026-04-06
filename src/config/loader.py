import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
from .schema import ModelConfig, AgentConfig, AppConfig

# Resolve the project root (.env lives at the repo root)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_loaded_dotenv = False


def load_config() -> AppConfig:
    """
    Load application config from a YAML file.
    - **Description**:
        - Reads AGENT_CONFIG_PATH from the environment and loads the YAML config.
        - Loads .env once (without overriding) so programmatically-set env vars
          (e.g. via ``--easypaper-config``) take precedence over .env defaults.

    - **Returns**:
        - `AppConfig`: The parsed application configuration.
    """
    global _loaded_dotenv
    if not _loaded_dotenv:
        dotenv_path = _PROJECT_ROOT / ".env"
        load_dotenv(dotenv_path=dotenv_path, override=False)
        _loaded_dotenv = True

    config_path = os.getenv("AGENT_CONFIG_PATH", "./configs/dev.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"❌ Config file not found: {config_path}. "
            f"Set environment variable AGENT_CONFIG_PATH to correct path."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AppConfig(**raw)