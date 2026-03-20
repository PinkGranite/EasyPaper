"""
Skill Loader
- **Description**:
    - Loads WritingSkill definitions from YAML files on disk
    - Recursively scans a skills directory for .yaml files
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from .models import WritingSkill

logger = logging.getLogger("uvicorn.error")


class SkillLoader:
    """
    Loads WritingSkill objects from YAML files.

    - **Methods**:
        - `load_directory()`: Recursively scan a directory and load all .yaml files
        - `load_merged()`: Builtin skills plus optional user dir; same `name` → user wins
        - `load_single()`: Load a single YAML file into a WritingSkill
    """

    @staticmethod
    def default_skills_dir() -> Path:
        return Path(__file__).resolve().parents[1] / "assets" / "skills"

    def resolve_skills_dir(self, skills_dir: Optional[Union[Path, str]]) -> Path:
        if skills_dir:
            user_path = Path(skills_dir)
            if user_path.exists():
                return user_path
            logger.warning("skills.loader: custom directory not found, fallback to builtin: %s", user_path)
        return self.default_skills_dir()

    def _load_directory_exact(self, skills_path: Path) -> List[WritingSkill]:
        """
        Load all *.yaml under *skills_path* with no fallback path.

        - **Args**:
            - `skills_path` (Path): Root directory to scan.

        - **Returns**:
            - `List[WritingSkill]`: Successfully loaded skills (may be empty).
        """
        skills: List[WritingSkill] = []
        if not skills_path.exists():
            logger.warning("skills.loader: directory not found: %s", skills_path)
            return skills

        for yaml_file in sorted(skills_path.rglob("*.yaml")):
            skill = self.load_single(yaml_file)
            if skill is not None:
                skills.append(skill)

        logger.info(
            "skills.loader: loaded %d skills from %s",
            len(skills),
            skills_path,
        )
        return skills

    def load_merged(
        self,
        user_skills_dir: Optional[Union[Path, str]] = None,
    ) -> List[WritingSkill]:
        """
        Load bundled skills, then optionally a user directory.
        Skills with the same `name` use the user-supplied definition last.

        - **Args**:
            - `user_skills_dir` (Path | str, optional): Extra directory to merge;
                if missing on disk, logs a warning and only bundled skills load.

        - **Returns**:
            - `List[WritingSkill]`: Deduplicated by `name`, user wins on collision.
        """
        by_name: Dict[str, WritingSkill] = {}
        builtin_path = self.default_skills_dir()
        for skill in self._load_directory_exact(builtin_path):
            by_name[skill.name] = skill

        if user_skills_dir:
            user_path = Path(user_skills_dir)
            if user_path.exists():
                overridden = 0
                user_skills = self._load_directory_exact(user_path)
                for skill in user_skills:
                    if skill.name in by_name:
                        overridden += 1
                    by_name[skill.name] = skill
                logger.info(
                    "skills.loader: merged %d skill definitions from %s "
                    "(%d names replaced bundled definitions)",
                    len(user_skills),
                    user_path,
                    overridden,
                )
            else:
                logger.warning(
                    "skills.loader: user skills_dir not found, bundled only: %s",
                    user_path,
                )

        merged = list(by_name.values())
        logger.info(
            "skills.loader: merged total %d unique skills (builtin=%s)",
            len(merged),
            builtin_path,
        )
        return merged

    def load_directory(self, skills_dir: Optional[Union[Path, str]] = None) -> List[WritingSkill]:
        """
        Recursively load all .yaml files under *skills_dir*.

        - **Args**:
            - `skills_dir` (Path): Root directory to scan

        - **Returns**:
            - `List[WritingSkill]`: All successfully loaded skills
        """
        skills_path = self.resolve_skills_dir(skills_dir)
        return self._load_directory_exact(skills_path)

    def load_single(self, path: Path) -> Optional[WritingSkill]:
        """
        Load a single YAML file into a WritingSkill.

        - **Args**:
            - `path` (Path): Path to the .yaml file

        - **Returns**:
            - `WritingSkill` or None if loading fails
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                logger.warning("skills.loader: invalid YAML (not a dict): %s", path)
                return None

            skill = WritingSkill(**data)
            logger.debug("skills.loader: loaded skill '%s' from %s", skill.name, path)
            return skill

        except Exception as e:
            logger.warning("skills.loader: failed to load %s: %s", path, e)
            return None
