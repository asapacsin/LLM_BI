"""Load YAML prompt templates."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.config import PROJECT_ROOT

PROMPTS_DIR = PROJECT_ROOT / "prompts"


def load_prompt_for_intent(intent: str) -> str:
    """Load intent-specific instruction; fallback to performance_monitoring."""
    path = PROMPTS_DIR / f"{intent}.yaml"
    if not path.exists():
        path = PROMPTS_DIR / "performance_monitoring.yaml"
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    system_path = PROMPTS_DIR / "system.yaml"
    with open(system_path, encoding="utf-8") as f:
        system_data = yaml.safe_load(f)
    return f"{system_data.get('system', '')}\n\n{data.get('instruction', '')}"
