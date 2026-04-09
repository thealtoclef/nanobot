"""Shared identity string builder for nanobot agent."""

import platform
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
)


def build_identity(workspace: Path) -> str:
    """Build the core identity section for nanobot."""
    system = platform.system()
    runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {sys.version.split()[0]}"

    template = _env.get_template("identity.md.jinja")
    return template.render(
        runtime=runtime,
        workspace_path=str(workspace.expanduser().resolve()),
        is_windows=(system == "Windows"),
    )
