"""Configuration loading utilities."""

import json
from pathlib import Path

import pydantic
from loguru import logger

from nanobot.config.schema import Config

_current_config_path: Path | None = None


def set_config_path(path: Path) -> None:
    """Set the current config path (used to derive data directory)."""
    global _current_config_path
    _current_config_path = path


def get_config_path() -> Path:
    """Get the configuration file path."""
    if _current_config_path:
        return _current_config_path
    return Path.home() / ".nanobot" / "config.json"


def load_config(config_path: Path | None = None, *, raise_errors: bool = False) -> Config:
    """Load configuration from file or create default.

    Args:
        config_path: Optional path to config file. Defaults to global config path.
        raise_errors: If True, raise on config errors instead of falling back to defaults.

    Returns:
        Validated Config instance.

    Raises:
        FileNotFoundError: If raise_errors=True and config file does not exist.
        json.JSONDecodeError: If raise_errors=True and config file is not valid JSON.
        pydantic.ValidationError: If raise_errors=True and config fails validation.
    """
    path = config_path or get_config_path()

    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return Config.model_validate(data)
        except json.JSONDecodeError as e:
            msg = f"Config file '{path}' contains invalid JSON: {e}"
            if raise_errors:
                raise
            logger.warning(msg)
            logger.warning("Using default configuration.")
        except pydantic.ValidationError as e:
            field_errors: dict[str, list[str]] = {}
            for err in e.errors():
                field = ".".join(str(loc) for loc in err["loc"])
                field_errors.setdefault(field, []).append(err["msg"])

            lines = [f"Config validation failed for '{path}':"]
            for field, errors in field_errors.items():
                for err_msg in errors:
                    lines.append(f"  - {field}: {err_msg}")

            msg = "\n".join(lines)
            if raise_errors:
                raise ValueError(msg) from e
            logger.warning(msg)
            logger.warning("Using default configuration.")

    return Config()


def save_config(config: Config, config_path: Path | None = None) -> None:
    """Save configuration to file."""
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = config.model_dump(mode="json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
