"""YAML config loading for the metric study.

Follows the training repo's ``YParams`` convention (a named section of a YAML file,
loaded with ``ruamel.yaml``, exposed as both attribute and dict access) but adds
validation and a stable config hash so every pipeline output can be traced back to
the exact settings that produced it.

Typical usage::

    cfg = StudyConfig.load("configs/metric_study.yaml", "default")
    cfg.seed
    cfg["downsample"]["tau"]
    cfg.config_hash
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

logger = logging.getLogger(__name__)

# Keys that must be present in every config section, with their expected type.
_REQUIRED_KEYS: dict[str, type] = {
    "seed": int,
    "resolutions": list,
    "downsample": dict,
    "preprocess": dict,
    "masks": dict,
    "degradations": dict,
    "metrics": dict,
    "chunk_size": int,
    "data_dir": str,
    "output_dir": str,
}


class ConfigError(ValueError):
    """Raised when a config section is missing required keys or has the wrong type."""


class StudyConfig:
    """Flat, validated view of one named section of a metric-study YAML file.

    Args:
        yaml_path: Path to the YAML config file.
        section: Name of the top-level section to load (e.g. ``"default"``,
            ``"dry_run"``, ``"production"``).
        params: The parsed section as a plain dict. Populated by :meth:`load`.
    """

    def __init__(self, yaml_path: Path, section: str, params: dict[str, Any]):
        self.yaml_path = Path(yaml_path)
        self.section = section
        self.params = params

    @classmethod
    def load(cls, yaml_path: str | Path, section: str, validate: bool = True) -> "StudyConfig":
        """Load and validate one section of a YAML config file.

        Args:
            yaml_path: Path to the YAML file.
            section: Name of the section (top-level key) to load.
            validate: If True (default), check that all required keys are present
                and of the expected type.

        Returns:
            A populated, validated ``StudyConfig``.

        Raises:
            ConfigError: If the section is missing, or a required key is missing
                or has the wrong type.
        """
        yaml_path = Path(yaml_path)
        yaml = YAML(typ="safe")
        with open(yaml_path) as f:
            data = yaml.load(f)
        if section not in data:
            raise ConfigError(f"Section '{section}' not found in {yaml_path}. Available: {list(data)}")
        params = dict(data[section])
        cfg = cls(yaml_path, section, params)
        if validate:
            cfg.validate()
        return cfg

    def validate(self) -> None:
        """Check required keys and types. Raises ConfigError on the first problem found."""
        for key, expected_type in _REQUIRED_KEYS.items():
            if key not in self.params:
                raise ConfigError(f"Config section '{self.section}' is missing required key '{key}'")
            if not isinstance(self.params[key], expected_type):
                raise ConfigError(
                    f"Config key '{key}' should be {expected_type.__name__}, "
                    f"got {type(self.params[key]).__name__}"
                )
        valid_resolutions = {"native", "512"}
        bad = set(self.params["resolutions"]) - valid_resolutions
        if bad:
            raise ConfigError(f"Unknown resolutions {bad}; expected subset of {valid_resolutions}")

    def __getattr__(self, key: str) -> Any:
        try:
            return self.params[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __getitem__(self, key: str) -> Any:
        return self.params[key]

    def __contains__(self, key: str) -> bool:
        return key in self.params

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    @property
    def config_hash(self) -> str:
        """A short, stable hash of the fully-resolved config, for provenance columns.

        Uses a canonical (sorted-key) JSON dump so the hash is independent of key
        order in the YAML file, and stable across processes and machines.
        """
        canonical = json.dumps(self.params, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def log(self) -> None:
        """Log the resolved config at INFO level, matching the repo's ``YParams.log()``."""
        logger.info("------------------ Config: %s (section=%s) ------------------", self.yaml_path, self.section)
        for key, val in self.params.items():
            logger.info("%s: %s", key, val)
        logger.info("config_hash: %s", self.config_hash)
        logger.info("---------------------------------------------------")
