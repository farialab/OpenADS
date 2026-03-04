"""Configuration loading utilities.

Provides robust YAML config loading with variable substitution.
"""

from __future__ import annotations
import logging
import re
from pathlib import Path
from typing import Dict, Any

import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and processes YAML configuration files.

    Supports variable substitution like ${project_dir} within config values.
    """

    @staticmethod
    def load(config_path: Path) -> Dict[str, Any]:
        """Load YAML config file with variable expansion.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Loaded and processed configuration dictionary

        Example:
            Config file content:
                project_dir: /path/to/project
                templates:
                    root: ${project_dir}/assets/templates

            After loading, templates.root will be: /path/to/project/assets/templates
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Expand variables recursively
        config = ConfigLoader._expand_variables(config, config)

        return config

    @staticmethod
    def _expand_variables(obj: Any, context: Dict[str, Any]) -> Any:
        """Recursively expand ${var} references in config.

        Args:
            obj: Object to process (dict, list, str, or other)
            context: Root config dict for variable lookup

        Returns:
            Processed object with variables expanded
        """
        if isinstance(obj, dict):
            return {k: ConfigLoader._expand_variables(v, context) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ConfigLoader._expand_variables(item, context) for item in obj]
        elif isinstance(obj, str):
            return ConfigLoader._expand_string(obj, context)
        else:
            return obj

    @staticmethod
    def _expand_string(s: str, context: Dict[str, Any]) -> str:
        """Expand ${var} references in a string.

        Args:
            s: String to process
            context: Root config dict for variable lookup

        Returns:
            String with variables expanded
        """
        pattern = re.compile(r'\$\{(\w+)\}')

        def replacer(match):
            var_name = match.group(1)
            if var_name in context:
                return str(context[var_name])
            else:
                logger.warning(f"Variable ${{{var_name}}} not found in config, leaving as-is")
                return match.group(0)

        return pattern.sub(replacer, s)
