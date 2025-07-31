#!/usr/bin/env python3
"""
Configuration loader for neoRL-industrial-gym.

This module provides a centralized way to load and manage configuration
across different environments (development, testing, production).
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when there's an error loading or validating configuration."""
    pass


class ConfigLoader:
    """Configuration loader with environment-specific overrides."""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files.
                       Defaults to config/ relative to this file.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent
        self.config_dir = Path(config_dir)
        self.environment = self._detect_environment()
        self._config_cache: Optional[Dict[str, Any]] = None
    
    def _detect_environment(self) -> str:
        """Detect the current environment from environment variables."""
        env = os.getenv('NEORL_ENV', '').lower()
        
        # Environment detection logic
        if env in ['production', 'prod']:
            return 'production'
        elif env in ['testing', 'test']:
            return 'testing'
        elif env in ['development', 'dev', '']:
            return 'development'
        else:
            logger.warning(f"Unknown environment '{env}', defaulting to development")
            return 'development'
    
    def load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load configuration for the current environment.
        
        Args:
            force_reload: If True, reload configuration from files.
                         Otherwise, use cached configuration if available.
        
        Returns:
            Dictionary containing the merged configuration.
            
        Raises:
            ConfigurationError: If configuration cannot be loaded or is invalid.
        """
        if self._config_cache is not None and not force_reload:
            return self._config_cache
        
        try:
            # Load base configuration
            base_config = self._load_base_config()
            
            # Load environment-specific configuration
            env_config = self._load_environment_config(self.environment)
            
            # Merge configurations (environment overrides base)
            merged_config = self._merge_configs(base_config, env_config)
            
            # Apply environment variable overrides
            final_config = self._apply_env_overrides(merged_config)
            
            # Validate configuration
            self._validate_config(final_config)
            
            # Cache the configuration
            self._config_cache = final_config
            
            logger.info(f"Configuration loaded for environment: {self.environment}")
            return final_config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}") from e
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration that applies to all environments."""
        base_config_file = self.config_dir / "base.yaml"
        
        if base_config_file.exists():
            return self._load_yaml_file(base_config_file)
        else:
            # Return minimal base configuration if file doesn't exist
            return {
                "environment": {"name": "unknown"},
                "logging": {"level": "INFO"},
                "safety": {"monitoring_enabled": True}
            }
    
    def _load_environment_config(self, environment: str) -> Dict[str, Any]:
        """Load environment-specific configuration."""
        env_config_file = self.config_dir / "environments" / f"{environment}.yaml"
        
        if not env_config_file.exists():
            logger.warning(f"Environment config file not found: {env_config_file}")
            return {}
        
        return self._load_yaml_file(env_config_file)
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Expand environment variables in the YAML content
            expanded_content = os.path.expandvars(content)
            
            config = yaml.safe_load(expanded_content)
            return config if config is not None else {}
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {file_path}: {str(e)}")
        except IOError as e:
            raise ConfigurationError(f"Cannot read {file_path}: {str(e)}")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        # Common environment variable mappings
        env_mappings = {
            'DEBUG': 'environment.debug',
            'LOG_LEVEL': 'logging.level',
            'MLFLOW_TRACKING_URI': 'mlflow.tracking_uri',
            'MLFLOW_EXPERIMENT_NAME': 'mlflow.experiment_name',
            'JAX_PLATFORM_NAME': 'jax.platform',
            'BATCH_SIZE': 'training.batch_size',
            'SAFETY_CONSTRAINT_THRESHOLD': 'safety.constraint_threshold',
            'DATA_DIR': 'paths.data',
            'MAX_MEMORY_USAGE_GB': 'resources.max_memory_gb',
        }
        
        result = config.copy()
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                self._set_nested_value(result, config_path, self._convert_env_value(env_value))
        
        return result
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested configuration value using dot notation."""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert string environment variable to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False
        
        # Integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate the configuration for required fields and valid values."""
        # Required top-level sections
        required_sections = ['environment', 'safety']
        
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Missing required configuration section: {section}")
        
        # Validate environment name
        valid_environments = ['development', 'testing', 'production']
        env_name = config.get('environment', {}).get('name')
        if env_name not in valid_environments:
            logger.warning(f"Unknown environment name: {env_name}")
        
        # Validate safety configuration
        safety_config = config.get('safety', {})
        if not isinstance(safety_config.get('monitoring_enabled'), bool):
            raise ConfigurationError("safety.monitoring_enabled must be a boolean")
        
        # Validate threshold values
        threshold = safety_config.get('constraint_threshold')
        if threshold is not None and (not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1):
            raise ConfigurationError("safety.constraint_threshold must be a number between 0 and 1")
        
        # Validate paths exist (for production)
        if env_name == 'production':
            paths_config = config.get('paths', {})
            for path_name, path_value in paths_config.items():
                if path_value and not Path(path_value).parent.exists():
                    logger.warning(f"Path {path_name} parent directory does not exist: {path_value}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'mlflow.tracking_uri')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        config = self.load_config()
        
        keys = key.split('.')
        current = config
        
        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default
    
    def get_environment(self) -> str:
        """Get the current environment name."""
        return self.environment
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == 'development'
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == 'testing'
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == 'production'


# Global configuration instance
_config_loader = None


def get_config_loader() -> ConfigLoader:
    """Get the global configuration loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def get_config() -> Dict[str, Any]:
    """Get the current configuration."""
    return get_config_loader().load_config()


def get_setting(key: str, default: Any = None) -> Any:
    """Get a configuration setting using dot notation."""
    return get_config_loader().get(key, default)


# Convenience functions
def is_development() -> bool:
    """Check if running in development environment."""
    return get_config_loader().is_development()


def is_testing() -> bool:
    """Check if running in testing environment."""
    return get_config_loader().is_testing()


def is_production() -> bool:
    """Check if running in production environment."""
    return get_config_loader().is_production()


if __name__ == "__main__":
    # Simple CLI for testing configuration
    import sys
    
    loader = ConfigLoader()
    
    if len(sys.argv) > 1:
        # Print specific configuration value
        key = sys.argv[1]
        value = loader.get(key)
        print(f"{key}: {value}")
    else:
        # Print current environment and basic info
        config = loader.load_config()
        print(f"Environment: {loader.get_environment()}")
        print(f"Debug mode: {config.get('environment', {}).get('debug', False)}")
        print(f"Log level: {config.get('logging', {}).get('level', 'INFO')}")
        print(f"Safety monitoring: {config.get('safety', {}).get('monitoring_enabled', True)}")