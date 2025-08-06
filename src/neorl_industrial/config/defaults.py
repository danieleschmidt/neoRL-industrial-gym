"""Default configuration values."""

DEFAULT_CONFIG = {
    "logging": {
        "level": "INFO",
        "directory": "./logs",
        "max_file_size": 100 * 1024 * 1024,  # 100MB
        "backup_count": 5,
        "enable_safety_logs": True,
        "enable_console": True,
        "enable_file": True,
    },
    "environments": {
        "default_safety_penalty": -100.0,
        "max_episode_steps": 1000,
        "dt": 0.1,
        "enable_safety_constraints": True,
        "constraint_violation_limit": 10,
    },
    "agents": {
        "cql": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "alpha": 0.2,
            "cql_alpha": 1.0,
            "safety_penalty": 100.0,
            "hidden_dims": [256, 256],
            "batch_size": 256,
            "use_layer_norm": False,
            "dropout_rate": 0.0,
        },
        "training": {
            "n_epochs": 100,
            "eval_freq": 10,
            "save_freq": 50,
            "early_stopping_patience": 20,
            "min_improvement": 1e-4,
        }
    },
    "datasets": {
        "data_directory": "./data",
        "cache_datasets": True,
        "normalize_observations": True,
        "normalize_rewards": True,
        "validation_split": 0.1,
    },
    "monitoring": {
        "enable_metrics_collection": True,
        "metrics_port": 8080,
        "health_check_interval": 30,  # seconds
        "safety_alert_threshold": 0.8,
        "performance_alert_threshold": 0.1,  # 10% degradation
    },
    "safety": {
        "max_constraint_violations": 5,
        "emergency_shutdown_threshold": 3,  # consecutive critical violations
        "constraint_buffer_size": 100,
        "enable_conservative_mode": True,
    },
    "experiment": {
        "tracking_backend": "mlflow",  # or "tensorboard", "wandb"
        "experiment_name": "neorl_industrial",
        "auto_log_metrics": True,
        "log_frequency": 10,  # steps
        "save_artifacts": True,
    },
}


def get_default_config():
    """Get a copy of the default configuration."""
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)