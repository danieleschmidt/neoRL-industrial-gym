"""Advanced logging system for industrial RL applications."""

import json
import logging
import logging.handlers
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


class IndustrialLogger:
    """Specialized logger for industrial RL with safety event tracking."""
    
    def __init__(
        self,
        name: str,
        log_dir: Optional[Union[str, Path]] = None,
        level: str = "INFO",
        enable_safety_logs: bool = True,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 5,
    ):
        """Initialize industrial logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
            enable_safety_logs: Enable safety event logging
            max_file_size: Maximum log file size before rotation
            backup_count: Number of backup log files to keep
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup log directory
        if log_dir is None:
            log_dir = Path("./logs")
        else:
            log_dir = Path(log_dir)
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / f"{name}.log",
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Safety event logger if enabled
        self.safety_logger = None
        if enable_safety_logs:
            self.safety_logger = logging.getLogger(f"{name}.safety")
            self.safety_logger.setLevel(logging.WARNING)
            
            safety_handler = logging.handlers.RotatingFileHandler(
                filename=log_dir / f"{name}_safety.log",
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            
            safety_formatter = SafetyFormatter()
            safety_handler.setFormatter(safety_formatter)
            self.safety_logger.addHandler(safety_handler)
        
        # Metrics for logging performance
        self.start_time = time.time()
        self.log_counts = {"DEBUG": 0, "INFO": 0, "WARNING": 0, "ERROR": 0, "CRITICAL": 0}
    
    def debug(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, msg, extra)
    
    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message.""" 
        self._log(logging.INFO, msg, extra)
    
    def warning(self, msg: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message."""
        self._log(logging.WARNING, msg, extra)
    
    def error(self, msg: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False) -> None:
        """Log error message."""
        self._log(logging.ERROR, msg, extra, exc_info=exc_info)
    
    def critical(self, msg: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, msg, extra, exc_info=exc_info)
    
    def safety_event(
        self,
        event_type: str,
        severity: str,
        details: Dict[str, Any],
        agent_id: Optional[str] = None,
        env_id: Optional[str] = None,
    ) -> None:
        """Log safety-critical event.
        
        Args:
            event_type: Type of safety event
            severity: Event severity (LOW, MEDIUM, HIGH, CRITICAL)
            details: Event details
            agent_id: Agent identifier
            env_id: Environment identifier
        """
        if self.safety_logger is None:
            return
        
        safety_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "severity": severity.upper(),
            "agent_id": agent_id,
            "env_id": env_id,
            "details": self._serialize_for_json(details),
        }
        
        level = {
            "LOW": logging.WARNING,
            "MEDIUM": logging.WARNING,
            "HIGH": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }.get(severity.upper(), logging.WARNING)
        
        self.safety_logger.log(level, json.dumps(safety_record))
    
    def log_training_progress(
        self,
        epoch: int,
        metrics: Dict[str, float],
        agent_id: Optional[str] = None,
    ) -> None:
        """Log training progress with metrics."""
        msg = f"Training Progress [Epoch {epoch}]"
        if agent_id:
            msg += f" [Agent: {agent_id}]"
        
        # Format metrics for display
        metric_strs = []
        for key, value in metrics.items():
            if isinstance(value, float):
                metric_strs.append(f"{key}={value:.4f}")
            else:
                metric_strs.append(f"{key}={value}")
        
        msg += f" | {' | '.join(metric_strs)}"
        
        self.info(msg, extra={"epoch": epoch, "metrics": metrics, "agent_id": agent_id})
    
    def log_evaluation_results(
        self,
        results: Dict[str, Any],
        agent_id: Optional[str] = None,
        env_id: Optional[str] = None,
    ) -> None:
        """Log evaluation results."""
        msg = "Evaluation Results"
        if agent_id:
            msg += f" [Agent: {agent_id}]"
        if env_id:
            msg += f" [Env: {env_id}]"
        
        # Extract key metrics
        return_mean = results.get("return_mean", 0)
        safety_violations = results.get("safety_violations", 0)
        success_rate = results.get("success_rate", 0)
        
        msg += f" | Return: {return_mean:.2f} | Safety Violations: {safety_violations} | Success Rate: {success_rate:.1%}"
        
        self.info(msg, extra={"results": results, "agent_id": agent_id, "env_id": env_id})
        
        # Log safety events if violations occurred
        if safety_violations > 0:
            self.safety_event(
                event_type="EVALUATION_SAFETY_VIOLATION",
                severity="MEDIUM",
                details={
                    "violation_count": safety_violations,
                    "success_rate": success_rate,
                    "return_mean": return_mean,
                },
                agent_id=agent_id,
                env_id=env_id,
            )
    
    def _log(self, level: int, msg: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False) -> None:
        """Internal logging method."""
        level_name = logging.getLevelName(level)
        self.log_counts[level_name] += 1
        
        # Serialize extra data for JSON compatibility
        if extra:
            extra = self._serialize_for_json(extra)
        
        self.logger.log(level, msg, extra=extra, exc_info=exc_info)
    
    def _serialize_for_json(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return self._serialize_for_json(obj.__dict__)
        else:
            return str(obj)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        runtime = time.time() - self.start_time
        return {
            "runtime_seconds": runtime,
            "log_counts": self.log_counts.copy(),
            "logs_per_second": sum(self.log_counts.values()) / max(runtime, 1),
        }


class ColoredFormatter(logging.Formatter):
    """Colored console formatter."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green  
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class SafetyFormatter(logging.Formatter):
    """Special formatter for safety events."""
    
    def format(self, record):
        try:
            # Parse JSON log message
            safety_data = json.loads(record.getMessage())
            
            # Format as structured safety log
            formatted = (
                f"{safety_data['timestamp']} | "
                f"SAFETY | {safety_data['severity']} | "
                f"{safety_data['event_type']} | "
                f"Agent: {safety_data.get('agent_id', 'N/A')} | "
                f"Env: {safety_data.get('env_id', 'N/A')} | "
                f"Details: {json.dumps(safety_data['details'])}"
            )
            
            record.msg = formatted
            return super().format(record)
        except (json.JSONDecodeError, KeyError):
            # Fallback to standard formatting
            return super().format(record)


# Global logger registry
_loggers: Dict[str, IndustrialLogger] = {}


def setup_logging(
    name: str = "neorl_industrial",
    log_dir: Optional[Union[str, Path]] = None,
    level: str = "INFO",
    **kwargs
) -> IndustrialLogger:
    """Setup logging for the application.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        **kwargs: Additional arguments for IndustrialLogger
        
    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = IndustrialLogger(
        name=name,
        log_dir=log_dir,
        level=level,
        **kwargs
    )
    
    _loggers[name] = logger
    
    # Log system startup
    logger.info(f"Logging system initialized for {name}")
    logger.info(f"Log directory: {log_dir or './logs'}")
    logger.info(f"Log level: {level}")
    
    return logger


def get_logger(name: str = "neorl_industrial") -> IndustrialLogger:
    """Get existing logger or create default one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name not in _loggers:
        return setup_logging(name)
    
    return _loggers[name]