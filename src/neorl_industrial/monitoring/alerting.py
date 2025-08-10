"""Advanced alerting and monitoring system for industrial RL."""

import time
import threading
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import logging
import json

from ..core.types import Array
from .logger import get_logger


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of alerts."""
    SAFETY_VIOLATION = "safety_violation"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    TRAINING_FAILURE = "training_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATA_QUALITY = "data_quality"
    SYSTEM_ERROR = "system_error"
    SECURITY_BREACH = "security_breach"


@dataclass
class Alert:
    """Represents a system alert."""
    id: str
    type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None
    acknowledgment_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time,
            "acknowledgment_time": self.acknowledgment_time,
        }
    
    def acknowledge(self):
        """Acknowledge the alert."""
        self.acknowledgment_time = time.time()
    
    def resolve(self):
        """Mark the alert as resolved."""
        self.resolved = True
        self.resolution_time = time.time()


@dataclass
class AlertRule:
    """Defines conditions for triggering alerts."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    alert_type: AlertType
    severity: AlertSeverity
    message_template: str
    cooldown_seconds: float = 300.0  # 5 minutes default cooldown
    max_alerts_per_hour: int = 10
    metadata_extractor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    
    def __post_init__(self):
        self._last_triggered = 0.0
        self._alert_count = 0
        self._alert_times = deque(maxlen=self.max_alerts_per_hour)
    
    def should_trigger(self, data: Dict[str, Any]) -> bool:
        """Check if alert should be triggered."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self._last_triggered < self.cooldown_seconds:
            return False
        
        # Check rate limiting
        hour_ago = current_time - 3600
        recent_alerts = [t for t in self._alert_times if t > hour_ago]
        if len(recent_alerts) >= self.max_alerts_per_hour:
            return False
        
        # Check condition
        if not self.condition(data):
            return False
        
        # Update tracking
        self._last_triggered = current_time
        self._alert_times.append(current_time)
        
        return True
    
    def create_alert(self, data: Dict[str, Any]) -> Alert:
        """Create alert from current data."""
        alert_id = f"{self.name}_{int(time.time() * 1000)}"
        
        # Extract metadata
        metadata = {}
        if self.metadata_extractor:
            try:
                metadata = self.metadata_extractor(data)
            except Exception as e:
                metadata["extraction_error"] = str(e)
        
        # Add relevant data
        metadata.update({
            "rule_name": self.name,
            "trigger_data": data,
        })
        
        return Alert(
            id=alert_id,
            type=self.alert_type,
            severity=self.severity,
            message=self.message_template.format(**data),
            timestamp=time.time(),
            metadata=metadata,
        )


class AlertManager:
    """Manages alerts and notifications for the RL system."""
    
    def __init__(self, max_alerts_stored: int = 10000):
        self.rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.max_alerts_stored = max_alerts_stored
        
        # Notification handlers
        self.notification_handlers: List[Callable[[Alert], None]] = []
        
        # Threading
        self._lock = threading.Lock()
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Statistics
        self.stats = {
            "total_alerts": 0,
            "alerts_by_type": defaultdict(int),
            "alerts_by_severity": defaultdict(int),
            "avg_resolution_time": 0.0,
        }
        
        self.logger = get_logger("alert_manager")
        
        # Add default rules
        self._add_default_rules()
        
        self.logger.info("Alert manager initialized")
    
    def _add_default_rules(self):
        """Add default alerting rules."""
        
        # Safety violation rule
        self.add_rule(AlertRule(
            name="safety_violation_critical",
            condition=lambda data: data.get("safety_violations", 0) > 0,
            alert_type=AlertType.SAFETY_VIOLATION,
            severity=AlertSeverity.CRITICAL,
            message_template="Critical safety violation detected: {safety_violations} violations",
            cooldown_seconds=60.0,  # Shorter cooldown for safety
            metadata_extractor=lambda data: {
                "violation_count": data.get("safety_violations", 0),
                "environment": data.get("environment", "unknown"),
            },
        ))
        
        # Training failure rule
        self.add_rule(AlertRule(
            name="training_failure",
            condition=lambda data: data.get("training_failed", False),
            alert_type=AlertType.TRAINING_FAILURE,
            severity=AlertSeverity.WARNING,
            message_template="Training failure detected: {failure_reason}",
            metadata_extractor=lambda data: {
                "agent_type": data.get("agent_type", "unknown"),
                "epoch": data.get("epoch", -1),
                "error": data.get("error", "unknown"),
            },
        ))
        
        # Performance degradation rule
        self.add_rule(AlertRule(
            name="performance_degradation",
            condition=lambda data: (
                data.get("current_performance", 0) < 
                data.get("baseline_performance", 0) * 0.8  # 20% degradation
            ),
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.WARNING,
            message_template="Performance degradation detected: {current_performance:.2f} vs baseline {baseline_performance:.2f}",
            cooldown_seconds=600.0,  # 10 minutes
            metadata_extractor=lambda data: {
                "degradation_percent": (
                    (data.get("baseline_performance", 1) - data.get("current_performance", 0)) / 
                    data.get("baseline_performance", 1) * 100
                ),
            },
        ))
        
        # Resource exhaustion rule
        self.add_rule(AlertRule(
            name="memory_exhaustion",
            condition=lambda data: data.get("memory_usage_percent", 0) > 90,
            alert_type=AlertType.RESOURCE_EXHAUSTION,
            severity=AlertSeverity.CRITICAL,
            message_template="High memory usage detected: {memory_usage_percent:.1f}%",
            cooldown_seconds=300.0,
        ))
        
        # Data quality rule
        self.add_rule(AlertRule(
            name="data_quality_issue",
            condition=lambda data: (
                data.get("nan_count", 0) > 0 or 
                data.get("inf_count", 0) > 0 or
                data.get("data_variance", 1) < 1e-6
            ),
            alert_type=AlertType.DATA_QUALITY,
            severity=AlertSeverity.WARNING,
            message_template="Data quality issue: NaN={nan_count}, Inf={inf_count}, Low variance={data_variance:.2e}",
            metadata_extractor=lambda data: {
                "dataset_size": data.get("dataset_size", 0),
                "affected_features": data.get("affected_features", []),
            },
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add an alerting rule."""
        with self._lock:
            self.rules.append(rule)
            self.logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove an alerting rule by name."""
        with self._lock:
            self.rules = [rule for rule in self.rules if rule.name != rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler."""
        self.notification_handlers.append(handler)
        self.logger.info("Added notification handler")
    
    def check_conditions(self, data: Dict[str, Any]):
        """Check all alert conditions against provided data."""
        triggered_alerts = []
        
        with self._lock:
            for rule in self.rules:
                try:
                    if rule.should_trigger(data):
                        alert = rule.create_alert(data)
                        self._process_alert(alert)
                        triggered_alerts.append(alert)
                        
                except Exception as e:
                    self.logger.error(f"Error checking rule {rule.name}: {e}")
        
        return triggered_alerts
    
    def _process_alert(self, alert: Alert):
        """Process a new alert."""
        # Store alert
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Limit history size
        if len(self.alert_history) > self.max_alerts_stored:
            self.alert_history = self.alert_history[-self.max_alerts_stored:]
        
        # Update statistics
        self.stats["total_alerts"] += 1
        self.stats["alerts_by_type"][alert.type.value] += 1
        self.stats["alerts_by_severity"][alert.severity.value] += 1
        
        # Send notifications
        self._notify(alert)
        
        self.logger.info(f"Alert triggered: {alert.id} - {alert.message}")
    
    def _notify(self, alert: Alert):
        """Send alert notifications."""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Notification handler failed: {e}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledge()
                self.logger.info(f"Alert acknowledged: {alert_id}")
                return True
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolve()
                
                # Update resolution time statistics
                resolution_time = alert.resolution_time - alert.timestamp
                current_avg = self.stats["avg_resolution_time"]
                total_resolved = sum(1 for a in self.alert_history if a.resolved)
                
                if total_resolved > 0:
                    self.stats["avg_resolution_time"] = (
                        (current_avg * (total_resolved - 1) + resolution_time) / total_resolved
                    )
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                self.logger.info(f"Alert resolved: {alert_id} (resolution time: {resolution_time:.2f}s)")
                return True
            return False
    
    def get_active_alerts(
        self, 
        alert_type: Optional[AlertType] = None,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get active alerts, optionally filtered."""
        with self._lock:
            alerts = list(self.active_alerts.values())
            
            if alert_type:
                alerts = [a for a in alerts if a.type == alert_type]
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_history(
        self,
        hours: int = 24,
        alert_type: Optional[AlertType] = None,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get alert history for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            alerts = [
                alert for alert in self.alert_history 
                if alert.timestamp > cutoff_time
            ]
            
            if alert_type:
                alerts = [a for a in alerts if a.type == alert_type]
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        with self._lock:
            stats = self.stats.copy()
            stats["active_alerts"] = len(self.active_alerts)
            stats["total_rules"] = len(self.rules)
            
            # Calculate alert rates
            current_time = time.time()
            hour_ago = current_time - 3600
            day_ago = current_time - 86400
            
            recent_alerts_1h = [
                a for a in self.alert_history 
                if a.timestamp > hour_ago
            ]
            recent_alerts_24h = [
                a for a in self.alert_history 
                if a.timestamp > day_ago
            ]
            
            stats["alerts_last_hour"] = len(recent_alerts_1h)
            stats["alerts_last_24h"] = len(recent_alerts_24h)
            
            return stats
    
    def export_alerts(self, filename: str):
        """Export alert history to JSON file."""
        with self._lock:
            alerts_data = {
                "export_timestamp": time.time(),
                "statistics": self.get_statistics(),
                "active_alerts": [alert.to_dict() for alert in self.active_alerts.values()],
                "alert_history": [alert.to_dict() for alert in self.alert_history],
            }
            
            with open(filename, 'w') as f:
                json.dump(alerts_data, f, indent=2)
            
            self.logger.info(f"Alerts exported to {filename}")
    
    def start_monitoring(self, data_source: Callable[[], Dict[str, Any]], interval: float = 60.0):
        """Start continuous monitoring with a data source.
        
        Args:
            data_source: Function that returns current system data
            interval: Monitoring interval in seconds
        """
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            self.logger.warning("Monitoring already running")
            return
        
        self._stop_monitoring.clear()
        
        def monitor_loop():
            while not self._stop_monitoring.wait(interval):
                try:
                    data = data_source()
                    if data:
                        self.check_conditions(data)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
        
        self._monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitoring_thread.start()
        
        self.logger.info(f"Started monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if self._monitoring_thread is None:
            return
        
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=5.0)
        
        if self._monitoring_thread.is_alive():
            self.logger.warning("Monitoring thread did not stop gracefully")
        
        self.logger.info("Monitoring stopped")


# Default notification handlers
def console_notification_handler(alert: Alert):
    """Simple console notification handler."""
    severity_colors = {
        AlertSeverity.INFO: "\033[94m",     # Blue
        AlertSeverity.WARNING: "\033[93m",  # Yellow
        AlertSeverity.CRITICAL: "\033[91m", # Red
        AlertSeverity.EMERGENCY: "\033[95m", # Magenta
    }
    
    reset_color = "\033[0m"
    color = severity_colors.get(alert.severity, "")
    
    print(f"{color}[{alert.severity.value.upper()}] {alert.message}{reset_color}")
    print(f"  Alert ID: {alert.id}")
    print(f"  Type: {alert.type.value}")
    print(f"  Time: {time.ctime(alert.timestamp)}")
    if alert.metadata:
        print(f"  Metadata: {alert.metadata}")
    print()


def file_notification_handler(log_file: str):
    """Create a file-based notification handler.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Notification handler function
    """
    def handler(alert: Alert):
        with open(log_file, 'a') as f:
            f.write(f"{time.ctime(alert.timestamp)} | {alert.severity.value.upper()} | {alert.message}\n")
            f.write(f"  ID: {alert.id} | Type: {alert.type.value}\n")
            if alert.metadata:
                f.write(f"  Metadata: {json.dumps(alert.metadata)}\n")
            f.write("\n")
    
    return handler


# Global alert manager instance
_alert_manager = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
        # Add default console handler
        _alert_manager.add_notification_handler(console_notification_handler)
    return _alert_manager
