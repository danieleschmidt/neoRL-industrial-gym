"""Real-time monitoring dashboard for industrial RL systems."""

import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json

@dataclass
class SystemMetric:
    """System performance metric."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str]
    alert_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None

@dataclass
class AlertEvent:
    """Alert event for monitoring."""
    level: str  # INFO, WARNING, CRITICAL
    message: str
    timestamp: float
    source: str
    details: Dict[str, Any]

class MonitoringDashboard:
    """Real-time monitoring dashboard with alerts and visualization."""
    
    def __init__(self, max_history: int = 10000):
        """Initialize monitoring dashboard.
        
        Args:
            max_history: Maximum number of historical metrics to store
        """
        self.max_history = max_history
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_history))
        self.alerts = deque(maxlen=1000)
        self.active_alerts = {}
        self.is_running = False
        self._lock = threading.Lock()
        
        # System health tracking
        self.system_health = {
            "overall_status": "HEALTHY",
            "last_update": time.time(),
            "component_status": {},
        }
        
        # Performance baselines
        self.baselines = {
            "training_throughput": 1000.0,  # samples/sec
            "inference_latency": 0.1,      # seconds
            "memory_usage": 0.8,           # 80% threshold
            "safety_violation_rate": 0.01,  # 1% threshold
        }
    
    def record_metric(self, metric: SystemMetric) -> None:
        """Record a system metric."""
        with self._lock:
            self.metrics_history[metric.name].append(metric)
            
            # Check for alerts
            self._check_metric_alerts(metric)
            
            # Update system health
            self._update_system_health(metric)
    
    def record_multiple_metrics(self, metrics: List[SystemMetric]) -> None:
        """Record multiple metrics efficiently."""
        with self._lock:
            for metric in metrics:
                self.metrics_history[metric.name].append(metric)
                self._check_metric_alerts(metric)
                self._update_system_health(metric)
    
    def _check_metric_alerts(self, metric: SystemMetric) -> None:
        """Check if metric triggers any alerts."""
        if metric.critical_threshold is not None and metric.value >= metric.critical_threshold:
            self._trigger_alert(
                level="CRITICAL",
                message=f"{metric.name} exceeded critical threshold: {metric.value:.3f} >= {metric.critical_threshold}",
                source=metric.name,
                details=asdict(metric)
            )
        elif metric.alert_threshold is not None and metric.value >= metric.alert_threshold:
            self._trigger_alert(
                level="WARNING",
                message=f"{metric.name} exceeded warning threshold: {metric.value:.3f} >= {metric.alert_threshold}",
                source=metric.name,
                details=asdict(metric)
            )
    
    def _trigger_alert(self, level: str, message: str, source: str, details: Dict[str, Any]) -> None:
        """Trigger an alert event."""
        alert = AlertEvent(
            level=level,
            message=message,
            timestamp=time.time(),
            source=source,
            details=details
        )
        
        self.alerts.append(alert)
        
        # Track active alerts (deduplicate by source and level)
        alert_key = f"{source}_{level}"
        self.active_alerts[alert_key] = alert
        
        # Auto-resolve alerts after 5 minutes if no new triggers
        if level != "CRITICAL":
            threading.Timer(300, lambda: self.active_alerts.pop(alert_key, None)).start()
    
    def _update_system_health(self, metric: SystemMetric) -> None:
        """Update overall system health status."""
        component = metric.tags.get("component", "unknown")
        
        # Determine component health based on metric
        if metric.critical_threshold and metric.value >= metric.critical_threshold:
            status = "CRITICAL"
        elif metric.alert_threshold and metric.value >= metric.alert_threshold:
            status = "WARNING"
        else:
            status = "HEALTHY"
        
        self.system_health["component_status"][component] = {
            "status": status,
            "last_metric": metric.name,
            "last_value": metric.value,
            "timestamp": metric.timestamp,
        }
        
        # Update overall status
        component_statuses = [comp["status"] for comp in self.system_health["component_status"].values()]
        if any(status == "CRITICAL" for status in component_statuses):
            self.system_health["overall_status"] = "CRITICAL"
        elif any(status == "WARNING" for status in component_statuses):
            self.system_health["overall_status"] = "WARNING"
        else:
            self.system_health["overall_status"] = "HEALTHY"
        
        self.system_health["last_update"] = time.time()
    
    def get_current_metrics(self, metric_names: Optional[List[str]] = None) -> Dict[str, SystemMetric]:
        """Get current values for specified metrics."""
        with self._lock:
            result = {}
            
            names_to_check = metric_names or list(self.metrics_history.keys())
            
            for name in names_to_check:
                if name in self.metrics_history and self.metrics_history[name]:
                    result[name] = self.metrics_history[name][-1]
            
            return result
    
    def get_metric_history(self, metric_name: str, duration_seconds: Optional[float] = None) -> List[SystemMetric]:
        """Get historical data for a metric."""
        with self._lock:
            if metric_name not in self.metrics_history:
                return []
            
            metrics = list(self.metrics_history[metric_name])
            
            if duration_seconds is not None:
                cutoff_time = time.time() - duration_seconds
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            return metrics
    
    def get_active_alerts(self, level_filter: Optional[str] = None) -> List[AlertEvent]:
        """Get currently active alerts."""
        with self._lock:
            alerts = list(self.active_alerts.values())
            
            if level_filter:
                alerts = [a for a in alerts if a.level == level_filter]
            
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        with self._lock:
            return self.system_health.copy()
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        with self._lock:
            current_time = time.time()
            
            # Collect recent metrics (last 5 minutes)
            recent_metrics = {}
            for name, history in self.metrics_history.items():
                recent = [m for m in history if current_time - m.timestamp <= 300]
                if recent:
                    recent_metrics[name] = {
                        "count": len(recent),
                        "avg_value": sum(m.value for m in recent) / len(recent),
                        "min_value": min(m.value for m in recent),
                        "max_value": max(m.value for m in recent),
                        "last_value": recent[-1].value,
                    }
            
            # Count alerts by level
            alert_counts = defaultdict(int)
            for alert in self.alerts:
                if current_time - alert.timestamp <= 3600:  # Last hour
                    alert_counts[alert.level] += 1
            
            return {
                "timestamp": current_time,
                "system_health": self.system_health,
                "recent_metrics": recent_metrics,
                "alert_summary": dict(alert_counts),
                "active_alerts_count": len(self.active_alerts),
                "total_metrics_tracked": len(self.metrics_history),
            }
    
    def export_metrics(self, format: str = "json", duration_seconds: Optional[float] = None) -> str:
        """Export metrics in specified format."""
        with self._lock:
            cutoff_time = time.time() - (duration_seconds or 3600)
            
            export_data = {
                "export_timestamp": time.time(),
                "duration_seconds": duration_seconds,
                "metrics": {},
                "alerts": [],
                "system_health": self.system_health,
            }
            
            # Export metrics
            for name, history in self.metrics_history.items():
                filtered_metrics = [
                    asdict(m) for m in history 
                    if m.timestamp >= cutoff_time
                ]
                if filtered_metrics:
                    export_data["metrics"][name] = filtered_metrics
            
            # Export alerts
            export_data["alerts"] = [
                asdict(alert) for alert in self.alerts
                if alert.timestamp >= cutoff_time
            ]
            
            if format.lower() == "json":
                return json.dumps(export_data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
    
    def clear_history(self, metric_name: Optional[str] = None) -> None:
        """Clear metric history."""
        with self._lock:
            if metric_name:
                if metric_name in self.metrics_history:
                    self.metrics_history[metric_name].clear()
            else:
                self.metrics_history.clear()
                self.alerts.clear()
                self.active_alerts.clear()
    
    def set_baseline(self, metric_name: str, baseline_value: float) -> None:
        """Set performance baseline for a metric."""
        self.baselines[metric_name] = baseline_value
    
    def check_performance_regression(self, metric_name: str, window_size: int = 100) -> Optional[Dict[str, Any]]:
        """Check for performance regression against baseline."""
        with self._lock:
            if metric_name not in self.metrics_history or metric_name not in self.baselines:
                return None
            
            recent_metrics = list(self.metrics_history[metric_name])[-window_size:]
            if len(recent_metrics) < 10:  # Need at least 10 samples
                return None
            
            baseline = self.baselines[metric_name]
            recent_avg = sum(m.value for m in recent_metrics) / len(recent_metrics)
            
            # Calculate regression percentage
            regression_pct = ((recent_avg - baseline) / baseline) * 100
            
            return {
                "metric_name": metric_name,
                "baseline": baseline,
                "recent_average": recent_avg,
                "regression_percent": regression_pct,
                "is_regression": regression_pct > 10,  # 10% threshold
                "sample_count": len(recent_metrics),
            }


# Global dashboard instance
_dashboard = None

def get_dashboard() -> MonitoringDashboard:
    """Get global monitoring dashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = MonitoringDashboard()
    return _dashboard

def record_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None, 
                 alert_threshold: Optional[float] = None, critical_threshold: Optional[float] = None) -> None:
    """Convenience function to record a metric."""
    metric = SystemMetric(
        name=name,
        value=value,
        timestamp=time.time(),
        tags=tags or {},
        alert_threshold=alert_threshold,
        critical_threshold=critical_threshold,
    )
    get_dashboard().record_metric(metric)

def get_health_status() -> str:
    """Get current system health status."""
    return get_dashboard().get_system_health()["overall_status"]