"""Security monitoring and threat detection for industrial RL systems."""

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import jax.numpy as jnp
    import numpy as np
except ImportError:
    import numpy as np
    jnp = np

from ..core.types import Array
from ..exceptions import SecurityError


class ThreatLevel(Enum):
    """Security threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    INVALID_INPUT = "invalid_input"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    INJECTION_ATTEMPT = "injection_attempt"
    DATA_EXFILTRATION = "data_exfiltration"


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: datetime
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source: str
    description: str
    metadata: Dict[str, Any]
    mitigated: bool = False


@dataclass
class SecurityMetrics:
    """Security monitoring metrics."""
    total_events: int
    events_by_level: Dict[ThreatLevel, int]
    detection_rate: float
    false_positive_rate: float
    response_time_ms: float
    detection_accuracy: float


class ThreatDetector(ABC):
    """Base class for threat detection algorithms."""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.logger = logging.getLogger(f"threat_detector.{name}")
    
    @abstractmethod
    def detect_threat(self, data: Any, metadata: Dict[str, Any]) -> Tuple[bool, ThreatLevel, str]:
        """Detect threat in data. Returns (is_threat, level, description)."""
        pass
    
    def enable(self):
        """Enable detector."""
        self.enabled = True
        self.logger.info(f"Detector {self.name} enabled")
    
    def disable(self):
        """Disable detector."""
        self.enabled = False
        self.logger.info(f"Detector {self.name} disabled")
    
    def is_enabled(self) -> bool:
        """Check if detector is enabled."""
        return self.enabled


class InputValidationDetector(ThreatDetector):
    """Detects malicious input patterns."""
    
    def __init__(self):
        super().__init__("input_validation")
        self.malicious_patterns = [
            "__import__", "eval(", "exec(", "subprocess", "os.system",
            "pickle.loads", "../", "..\\", "<script>", "javascript:"
        ]
        self.max_input_size = 1024 * 1024  # 1MB
    
    def detect_threat(self, data: Any, metadata: Dict[str, Any]) -> Tuple[bool, ThreatLevel, str]:
        """Detect malicious input patterns."""
        if isinstance(data, str):
            # Check for malicious patterns
            data_lower = data.lower()
            for pattern in self.malicious_patterns:
                if pattern in data_lower:
                    return True, ThreatLevel.HIGH, f"Malicious pattern detected: {pattern}"
            
            # Check input size
            if len(data) > self.max_input_size:
                return True, ThreatLevel.MEDIUM, f"Input size exceeds limit: {len(data)} bytes"
        
        elif isinstance(data, (list, tuple, dict)):
            # Check for deeply nested structures (potential DoS)
            if self._check_depth(data, 0) > 100:
                return True, ThreatLevel.MEDIUM, "Deeply nested data structure detected"
        
        return False, ThreatLevel.LOW, ""
    
    def _check_depth(self, obj, current_depth):
        """Check nesting depth of data structure."""
        if current_depth > 100:  # Prevent infinite recursion
            return current_depth
        
        if isinstance(obj, dict):
            return max([self._check_depth(v, current_depth + 1) for v in obj.values()] + [current_depth])
        elif isinstance(obj, (list, tuple)):
            return max([self._check_depth(item, current_depth + 1) for item in obj] + [current_depth])
        else:
            return current_depth


class RateLimitDetector(ThreatDetector):
    """Detects rate limiting violations."""
    
    def __init__(self, max_requests_per_minute: int = 100):
        super().__init__("rate_limit")
        self.max_requests_per_minute = max_requests_per_minute
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_requests_per_minute))
    
    def detect_threat(self, data: Any, metadata: Dict[str, Any]) -> Tuple[bool, ThreatLevel, str]:
        """Detect rate limit violations."""
        source = metadata.get("source", "unknown")
        current_time = time.time()
        
        # Clean old requests
        self.request_counts[source] = deque(
            [t for t in self.request_counts[source] if current_time - t < 60],
            maxlen=self.max_requests_per_minute
        )
        
        # Add current request
        self.request_counts[source].append(current_time)
        
        # Check if limit exceeded
        if len(self.request_counts[source]) > self.max_requests_per_minute:
            return True, ThreatLevel.MEDIUM, f"Rate limit exceeded: {len(self.request_counts[source])} requests/minute"
        
        return False, ThreatLevel.LOW, ""


class AnomalyDetector(ThreatDetector):
    """Detects anomalous behavior patterns."""
    
    def __init__(self, baseline_size: int = 1000):
        super().__init__("anomaly")
        self.baseline_size = baseline_size
        self.baseline_data: deque = deque(maxlen=baseline_size)
        self.baseline_established = False
        self.anomaly_threshold = 3.0  # Standard deviations
    
    def detect_threat(self, data: Any, metadata: Dict[str, Any]) -> Tuple[bool, ThreatLevel, str]:
        """Detect anomalous patterns in numerical data."""
        if not isinstance(data, (int, float, np.ndarray)):
            return False, ThreatLevel.LOW, ""
        
        # Convert to scalar if array
        if isinstance(data, np.ndarray):
            if data.size == 1:
                data = float(data.item())
            else:
                data = float(np.mean(data))  # Use mean for arrays
        
        # Build baseline
        if not self.baseline_established:
            self.baseline_data.append(data)
            if len(self.baseline_data) >= self.baseline_size:
                self.baseline_established = True
            return False, ThreatLevel.LOW, ""
        
        # Check for anomalies
        baseline_array = np.array(self.baseline_data)
        mean_val = np.mean(baseline_array)
        std_val = np.std(baseline_array)
        
        if std_val > 0:
            z_score = abs(data - mean_val) / std_val
            if z_score > self.anomaly_threshold:
                threat_level = ThreatLevel.HIGH if z_score > 5.0 else ThreatLevel.MEDIUM
                return True, threat_level, f"Anomalous value detected: z-score={z_score:.2f}"
        
        # Update baseline with normal values
        self.baseline_data.append(data)
        return False, ThreatLevel.LOW, ""


class InjectionDetector(ThreatDetector):
    """Detects injection attempts."""
    
    def __init__(self):
        super().__init__("injection")
        self.sql_patterns = [
            "union select", "drop table", "insert into", "delete from",
            "exec ", "execute ", "sp_", "xp_", "'; --", "' or '1'='1"
        ]
        self.cmd_patterns = [
            "; rm -rf", "; cat ", "; ls ", "| nc ", "&& ", "|| ",
            "$(", "`", "eval(", "system(", "exec("
        ]
    
    def detect_threat(self, data: Any, metadata: Dict[str, Any]) -> Tuple[bool, ThreatLevel, str]:
        """Detect injection patterns."""
        if not isinstance(data, str):
            return False, ThreatLevel.LOW, ""
        
        data_lower = data.lower()
        
        # Check for SQL injection
        for pattern in self.sql_patterns:
            if pattern in data_lower:
                return True, ThreatLevel.HIGH, f"SQL injection pattern detected: {pattern}"
        
        # Check for command injection
        for pattern in self.cmd_patterns:
            if pattern in data_lower:
                return True, ThreatLevel.CRITICAL, f"Command injection pattern detected: {pattern}"
        
        return False, ThreatLevel.LOW, ""


class SecurityMonitor:
    """Main security monitoring system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize security monitor."""
        self.config = config or {}
        self.logger = logging.getLogger("security_monitor")
        
        # Security event storage
        self.security_events: deque = deque(maxlen=10000)
        
        # Threat detection
        self.detectors: Dict[str, ThreatDetector] = {
            "input_validation": InputValidationDetector(),
            "rate_limit": RateLimitDetector(
                max_requests_per_minute=self.config.get("max_requests_per_minute", 100)
            ),
            "anomaly": AnomalyDetector(
                baseline_size=self.config.get("anomaly_baseline_size", 1000)
            ),
            "injection": InjectionDetector()
        }
        
        # Blocking and mitigation
        self.blocked_sources: Set[str] = set()
        self.blocked_until: Dict[str, datetime] = {}
        self.auto_block_threshold = self.config.get("auto_block_threshold", 5)
        self.block_duration_minutes = self.config.get("block_duration_minutes", 60)
        
        # Metrics
        self.blocked_requests = 0
        self.false_positives = 0
        
    def monitor_data(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Monitor data for security threats."""
        metadata = metadata or {}
        source = metadata.get("source", "unknown")
        
        # Check if source is blocked
        if self._is_source_blocked(source):
            self.blocked_requests += 1
            self.logger.warning(f"Blocked request from {source}")
            return False
        
        # Run threat detection
        for detector_name, detector in self.detectors.items():
            if not detector.is_enabled():
                continue
                
            try:
                is_threat, threat_level, description = detector.detect_threat(data, metadata)
                
                if is_threat:
                    event = SecurityEvent(
                        timestamp=datetime.now(),
                        event_type=self._get_event_type(detector_name),
                        threat_level=threat_level,
                        source=source,
                        description=description,
                        metadata=metadata
                    )
                    
                    self._handle_security_event(event)
                    return False  # Block the request
                    
            except Exception as e:
                self.logger.error(f"Error in detector {detector_name}: {e}")
        
        return True  # Allow the request
    
    def _get_event_type(self, detector_name: str) -> SecurityEventType:
        """Map detector name to event type."""
        mapping = {
            "input_validation": SecurityEventType.INVALID_INPUT,
            "rate_limit": SecurityEventType.RATE_LIMIT_EXCEEDED,
            "anomaly": SecurityEventType.ANOMALOUS_BEHAVIOR,
            "injection": SecurityEventType.INJECTION_ATTEMPT
        }
        return mapping.get(detector_name, SecurityEventType.ANOMALOUS_BEHAVIOR)
    
    def _handle_security_event(self, event: SecurityEvent):
        """Handle detected security event."""
        # Store event
        self.security_events.append(event)
        
        # Log event
        log_level = {
            ThreatLevel.LOW: logging.INFO,
            ThreatLevel.MEDIUM: logging.WARNING,
            ThreatLevel.HIGH: logging.ERROR,
            ThreatLevel.CRITICAL: logging.CRITICAL
        }.get(event.threat_level, logging.INFO)
        
        self.logger.log(
            log_level,
            f"Security event: {event.event_type.value} from {event.source} - {event.description}"
        )
        
        # Apply automatic mitigation
        self._apply_mitigation(event)
    
    def _apply_mitigation(self, event: SecurityEvent):
        """Apply automatic mitigation measures."""
        source = event.source
        
        # Count recent events from this source
        recent_events = [
            e for e in self.security_events
            if e.source == source and 
            (datetime.now() - e.timestamp).total_seconds() < 300  # Last 5 minutes
        ]
        
        # Auto-block if threshold exceeded
        if len(recent_events) >= self.auto_block_threshold:
            self._block_source(source, self.block_duration_minutes)
            event.mitigated = True
        
        # Immediate block for critical threats
        if event.threat_level == ThreatLevel.CRITICAL:
            self._block_source(source, self.block_duration_minutes * 2)
            event.mitigated = True
    
    def _block_source(self, source: str, duration_minutes: int):
        """Block a source for specified duration."""
        self.blocked_sources.add(source)
        self.blocked_until[source] = datetime.now() + timedelta(minutes=duration_minutes)
        
        self.logger.warning(f"Blocked source {source} for {duration_minutes} minutes")
    
    def _is_source_blocked(self, source: str) -> bool:
        """Check if source is currently blocked."""
        if source not in self.blocked_sources:
            return False
        
        # Check if block has expired
        if source in self.blocked_until:
            if datetime.now() > self.blocked_until[source]:
                self.blocked_sources.remove(source)
                del self.blocked_until[source]
                self.logger.info(f"Unblocked source {source} (block expired)")
                return False
        
        return True
    
    def unblock_source(self, source: str):
        """Manually unblock a source."""
        if source in self.blocked_sources:
            self.blocked_sources.remove(source)
            if source in self.blocked_until:
                del self.blocked_until[source]
            self.logger.info(f"Manually unblocked source {source}")
    
    def get_security_metrics(self) -> SecurityMetrics:
        """Get security monitoring metrics."""
        # Count events by level
        events_by_level = defaultdict(int)
        total_events = len(self.security_events)
        
        for event in self.security_events:
            events_by_level[event.threat_level] += 1
        
        # Calculate detection rate (simplified)
        detection_rate = total_events / max(total_events + self.blocked_requests, 1)
        
        # Calculate false positive rate (simplified - would need manual verification)
        false_positive_rate = self.false_positives / max(total_events, 1)
        
        # Average response time (simplified)
        response_time_ms = 10.0  # Placeholder
        
        # Detection accuracy (simplified)
        detection_accuracy = max(0.0, 1.0 - false_positive_rate)
        
        return SecurityMetrics(
            total_events=total_events,
            events_by_level=dict(events_by_level),
            detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
            response_time_ms=response_time_ms,
            detection_accuracy=detection_accuracy
        )
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        metrics = self.get_security_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "total_events": metrics.total_events,
                "blocked_requests": self.blocked_requests,
                "detection_rate": f"{metrics.detection_rate:.2%}",
                "false_positive_rate": f"{metrics.false_positive_rate:.2%}",
                "detection_accuracy": f"{metrics.detection_accuracy:.2%}",
                "avg_response_time": f"{metrics.response_time_ms:.2f}ms"
            },
            "events_by_level": {
                level.value: count for level, count in metrics.events_by_level.items()
            },
            "active_blocks": len(self.blocked_sources),
            "detector_status": {
                name: detector.is_enabled() for name, detector in self.detectors.items()
            },
            "recent_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "type": event.event_type,
                    "level": event.threat_level.value,
                    "source": event.source,
                    "mitigated": event.mitigated
                }
                for event in list(self.security_events)[-20:]  # Last 20 events
            ]
        }
        
    def enable_detector(self, detector_name: str):
        """Enable specific detector."""
        if detector_name in self.detectors:
            self.detectors[detector_name].enable()
        else:
            self.logger.error(f"Unknown detector: {detector_name}")
            
    def disable_detector(self, detector_name: str):
        """Disable specific detector."""
        if detector_name in self.detectors:
            self.detectors[detector_name].disable()
        else:
            self.logger.error(f"Unknown detector: {detector_name}")
            
    def reset_security_state(self):
        """Reset security monitoring state."""
        self.security_events.clear()
        self.blocked_sources.clear()
        self.blocked_until.clear()
        self.blocked_requests = 0
        self.false_positives = 0
        
        # Reset detector states
        for detector in self.detectors.values():
            if hasattr(detector, 'baseline_data'):
                detector.baseline_data.clear()
                detector.baseline_established = False
        
        self.logger.info("Security state reset")


# Global security monitor instance
_security_monitor = None

def get_security_monitor() -> SecurityMonitor:
    """Get global security monitor instance."""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor()
    return _security_monitor