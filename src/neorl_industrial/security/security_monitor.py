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


@dataclass
class SecurityEvent:
    """Security event record."""
    
    timestamp: datetime
    event_type: str
    threat_level: ThreatLevel
    source: str
    details: Dict[str, Any]
    mitigated: bool = False
    

@dataclass
class SecurityMetrics:
    """Security monitoring metrics."""
    
    total_events: int
    events_by_level: Dict[ThreatLevel, int]
    blocked_attacks: int
    false_positives: int
    detection_accuracy: float
    response_time_ms: float


class SecurityDetector(ABC):
    """Abstract base class for security detectors."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.logger = logging.getLogger(f"SecurityDetector.{name}")
        
    @abstractmethod
    def detect(self, input_data: Any, context: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect security threats in input data."""
        pass
        
    def is_enabled(self) -> bool:
        """Check if detector is enabled."""
        return self.enabled
        
    def enable(self):
        """Enable detector."""
        self.enabled = True
        self.logger.info(f"{self.name} detector enabled")
        
    def disable(self):
        """Disable detector."""
        self.enabled = False
        self.logger.info(f"{self.name} detector disabled")


class AnomalyDetector(SecurityDetector):
    """Detect anomalous input patterns that may indicate attacks."""
    
    def __init__(self, baseline_samples: int = 1000, anomaly_threshold: float = 3.0):
        super().__init__("AnomalyDetector")
        self.baseline_samples = baseline_samples
        self.anomaly_threshold = anomaly_threshold
        
        # Baseline statistics
        self.baseline_data = deque(maxlen=baseline_samples)
        self.baseline_mean = None
        self.baseline_std = None
        self.baseline_established = False
        
    def detect(self, input_data: Any, context: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect anomalous patterns in input data."""
        if not self.enabled:
            return []
            
        events = []
        
        try:
            data_array = jnp.asarray(input_data).flatten()
            
            # Update baseline
            self._update_baseline(data_array)
            
            if self.baseline_established:
                # Calculate anomaly score
                anomaly_score = self._calculate_anomaly_score(data_array)
                
                if anomaly_score > self.anomaly_threshold:
                    threat_level = self._determine_threat_level(anomaly_score)
                    
                    event = SecurityEvent(
                        timestamp=datetime.now(),
                        event_type="anomaly_detected",
                        threat_level=threat_level,
                        source="input_data",
                        details={
                            "anomaly_score": float(anomaly_score),
                            "threshold": self.anomaly_threshold,
                            "data_shape": data_array.shape,
                            "context": context
                        }
                    )
                    events.append(event)
                    
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            
        return events
        
    def _update_baseline(self, data: Array):
        """Update baseline statistics with new data."""
        self.baseline_data.extend(data.flatten())
        
        if len(self.baseline_data) >= self.baseline_samples:
            self.baseline_mean = jnp.mean(jnp.array(list(self.baseline_data)))
            self.baseline_std = jnp.std(jnp.array(list(self.baseline_data)))
            self.baseline_established = True
            
    def _calculate_anomaly_score(self, data: Array) -> float:
        """Calculate anomaly score using z-score method."""
        if not self.baseline_established:
            return 0.0
            
        z_scores = jnp.abs((data - self.baseline_mean) / (self.baseline_std + 1e-8))
        return float(jnp.max(z_scores))
        
    def _determine_threat_level(self, anomaly_score: float) -> ThreatLevel:
        """Determine threat level based on anomaly score."""
        if anomaly_score > 10.0:
            return ThreatLevel.CRITICAL
        elif anomaly_score > 6.0:
            return ThreatLevel.HIGH
        elif anomaly_score > 4.0:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW


class AdversarialDetector(SecurityDetector):
    """Detect adversarial attacks on RL agents."""
    
    def __init__(self, perturbation_threshold: float = 0.1):
        super().__init__("AdversarialDetector")
        self.perturbation_threshold = perturbation_threshold
        self.previous_input = None
        
    def detect(self, input_data: Any, context: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect adversarial perturbations in input."""
        if not self.enabled:
            return []
            
        events = []
        
        try:
            current_input = jnp.asarray(input_data)
            
            if self.previous_input is not None:
                # Calculate perturbation magnitude
                perturbation = jnp.linalg.norm(current_input - self.previous_input)
                
                # Check for sudden large perturbations
                if perturbation > self.perturbation_threshold:
                    # Additional checks for adversarial patterns
                    if self._is_adversarial_pattern(current_input, self.previous_input):
                        event = SecurityEvent(
                            timestamp=datetime.now(),
                            event_type="adversarial_attack",
                            threat_level=ThreatLevel.HIGH,
                            source="input_perturbation",
                            details={
                                "perturbation_magnitude": float(perturbation),
                                "threshold": self.perturbation_threshold,
                                "pattern_detected": True,
                                "context": context
                            }
                        )
                        events.append(event)
                        
            self.previous_input = current_input
            
        except Exception as e:
            self.logger.error(f"Adversarial detection failed: {e}")
            
        return events
        
    def _is_adversarial_pattern(self, current: Array, previous: Array) -> bool:
        """Check for patterns typical of adversarial attacks."""
        try:
            # Check for uniform perturbations (common in adversarial attacks)
            diff = current - previous
            
            # Calculate variance of perturbation
            perturbation_variance = jnp.var(diff)
            
            # Low variance suggests uniform perturbation (adversarial)
            return perturbation_variance < 0.01
            
        except Exception:
            return False


class InputValidationDetector(SecurityDetector):
    """Detect malicious or malformed inputs."""
    
    def __init__(self, max_input_size: int = 10000):
        super().__init__("InputValidationDetector")
        self.max_input_size = max_input_size
        self.blacklisted_patterns = [
            # Add patterns that indicate malicious input
            "eval(",
            "exec(",
            "__import__",
            "subprocess",
            "os.system"
        ]
        
    def detect(self, input_data: Any, context: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect malicious input patterns."""
        if not self.enabled:
            return []
            
        events = []
        
        try:
            # Check input size
            if self._check_input_size(input_data):
                event = SecurityEvent(
                    timestamp=datetime.now(),
                    event_type="oversized_input",
                    threat_level=ThreatLevel.MEDIUM,
                    source="input_validation",
                    details={
                        "input_size": len(str(input_data)),
                        "max_size": self.max_input_size,
                        "context": context
                    }
                )
                events.append(event)
                
            # Check for malicious patterns
            if self._check_malicious_patterns(input_data):
                event = SecurityEvent(
                    timestamp=datetime.now(),
                    event_type="malicious_pattern",
                    threat_level=ThreatLevel.CRITICAL,
                    source="input_validation",
                    details={
                        "patterns_found": self._get_found_patterns(input_data),
                        "context": context
                    }
                )
                events.append(event)
                
        except Exception as e:
            self.logger.error(f"Input validation detection failed: {e}")
            
        return events
        
    def _check_input_size(self, input_data: Any) -> bool:
        """Check if input exceeds size limits."""
        try:
            input_str = str(input_data)
            return len(input_str) > self.max_input_size
        except Exception:
            return False
            
    def _check_malicious_patterns(self, input_data: Any) -> bool:
        """Check for blacklisted patterns in input."""
        try:
            input_str = str(input_data).lower()
            return any(pattern in input_str for pattern in self.blacklisted_patterns)
        except Exception:
            return False
            
    def _get_found_patterns(self, input_data: Any) -> List[str]:
        """Get list of malicious patterns found in input."""
        try:
            input_str = str(input_data).lower()
            return [pattern for pattern in self.blacklisted_patterns if pattern in input_str]
        except Exception:
            return []


class RateLimitDetector(SecurityDetector):
    """Detect rate limiting violations and potential DoS attacks."""
    
    def __init__(self, max_requests_per_minute: int = 100, max_requests_per_hour: int = 1000):
        super().__init__("RateLimitDetector")
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        
        # Request tracking
        self.request_times = deque()
        self.source_requests = defaultdict(deque)
        
    def detect(self, input_data: Any, context: Dict[str, Any]) -> List[SecurityEvent]:
        """Detect rate limiting violations."""
        if not self.enabled:
            return []
            
        events = []
        current_time = datetime.now()
        source = context.get("source_ip", "unknown")
        
        try:
            # Track request
            self.request_times.append(current_time)
            self.source_requests[source].append(current_time)
            
            # Clean old requests
            self._clean_old_requests(current_time)
            
            # Check global rate limits
            minute_requests = self._count_recent_requests(self.request_times, timedelta(minutes=1))
            hour_requests = self._count_recent_requests(self.request_times, timedelta(hours=1))
            
            if minute_requests > self.max_requests_per_minute:
                event = SecurityEvent(
                    timestamp=current_time,
                    event_type="rate_limit_exceeded",
                    threat_level=ThreatLevel.HIGH,
                    source="global",
                    details={
                        "requests_per_minute": minute_requests,
                        "limit": self.max_requests_per_minute,
                        "context": context
                    }
                )
                events.append(event)
                
            if hour_requests > self.max_requests_per_hour:
                event = SecurityEvent(
                    timestamp=current_time,
                    event_type="rate_limit_exceeded",
                    threat_level=ThreatLevel.CRITICAL,
                    source="global",
                    details={
                        "requests_per_hour": hour_requests,
                        "limit": self.max_requests_per_hour,
                        "context": context
                    }
                )
                events.append(event)
                
            # Check per-source rate limits
            source_minute_requests = self._count_recent_requests(
                self.source_requests[source], timedelta(minutes=1)
            )
            
            if source_minute_requests > self.max_requests_per_minute // 10:  # 10% of global limit per source
                event = SecurityEvent(
                    timestamp=current_time,
                    event_type="source_rate_limit_exceeded", 
                    threat_level=ThreatLevel.MEDIUM,
                    source=source,
                    details={
                        "source_requests_per_minute": source_minute_requests,
                        "limit": self.max_requests_per_minute // 10,
                        "context": context
                    }
                )
                events.append(event)
                
        except Exception as e:
            self.logger.error(f"Rate limit detection failed: {e}")
            
        return events
        
    def _clean_old_requests(self, current_time: datetime):
        """Remove old request records."""
        cutoff_time = current_time - timedelta(hours=1)
        
        # Clean global requests
        while self.request_times and self.request_times[0] < cutoff_time:
            self.request_times.popleft()
            
        # Clean per-source requests
        for source in list(self.source_requests.keys()):
            while (self.source_requests[source] and 
                   self.source_requests[source][0] < cutoff_time):
                self.source_requests[source].popleft()
                
            # Remove empty source records
            if not self.source_requests[source]:
                del self.source_requests[source]
                
    def _count_recent_requests(self, request_queue: deque, time_window: timedelta) -> int:
        """Count requests within time window."""
        if not request_queue:
            return 0
            
        cutoff_time = datetime.now() - time_window
        return sum(1 for req_time in request_queue if req_time >= cutoff_time)


class SecurityMonitor:
    """Comprehensive security monitoring system for industrial RL."""
    
    def __init__(self, 
                 enabled_detectors: Optional[List[str]] = None,
                 log_events: bool = True,
                 auto_mitigation: bool = True):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.log_events = log_events
        self.auto_mitigation = auto_mitigation
        
        # Initialize detectors
        self.detectors = {
            "anomaly": AnomalyDetector(),
            "adversarial": AdversarialDetector(),
            "input_validation": InputValidationDetector(),
            "rate_limit": RateLimitDetector()
        }
        
        # Enable specified detectors
        if enabled_detectors:
            for name, detector in self.detectors.items():
                if name not in enabled_detectors:
                    detector.disable()
                    
        # Event tracking
        self.security_events = deque(maxlen=10000)
        self.blocked_requests = 0
        self.false_positives = 0
        
        # Threat mitigation
        self.blocked_sources: Set[str] = set()
        self.blocked_until: Dict[str, datetime] = {}
        
        self.logger.info("Security monitor initialized")
        
    def monitor_input(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[SecurityEvent]]:
        """Monitor input for security threats."""
        if context is None:
            context = {}
            
        start_time = time.time()
        all_events = []
        
        # Check if source is blocked
        source = context.get("source_ip", "unknown")
        if self._is_source_blocked(source):
            block_event = SecurityEvent(
                timestamp=datetime.now(),
                event_type="blocked_source",
                threat_level=ThreatLevel.HIGH,
                source=source,
                details={"reason": "source_blocked", "context": context},
                mitigated=True
            )
            all_events.append(block_event)
            return False, all_events
            
        # Run all enabled detectors
        for detector_name, detector in self.detectors.items():
            if detector.is_enabled():
                try:
                    events = detector.detect(input_data, context)
                    all_events.extend(events)
                except Exception as e:
                    self.logger.error(f"Detector {detector_name} failed: {e}")
                    
        # Process detected events
        should_allow = self._process_events(all_events, context)
        
        # Log events
        if self.log_events and all_events:
            for event in all_events:
                self._log_event(event)
                
        # Store events
        self.security_events.extend(all_events)
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000
        context["response_time_ms"] = response_time
        
        return should_allow, all_events
        
    def _is_source_blocked(self, source: str) -> bool:
        """Check if source is currently blocked."""
        if source in self.blocked_sources:
            # Check if block has expired
            if source in self.blocked_until:
                if datetime.now() > self.blocked_until[source]:
                    self._unblock_source(source)
                    return False
            return True
        return False
        
    def _process_events(self, events: List[SecurityEvent], context: Dict[str, Any]) -> bool:
        """Process security events and determine if request should be allowed."""
        if not events:
            return True
            
        if not self.auto_mitigation:
            # Just log without mitigation
            return True
            
        # Check for critical threats
        critical_events = [e for e in events if e.threat_level == ThreatLevel.CRITICAL]
        high_events = [e for e in events if e.threat_level == ThreatLevel.HIGH]
        
        if critical_events:
            # Block source for critical threats
            source = context.get("source_ip", "unknown")
            self._block_source(source, duration_minutes=60)
            self.blocked_requests += 1
            return False
            
        if len(high_events) >= 2:
            # Block source for multiple high-level threats
            source = context.get("source_ip", "unknown")
            self._block_source(source, duration_minutes=10)
            self.blocked_requests += 1
            return False
            
        return True
        
    def _block_source(self, source: str, duration_minutes: int = 10):
        """Block source for specified duration."""
        self.blocked_sources.add(source)
        self.blocked_until[source] = datetime.now() + timedelta(minutes=duration_minutes)
        self.logger.warning(f"Blocked source {source} for {duration_minutes} minutes")
        
    def _unblock_source(self, source: str):
        """Unblock source."""
        self.blocked_sources.discard(source)
        self.blocked_until.pop(source, None)
        self.logger.info(f"Unblocked source {source}")
        
    def _log_event(self, event: SecurityEvent):
        """Log security event."""
        level_map = {
            ThreatLevel.LOW: logging.INFO,
            ThreatLevel.MEDIUM: logging.WARNING,
            ThreatLevel.HIGH: logging.ERROR,
            ThreatLevel.CRITICAL: logging.CRITICAL
        }
        
        log_level = level_map.get(event.threat_level, logging.INFO)
        
        self.logger.log(
            log_level,
            f"Security Event: {event.event_type} | "
            f"Level: {event.threat_level.value} | "
            f"Source: {event.source} | "
            f"Details: {event.details}"
        )
        
    def get_security_metrics(self) -> SecurityMetrics:
        """Get comprehensive security metrics."""
        total_events = len(self.security_events)
        events_by_level = defaultdict(int)
        
        for event in self.security_events:
            events_by_level[event.threat_level] += 1
            
        # Calculate detection accuracy (simplified)
        detection_accuracy = 1.0 - (self.false_positives / max(total_events, 1))
        
        # Calculate average response time
        recent_events = list(self.security_events)[-100:]  # Last 100 events
        response_times = [
            event.details.get("response_time_ms", 0) 
            for event in recent_events 
            if "response_time_ms" in event.details
        ]
        avg_response_time = np.mean(response_times) if response_times else 0.0
        
        return SecurityMetrics(
            total_events=total_events,
            events_by_level=dict(events_by_level),
            blocked_attacks=self.blocked_requests,
            false_positives=self.false_positives,
            detection_accuracy=detection_accuracy,
            response_time_ms=avg_response_time
        )
        
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        metrics = self.get_security_metrics()
        
        return {
            "summary": {
                "total_events": metrics.total_events,
                "blocked_attacks": metrics.blocked_attacks,
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