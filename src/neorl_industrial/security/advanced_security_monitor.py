"""Advanced security monitoring with threat detection and response."""

import time
import hashlib
import json
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
import numpy as np
from functools import wraps
import ipaddress

from ..monitoring.logger import get_logger
from ..core.types import Array, MetricsDict


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_DENIED = "auth_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_BREACH_ATTEMPT = "data_breach"
    INJECTION_ATTEMPT = "injection_attempt"
    RATE_LIMIT_EXCEEDED = "rate_limit"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_type: SecurityEventType
    threat_level: ThreatLevel
    timestamp: float
    source_ip: str
    user_id: Optional[str]
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_type': self.event_type.value,
            'threat_level': self.threat_level.value,
            'timestamp': self.timestamp,
            'source_ip': self.source_ip,
            'user_id': self.user_id,
            'description': self.description,
            'metadata': self.metadata
        }


@dataclass
class ThreatProfile:
    """Profile of potential threat source."""
    source_ip: str
    first_seen: float
    last_seen: float
    event_count: int = 0
    event_types: Set[SecurityEventType] = field(default_factory=set)
    risk_score: float = 0.0
    is_blocked: bool = False
    block_reason: str = ""
    
    def update_risk_score(self) -> None:
        """Update risk score based on events."""
        # Base risk from event count
        count_risk = min(self.event_count * 0.1, 0.5)
        
        # Risk from event diversity
        diversity_risk = len(self.event_types) * 0.1
        
        # Time-based risk (recent activity is more risky)
        time_risk = max(0, 0.3 - (time.time() - self.last_seen) / 3600)
        
        self.risk_score = min(1.0, count_risk + diversity_risk + time_risk)


class AdvancedSecurityMonitor:
    """Advanced security monitoring with ML-based threat detection."""
    
    def __init__(
        self,
        max_events: int = 10000,
        rate_limit_window: int = 300,  # 5 minutes
        rate_limit_threshold: int = 100,
        anomaly_detection_window: int = 1000,
        enable_ml_detection: bool = True
    ):
        """Initialize security monitor.
        
        Args:
            max_events: Maximum events to keep in memory
            rate_limit_window: Rate limiting window in seconds
            rate_limit_threshold: Max requests per window
            anomaly_detection_window: Window size for anomaly detection
            enable_ml_detection: Enable ML-based threat detection
        """
        self.max_events = max_events
        self.rate_limit_window = rate_limit_window
        self.rate_limit_threshold = rate_limit_threshold
        self.anomaly_detection_window = anomaly_detection_window
        self.enable_ml_detection = enable_ml_detection
        
        # Event storage
        self.security_events = deque(maxlen=max_events)
        self.threat_profiles: Dict[str, ThreatProfile] = {}
        
        # Rate limiting
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Blocked IPs and users
        self.blocked_ips: Set[str] = set()
        self.blocked_users: Set[str] = set()
        
        # Anomaly detection
        self.behavior_baselines: Dict[str, Dict[str, float]] = {}
        self.anomaly_detector = None
        
        # Security rules
        self.security_rules: List[Callable[[SecurityEvent], bool]] = []
        self.auto_response_rules: Dict[ThreatLevel, Callable[[SecurityEvent], None]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Monitoring
        self.logger = get_logger("AdvancedSecurityMonitor")
        
        # Initialize components
        self._initialize_default_rules()
        if self.enable_ml_detection:
            self._initialize_ml_detection()
        
        self.logger.info("Advanced security monitor initialized")
    
    def _initialize_default_rules(self) -> None:
        """Initialize default security rules."""
        
        # Rule: Multiple authentication failures
        def auth_failure_rule(event: SecurityEvent) -> bool:
            if event.event_type != SecurityEventType.AUTHENTICATION_FAILURE:
                return False
            
            recent_failures = [
                e for e in self.security_events
                if (e.event_type == SecurityEventType.AUTHENTICATION_FAILURE and
                    e.source_ip == event.source_ip and
                    time.time() - e.timestamp < 300)  # 5 minutes
            ]
            
            return len(recent_failures) >= 5
        
        # Rule: Suspicious SQL injection patterns
        def injection_rule(event: SecurityEvent) -> bool:
            if event.event_type != SecurityEventType.INJECTION_ATTEMPT:
                return False
            
            suspicious_patterns = [
                "union select", "drop table", "insert into",
                "delete from", "update set", "script>", "javascript:"
            ]
            
            description = event.description.lower()
            return any(pattern in description for pattern in suspicious_patterns)
        
        # Rule: High-frequency requests (potential DDoS)
        def rate_limit_rule(event: SecurityEvent) -> bool:
            if event.event_type != SecurityEventType.RATE_LIMIT_EXCEEDED:
                return False
            
            return event.metadata.get('request_count', 0) > self.rate_limit_threshold
        
        self.security_rules.extend([auth_failure_rule, injection_rule, rate_limit_rule])
        
        # Auto-response rules
        self.auto_response_rules[ThreatLevel.HIGH] = self._block_ip_response
        self.auto_response_rules[ThreatLevel.CRITICAL] = self._emergency_response
    
    def _initialize_ml_detection(self) -> None:
        """Initialize ML-based anomaly detection."""
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            self.anomaly_detector = IsolationForest(
                contamination=0.05,  # 5% anomaly rate
                random_state=42,
                n_estimators=100
            )
            
            self.scaler = StandardScaler()
            self.logger.info("ML-based anomaly detection initialized")
            
        except ImportError:
            self.logger.warning("sklearn not available, disabling ML detection")
            self.enable_ml_detection = False
    
    def record_event(
        self,
        event_type: SecurityEventType,
        source_ip: str,
        description: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_assess: bool = True
    ) -> SecurityEvent:
        """Record security event."""
        
        with self._lock:
            # Create event
            event = SecurityEvent(
                event_type=event_type,
                threat_level=ThreatLevel.LOW,  # Will be updated by assessment
                timestamp=time.time(),
                source_ip=source_ip,
                user_id=user_id,
                description=description,
                metadata=metadata or {}
            )
            
            # Assess threat level if enabled
            if auto_assess:
                event.threat_level = self._assess_threat_level(event)
            
            # Store event
            self.security_events.append(event)
            
            # Update threat profile
            self._update_threat_profile(event)
            
            # Check security rules
            self._evaluate_security_rules(event)
            
            # Auto-response if needed
            if event.threat_level in self.auto_response_rules:
                try:
                    self.auto_response_rules[event.threat_level](event)
                except Exception as e:
                    self.logger.error(f"Auto-response failed: {e}")
            
            # Log event
            self.logger.warning(
                f"Security event: {event_type.value} from {source_ip} "
                f"(threat_level={event.threat_level.value})"
            )
            
            return event
    
    def _assess_threat_level(self, event: SecurityEvent) -> ThreatLevel:
        """Assess threat level of security event."""
        
        # Base threat levels by event type
        base_threats = {
            SecurityEventType.AUTHENTICATION_FAILURE: ThreatLevel.LOW,
            SecurityEventType.AUTHORIZATION_DENIED: ThreatLevel.LOW,
            SecurityEventType.SUSPICIOUS_ACTIVITY: ThreatLevel.MEDIUM,
            SecurityEventType.DATA_BREACH_ATTEMPT: ThreatLevel.HIGH,
            SecurityEventType.INJECTION_ATTEMPT: ThreatLevel.HIGH,
            SecurityEventType.RATE_LIMIT_EXCEEDED: ThreatLevel.MEDIUM,
            SecurityEventType.ANOMALOUS_BEHAVIOR: ThreatLevel.MEDIUM,
            SecurityEventType.PRIVILEGE_ESCALATION: ThreatLevel.CRITICAL,
            SecurityEventType.RESOURCE_EXHAUSTION: ThreatLevel.HIGH,
        }
        
        base_level = base_threats.get(event.event_type, ThreatLevel.LOW)
        
        # Escalate based on historical behavior
        threat_profile = self.threat_profiles.get(event.source_ip)
        if threat_profile and threat_profile.risk_score > 0.7:
            if base_level == ThreatLevel.LOW:
                base_level = ThreatLevel.MEDIUM
            elif base_level == ThreatLevel.MEDIUM:
                base_level = ThreatLevel.HIGH
        
        # Escalate based on frequency
        recent_events = [
            e for e in self.security_events
            if (e.source_ip == event.source_ip and
                time.time() - e.timestamp < 300)  # 5 minutes
        ]
        
        if len(recent_events) > 10:
            if base_level == ThreatLevel.LOW:
                base_level = ThreatLevel.MEDIUM
            elif base_level == ThreatLevel.MEDIUM:
                base_level = ThreatLevel.HIGH
        
        # ML-based assessment
        if self.enable_ml_detection:
            ml_risk = self._ml_assess_risk(event)
            if ml_risk > 0.8 and base_level != ThreatLevel.CRITICAL:
                # Escalate one level
                level_map = {
                    ThreatLevel.LOW: ThreatLevel.MEDIUM,
                    ThreatLevel.MEDIUM: ThreatLevel.HIGH,
                    ThreatLevel.HIGH: ThreatLevel.CRITICAL
                }
                base_level = level_map.get(base_level, base_level)
        
        return base_level
    
    def _ml_assess_risk(self, event: SecurityEvent) -> float:
        """Use ML to assess event risk."""
        
        if not self.enable_ml_detection or self.anomaly_detector is None:
            return 0.0
        
        try:
            # Extract features
            features = self._extract_event_features(event)
            
            # Check if we have enough data to make prediction
            if len(self.security_events) < 100:
                return 0.0
            
            # Get anomaly score
            anomaly_score = self.anomaly_detector.decision_function([features])[0]
            
            # Convert to risk probability (0-1)
            risk_score = max(0.0, min(1.0, (0.5 - anomaly_score) / 0.5))
            
            return risk_score
            
        except Exception as e:
            self.logger.debug(f"ML risk assessment failed: {e}")
            return 0.0
    
    def _extract_event_features(self, event: SecurityEvent) -> List[float]:
        """Extract features from security event for ML."""
        
        # Get threat profile
        threat_profile = self.threat_profiles.get(event.source_ip, ThreatProfile(
            source_ip=event.source_ip,
            first_seen=event.timestamp,
            last_seen=event.timestamp
        ))
        
        # Feature vector
        features = [
            # Event type (one-hot encoded simplified)
            float(event.event_type == SecurityEventType.AUTHENTICATION_FAILURE),
            float(event.event_type == SecurityEventType.INJECTION_ATTEMPT),
            float(event.event_type == SecurityEventType.DATA_BREACH_ATTEMPT),
            
            # Temporal features
            event.timestamp % 86400,  # Time of day
            event.timestamp % 604800,  # Day of week
            
            # Historical features
            threat_profile.event_count,
            len(threat_profile.event_types),
            threat_profile.risk_score,
            
            # Request patterns
            len(self.request_history.get(event.source_ip, [])),
            
            # String features (simplified)
            len(event.description),
            float('sql' in event.description.lower()),
            float('script' in event.description.lower()),
        ]
        
        return features
    
    def _update_threat_profile(self, event: SecurityEvent) -> None:
        """Update threat profile for source."""
        
        if event.source_ip not in self.threat_profiles:
            self.threat_profiles[event.source_ip] = ThreatProfile(
                source_ip=event.source_ip,
                first_seen=event.timestamp,
                last_seen=event.timestamp
            )
        
        profile = self.threat_profiles[event.source_ip]
        profile.last_seen = event.timestamp
        profile.event_count += 1
        profile.event_types.add(event.event_type)
        profile.update_risk_score()
        
        # Auto-block high-risk sources
        if profile.risk_score > 0.9 and not profile.is_blocked:
            self.block_ip(event.source_ip, f"High risk score: {profile.risk_score:.2f}")
    
    def _evaluate_security_rules(self, event: SecurityEvent) -> None:
        """Evaluate event against security rules."""
        
        for rule in self.security_rules:
            try:
                if rule(event):
                    self.logger.warning(f"Security rule triggered for event: {event.event_type.value}")
                    
                    # Escalate threat level
                    if event.threat_level == ThreatLevel.LOW:
                        event.threat_level = ThreatLevel.MEDIUM
                    elif event.threat_level == ThreatLevel.MEDIUM:
                        event.threat_level = ThreatLevel.HIGH
                    
            except Exception as e:
                self.logger.error(f"Security rule evaluation failed: {e}")
    
    def _block_ip_response(self, event: SecurityEvent) -> None:
        """Automatic IP blocking response."""
        self.block_ip(event.source_ip, f"Auto-blocked due to {event.event_type.value}")
    
    def _emergency_response(self, event: SecurityEvent) -> None:
        """Emergency response for critical threats."""
        # Block IP
        self.block_ip(event.source_ip, f"EMERGENCY: {event.event_type.value}")
        
        # Block user if available
        if event.user_id:
            self.block_user(event.user_id, f"EMERGENCY: {event.event_type.value}")
        
        # Send alert (in real implementation, this would notify administrators)
        self.logger.critical(
            f"EMERGENCY RESPONSE ACTIVATED: {event.event_type.value} "
            f"from {event.source_ip} (user: {event.user_id})"
        )
    
    def check_rate_limit(self, source_ip: str, endpoint: str = "default") -> bool:
        """Check if source is within rate limits."""
        
        with self._lock:
            current_time = time.time()
            key = f"{source_ip}:{endpoint}"
            
            # Clean old requests
            while (self.request_history[key] and
                   current_time - self.request_history[key][0] > self.rate_limit_window):
                self.request_history[key].popleft()
            
            # Check if over limit
            if len(self.request_history[key]) >= self.rate_limit_threshold:
                # Record rate limit event
                self.record_event(
                    event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
                    source_ip=source_ip,
                    description=f"Rate limit exceeded for {endpoint}",
                    metadata={
                        'endpoint': endpoint,
                        'request_count': len(self.request_history[key]),
                        'window_size': self.rate_limit_window
                    }
                )
                return False
            
            # Record request
            self.request_history[key].append(current_time)
            return True
    
    def block_ip(self, ip_address: str, reason: str = "") -> None:
        """Block IP address."""
        with self._lock:
            self.blocked_ips.add(ip_address)
            
            if ip_address in self.threat_profiles:
                self.threat_profiles[ip_address].is_blocked = True
                self.threat_profiles[ip_address].block_reason = reason
            
            self.logger.warning(f"Blocked IP {ip_address}: {reason}")
    
    def block_user(self, user_id: str, reason: str = "") -> None:
        """Block user."""
        with self._lock:
            self.blocked_users.add(user_id)
            self.logger.warning(f"Blocked user {user_id}: {reason}")
    
    def unblock_ip(self, ip_address: str) -> None:
        """Unblock IP address."""
        with self._lock:
            self.blocked_ips.discard(ip_address)
            
            if ip_address in self.threat_profiles:
                self.threat_profiles[ip_address].is_blocked = False
                self.threat_profiles[ip_address].block_reason = ""
            
            self.logger.info(f"Unblocked IP {ip_address}")
    
    def unblock_user(self, user_id: str) -> None:
        """Unblock user."""
        with self._lock:
            self.blocked_users.discard(user_id)
            self.logger.info(f"Unblocked user {user_id}")
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked."""
        return ip_address in self.blocked_ips
    
    def is_user_blocked(self, user_id: str) -> bool:
        """Check if user is blocked."""
        return user_id in self.blocked_users
    
    def validate_request(
        self,
        source_ip: str,
        user_id: Optional[str] = None,
        endpoint: str = "default"
    ) -> Tuple[bool, str]:
        """Validate incoming request."""
        
        # Check IP block
        if self.is_ip_blocked(source_ip):
            return False, f"IP {source_ip} is blocked"
        
        # Check user block
        if user_id and self.is_user_blocked(user_id):
            return False, f"User {user_id} is blocked"
        
        # Check rate limits
        if not self.check_rate_limit(source_ip, endpoint):
            return False, "Rate limit exceeded"
        
        return True, "Request allowed"
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics."""
        with self._lock:
            # Event counts by type
            event_counts = defaultdict(int)
            threat_counts = defaultdict(int)
            
            for event in self.security_events:
                event_counts[event.event_type.value] += 1
                threat_counts[event.threat_level.value] += 1
            
            # Risk distribution
            risk_scores = [p.risk_score for p in self.threat_profiles.values()]
            avg_risk = np.mean(risk_scores) if risk_scores else 0.0
            max_risk = max(risk_scores) if risk_scores else 0.0
            
            return {
                'total_events': len(self.security_events),
                'event_counts': dict(event_counts),
                'threat_counts': dict(threat_counts),
                'blocked_ips': len(self.blocked_ips),
                'blocked_users': len(self.blocked_users),
                'threat_profiles': len(self.threat_profiles),
                'average_risk_score': avg_risk,
                'max_risk_score': max_risk,
                'high_risk_sources': len([p for p in self.threat_profiles.values() if p.risk_score > 0.7]),
                'ml_detection_enabled': self.enable_ml_detection,
                'active_security_rules': len(self.security_rules)
            }
    
    def get_threat_intelligence(self) -> Dict[str, Any]:
        """Get threat intelligence summary."""
        with self._lock:
            # Top threat sources
            top_threats = sorted(
                self.threat_profiles.values(),
                key=lambda p: p.risk_score,
                reverse=True
            )[:10]
            
            # Recent critical events
            recent_critical = [
                event.to_dict() for event in self.security_events
                if event.threat_level == ThreatLevel.CRITICAL and
                   time.time() - event.timestamp < 3600  # Last hour
            ]
            
            # Attack patterns
            attack_patterns = defaultdict(int)
            for event in self.security_events:
                if event.event_type in [SecurityEventType.INJECTION_ATTEMPT, 
                                      SecurityEventType.DATA_BREACH_ATTEMPT]:
                    attack_patterns[event.event_type.value] += 1
            
            return {
                'top_threat_sources': [
                    {
                        'ip': p.source_ip,
                        'risk_score': p.risk_score,
                        'event_count': p.event_count,
                        'first_seen': p.first_seen,
                        'is_blocked': p.is_blocked
                    } for p in top_threats
                ],
                'recent_critical_events': recent_critical,
                'attack_patterns': dict(attack_patterns),
                'security_posture': self._assess_security_posture()
            }
    
    def _assess_security_posture(self) -> Dict[str, Any]:
        """Assess overall security posture."""
        metrics = self.get_security_metrics()
        
        # Calculate posture score (0-100)
        posture_score = 100
        
        # Deduct for high-risk sources
        posture_score -= min(50, metrics['high_risk_sources'] * 5)
        
        # Deduct for recent critical events
        critical_events = metrics['threat_counts'].get('critical', 0)
        posture_score -= min(30, critical_events * 10)
        
        # Deduct for high average risk
        posture_score -= min(20, metrics['average_risk_score'] * 20)
        
        posture_score = max(0, posture_score)
        
        # Determine posture level
        if posture_score >= 80:
            posture_level = "strong"
        elif posture_score >= 60:
            posture_level = "moderate"
        elif posture_score >= 40:
            posture_level = "weak"
        else:
            posture_level = "critical"
        
        return {
            'score': posture_score,
            'level': posture_level,
            'recommendations': self._generate_security_recommendations(metrics, posture_score)
        }
    
    def _generate_security_recommendations(
        self, 
        metrics: Dict[str, Any], 
        posture_score: int
    ) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        if metrics['high_risk_sources'] > 5:
            recommendations.append("High number of risky sources - consider stricter blocking policies")
        
        if metrics['threat_counts'].get('critical', 0) > 0:
            recommendations.append("Critical threats detected - investigate and respond immediately")
        
        if not self.enable_ml_detection:
            recommendations.append("Enable ML-based threat detection for improved accuracy")
        
        if len(self.security_rules) < 5:
            recommendations.append("Consider adding more security rules for comprehensive protection")
        
        if posture_score < 60:
            recommendations.append("Security posture is weak - immediate security review required")
        
        return recommendations
    
    def add_security_rule(self, rule: Callable[[SecurityEvent], bool]) -> None:
        """Add custom security rule."""
        with self._lock:
            self.security_rules.append(rule)
            self.logger.info(f"Added security rule (total: {len(self.security_rules)})")
    
    def remove_security_rule(self, rule: Callable[[SecurityEvent], bool]) -> None:
        """Remove security rule."""
        with self._lock:
            if rule in self.security_rules:
                self.security_rules.remove(rule)
                self.logger.info(f"Removed security rule (total: {len(self.security_rules)})")


def security_monitor(monitor: AdvancedSecurityMonitor):
    """Decorator for monitoring function calls."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract request info (simplified)
            source_ip = kwargs.get('source_ip', '127.0.0.1')
            user_id = kwargs.get('user_id')
            
            # Validate request
            is_valid, reason = monitor.validate_request(source_ip, user_id, func.__name__)
            
            if not is_valid:
                monitor.record_event(
                    SecurityEventType.AUTHORIZATION_DENIED,
                    source_ip,
                    f"Request blocked: {reason}",
                    user_id
                )
                raise SecurityError(reason)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Monitor for potential security issues
                if any(keyword in str(e).lower() for keyword in ['injection', 'breach', 'exploit']):
                    monitor.record_event(
                        SecurityEventType.SUSPICIOUS_ACTIVITY,
                        source_ip,
                        f"Potential security issue in {func.__name__}: {str(e)}",
                        user_id
                    )
                raise
        
        return wrapper
    
    return decorator


class SecurityError(Exception):
    """Security-related error."""
    pass


# Global security monitor instance
_security_monitor = AdvancedSecurityMonitor()


def get_security_monitor() -> AdvancedSecurityMonitor:
    """Get global security monitor instance."""
    return _security_monitor