"""Enhanced security framework for industrial RL systems."""

import hashlib
import hmac
import secrets
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
from pathlib import Path
import base64

import numpy as np
import jax.numpy as jnp

class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    """Security event record."""
    event_type: str
    threat_level: ThreatLevel
    timestamp: float
    source: str
    details: Dict[str, Any]
    mitigation_applied: bool = False
    event_id: str = field(default_factory=lambda: secrets.token_hex(8))

class SecurityValidator:
    """Validates inputs and data for security threats."""
    
    def __init__(self):
        self.logger = logging.getLogger("security_validator")
        
        # Known malicious patterns
        self.malicious_patterns = [
            b"__import__",
            b"eval(",
            b"exec(",
            b"subprocess",
            b"os.system",
            b"pickle.loads",
            b"yaml.unsafe_load",
        ]
        
        # Input size limits
        self.max_input_size = 10 * 1024 * 1024  # 10MB
        self.max_array_size = 100 * 1024 * 1024  # 100MB
        
    def validate_input_data(
        self, 
        data: Any, 
        expected_type: Optional[type] = None,
        max_size: Optional[int] = None
    ) -> bool:
        """Validate input data for security threats."""
        
        try:
            # Type validation
            if expected_type and not isinstance(data, expected_type):
                self.logger.warning(f"Type mismatch: expected {expected_type}, got {type(data)}")
                return False
            
            # Size validation
            size_limit = max_size or self.max_input_size
            data_size = self._estimate_size(data)
            
            if data_size > size_limit:
                self.logger.warning(f"Input too large: {data_size} bytes > {size_limit}")
                return False
            
            # Content validation
            if isinstance(data, (str, bytes)):
                return self._validate_content(data)
            elif isinstance(data, dict):
                return self._validate_dict(data)
            elif isinstance(data, (list, tuple)):
                return self._validate_sequence(data)
            elif isinstance(data, (np.ndarray, jnp.ndarray)):
                return self._validate_array(data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Input validation failed: {e}")
            return False
    
    def _validate_content(self, content: Union[str, bytes]) -> bool:
        """Validate string/bytes content for malicious patterns."""
        if isinstance(content, str):
            content = content.encode('utf-8')
        
        # Check for malicious patterns
        for pattern in self.malicious_patterns:
            if pattern in content:
                self.logger.warning(f"Malicious pattern detected: {pattern}")
                return False
        
        return True
    
    def _validate_dict(self, data: Dict[str, Any]) -> bool:
        """Validate dictionary data."""
        # Check for dangerous keys
        dangerous_keys = ["__class__", "__dict__", "__globals__", "func_globals"]
        
        for key in data.keys():
            if key in dangerous_keys:
                self.logger.warning(f"Dangerous key detected: {key}")
                return False
            
            # Recursively validate values
            if not self.validate_input_data(data[key]):
                return False
        
        return True
    
    def _validate_sequence(self, data: Union[List, tuple]) -> bool:
        """Validate sequence data."""
        for item in data:
            if not self.validate_input_data(item):
                return False
        return True
    
    def _validate_array(self, array: Union[np.ndarray, jnp.ndarray]) -> bool:
        """Validate numpy/JAX arrays."""
        # Check for NaN/Inf values that could cause issues
        if np.any(np.isnan(array)) or np.any(np.isinf(array)):
            self.logger.warning("Array contains NaN or Inf values")
            return False
        
        # Check array size
        if array.nbytes > self.max_array_size:
            self.logger.warning(f"Array too large: {array.nbytes} bytes")
            return False
        
        # Check for suspicious values
        if np.any(np.abs(array) > 1e10):
            self.logger.warning("Array contains extremely large values")
            return False
        
        return True
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data."""
        if isinstance(data, (str, bytes)):
            return len(data)
        elif isinstance(data, (np.ndarray, jnp.ndarray)):
            return data.nbytes
        elif isinstance(data, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in data.items())
        elif isinstance(data, (list, tuple)):
            return sum(self._estimate_size(item) for item in data)
        else:
            return 64  # Default estimate for other types

class DataEncryption:
    """Handles encryption/decryption of sensitive data."""
    
    def __init__(self, key: Optional[bytes] = None):
        """Initialize with encryption key."""
        self.key = key or self._generate_key()
        self.logger = logging.getLogger("data_encryption")
    
    def _generate_key(self) -> bytes:
        """Generate a secure encryption key."""
        return secrets.token_bytes(32)  # 256-bit key
    
    def encrypt_data(self, data: Any) -> Dict[str, Any]:
        """Encrypt data using simple XOR (for demo - use proper crypto in production)."""
        try:
            # Serialize data
            if isinstance(data, (np.ndarray, jnp.ndarray)):
                serialized = data.tobytes()
                data_type = "array"
                shape = data.shape
                dtype = str(data.dtype)
            else:
                serialized = json.dumps(data, default=str).encode('utf-8')
                data_type = "json"
                shape = None
                dtype = None
            
            # Simple XOR encryption (demo only - use AES in production)
            encrypted = self._xor_encrypt(serialized)
            
            return {
                "encrypted_data": base64.b64encode(encrypted).decode('ascii'),
                "data_type": data_type,
                "shape": shape,
                "dtype": dtype,
                "timestamp": time.time(),
            }
            
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, encrypted_dict: Dict[str, Any]) -> Any:
        """Decrypt data."""
        try:
            encrypted_bytes = base64.b64decode(encrypted_dict["encrypted_data"])
            decrypted = self._xor_decrypt(encrypted_bytes)
            
            if encrypted_dict["data_type"] == "array":
                # Reconstruct array
                array = np.frombuffer(decrypted, dtype=encrypted_dict["dtype"])
                return array.reshape(encrypted_dict["shape"])
            else:
                # Reconstruct JSON
                return json.loads(decrypted.decode('utf-8'))
                
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise
    
    def _xor_encrypt(self, data: bytes) -> bytes:
        """Simple XOR encryption (demo only)."""
        key_repeated = (self.key * ((len(data) // len(self.key)) + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, key_repeated))
    
    def _xor_decrypt(self, data: bytes) -> bytes:
        """Simple XOR decryption (demo only)."""
        return self._xor_encrypt(data)  # XOR is symmetric

class AuditLogger:
    """Secure audit logging for compliance and forensics."""
    
    def __init__(self, log_file: Optional[Path] = None):
        """Initialize audit logger."""
        self.log_file = log_file or Path("security_audit.log")
        self.logger = logging.getLogger("audit_logger")
        
        # Create separate file handler for audit logs
        self._setup_audit_handler()
        
        # Event counters
        self.event_counters = defaultdict(int)
        self.session_id = secrets.token_hex(8)
        
        # Integrity protection
        self.log_hash_chain = []
        
    def _setup_audit_handler(self):
        """Setup dedicated audit log handler."""
        handler = logging.FileHandler(self.log_file, mode='a')
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S UTC'
        )
        handler.setFormatter(formatter)
        
        audit_logger = logging.getLogger("security_audit")
        audit_logger.addHandler(handler)
        audit_logger.setLevel(logging.INFO)
        
        self.audit_logger = audit_logger
    
    def log_event(
        self, 
        event_type: str, 
        details: Dict[str, Any],
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None
    ) -> str:
        """Log a security event."""
        
        event_id = secrets.token_hex(8)
        timestamp = time.time()
        
        # Create audit record
        audit_record = {
            "event_id": event_id,
            "session_id": self.session_id,
            "timestamp": timestamp,
            "event_type": event_type,
            "user_id": user_id,
            "source_ip": source_ip,
            "details": details,
        }
        
        # Add integrity hash
        record_hash = self._calculate_record_hash(audit_record)
        audit_record["integrity_hash"] = record_hash
        
        # Update hash chain
        prev_hash = self.log_hash_chain[-1] if self.log_hash_chain else "0"
        chain_hash = hashlib.sha256(f"{prev_hash}{record_hash}".encode()).hexdigest()
        self.log_hash_chain.append(chain_hash)
        audit_record["chain_hash"] = chain_hash
        
        # Log the event
        log_message = json.dumps(audit_record, default=str)
        self.audit_logger.info(log_message)
        
        # Update counters
        self.event_counters[event_type] += 1
        
        return event_id
    
    def _calculate_record_hash(self, record: Dict[str, Any]) -> str:
        """Calculate integrity hash for audit record."""
        # Remove any existing hash fields
        clean_record = {k: v for k, v in record.items() 
                      if k not in ["integrity_hash", "chain_hash"]}
        
        record_str = json.dumps(clean_record, sort_keys=True, default=str)
        return hashlib.sha256(record_str.encode()).hexdigest()
    
    def verify_log_integrity(self, log_file: Optional[Path] = None) -> bool:
        """Verify the integrity of audit logs."""
        target_file = log_file or self.log_file
        
        if not target_file.exists():
            return True  # Empty log is valid
        
        try:
            with open(target_file, 'r') as f:
                lines = f.readlines()
            
            prev_chain_hash = "0"
            
            for line in lines:
                if not line.strip():
                    continue
                
                try:
                    # Parse log entry (extract JSON from log format)
                    json_part = line.split(' | ')[-1].strip()
                    record = json.loads(json_part)
                    
                    # Verify record hash
                    stored_hash = record.get("integrity_hash")
                    calculated_hash = self._calculate_record_hash(record)
                    
                    if stored_hash != calculated_hash:
                        self.logger.error(f"Record integrity violation: {record.get('event_id')}")
                        return False
                    
                    # Verify chain hash
                    stored_chain_hash = record.get("chain_hash")
                    calculated_chain_hash = hashlib.sha256(
                        f"{prev_chain_hash}{stored_hash}".encode()
                    ).hexdigest()
                    
                    if stored_chain_hash != calculated_chain_hash:
                        self.logger.error(f"Chain integrity violation: {record.get('event_id')}")
                        return False
                    
                    prev_chain_hash = stored_chain_hash
                    
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.warning(f"Malformed log entry: {e}")
                    continue
            
            return True
            
        except Exception as e:
            self.logger.error(f"Log verification failed: {e}")
            return False
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get summary of audit events."""
        return {
            "session_id": self.session_id,
            "total_events": sum(self.event_counters.values()),
            "event_types": dict(self.event_counters),
            "log_file": str(self.log_file),
            "chain_length": len(self.log_hash_chain),
        }

class SecurityManager:
    """Central security management system."""
    
    def __init__(self):
        """Initialize security manager."""
        self.validator = SecurityValidator()
        self.encryptor = DataEncryption()
        self.auditor = AuditLogger()
        
        self.security_events = deque(maxlen=1000)
        self.threat_counters = defaultdict(int)
        
        # Security policies
        self.policies = {
            "max_failed_validations": 10,
            "alert_threshold_minutes": 5,
            "auto_block_threshold": 5,
        }
        
        self.logger = logging.getLogger("security_manager")
        self._lock = threading.Lock()
        
        # Track blocked sources
        self.blocked_sources = set()
        self.failed_validations = defaultdict(int)
        
    def validate_and_log(
        self, 
        data: Any, 
        operation: str,
        source: str = "unknown",
        user_id: Optional[str] = None
    ) -> bool:
        """Validate data and log security events."""
        
        with self._lock:
            # Check if source is blocked
            if source in self.blocked_sources:
                self._record_security_event(
                    "blocked_access_attempt",
                    ThreatLevel.HIGH,
                    source,
                    {"operation": operation, "reason": "source_blocked"}
                )
                return False
            
            # Validate data
            is_valid = self.validator.validate_input_data(data)
            
            if not is_valid:
                self.failed_validations[source] += 1
                
                # Log validation failure
                event_details = {
                    "operation": operation,
                    "data_type": type(data).__name__,
                    "failure_count": self.failed_validations[source],
                }
                
                self._record_security_event(
                    "validation_failure",
                    ThreatLevel.MEDIUM,
                    source,
                    event_details
                )
                
                # Check if we should block this source
                if self.failed_validations[source] >= self.policies["auto_block_threshold"]:
                    self.blocked_sources.add(source)
                    
                    self._record_security_event(
                        "source_blocked",
                        ThreatLevel.HIGH,
                        source,
                        {"reason": "too_many_failures", "total_failures": self.failed_validations[source]}
                    )
                
                return False
            
            else:
                # Log successful validation
                self.auditor.log_event(
                    "data_validation_success",
                    {
                        "operation": operation,
                        "data_type": type(data).__name__,
                        "source": source,
                    },
                    user_id=user_id
                )
                
                # Reset failure count on success
                self.failed_validations[source] = 0
                
                return True
    
    def _record_security_event(
        self, 
        event_type: str, 
        threat_level: ThreatLevel,
        source: str, 
        details: Dict[str, Any]
    ) -> None:
        """Record a security event."""
        
        event = SecurityEvent(
            event_type=event_type,
            threat_level=threat_level,
            timestamp=time.time(),
            source=source,
            details=details
        )
        
        self.security_events.append(event)
        self.threat_counters[threat_level] += 1
        
        # Log to audit system
        self.auditor.log_event(
            event_type,
            {
                "threat_level": threat_level.value,
                "source": source,
                "details": details,
                "event_id": event.event_id,
            }
        )
        
        # Log based on threat level
        if threat_level == ThreatLevel.CRITICAL:
            self.logger.critical(f"CRITICAL SECURITY EVENT: {event_type} from {source}")
        elif threat_level == ThreatLevel.HIGH:
            self.logger.error(f"HIGH THREAT: {event_type} from {source}")
        elif threat_level == ThreatLevel.MEDIUM:
            self.logger.warning(f"MEDIUM THREAT: {event_type} from {source}")
        else:
            self.logger.info(f"LOW THREAT: {event_type} from {source}")
    
    def encrypt_sensitive_data(self, data: Any) -> Dict[str, Any]:
        """Encrypt sensitive data."""
        try:
            encrypted = self.encryptor.encrypt_data(data)
            
            self.auditor.log_event(
                "data_encryption",
                {"data_type": type(data).__name__}
            )
            
            return encrypted
            
        except Exception as e:
            self._record_security_event(
                "encryption_failure",
                ThreatLevel.HIGH,
                "system",
                {"error": str(e)}
            )
            raise
    
    def decrypt_sensitive_data(self, encrypted_dict: Dict[str, Any]) -> Any:
        """Decrypt sensitive data."""
        try:
            decrypted = self.encryptor.decrypt_data(encrypted_dict)
            
            self.auditor.log_event(
                "data_decryption",
                {"data_type": encrypted_dict.get("data_type", "unknown")}
            )
            
            return decrypted
            
        except Exception as e:
            self._record_security_event(
                "decryption_failure",
                ThreatLevel.HIGH,
                "system",
                {"error": str(e)}
            )
            raise
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        with self._lock:
            recent_events = [
                event for event in self.security_events 
                if time.time() - event.timestamp < 3600  # Last hour
            ]
            
            return {
                "total_events": len(self.security_events),
                "recent_events": len(recent_events),
                "threat_levels": dict(self.threat_counters),
                "blocked_sources": len(self.blocked_sources),
                "failed_validations": dict(self.failed_validations),
                "policies": self.policies,
                "audit_summary": self.auditor.get_audit_summary(),
            }
    
    def unblock_source(self, source: str) -> bool:
        """Manually unblock a source."""
        with self._lock:
            if source in self.blocked_sources:
                self.blocked_sources.remove(source)
                self.failed_validations[source] = 0
                
                self._record_security_event(
                    "source_unblocked",
                    ThreatLevel.LOW,
                    source,
                    {"reason": "manual_unblock"}
                )
                
                return True
            return False
    
    def reset_security_state(self) -> None:
        """Reset security state (for testing/maintenance)."""
        with self._lock:
            self.blocked_sources.clear()
            self.failed_validations.clear()
            self.security_events.clear()
            self.threat_counters.clear()
            
            self.auditor.log_event(
                "security_state_reset",
                {"reason": "manual_reset"}
            )

# Global security manager instance
_security_manager = None

def get_security_manager() -> SecurityManager:
    """Get global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager

def secure_validate(data: Any, operation: str, source: str = "unknown") -> bool:
    """Convenience function for secure validation."""
    return get_security_manager().validate_and_log(data, operation, source)

def secure_encrypt(data: Any) -> Dict[str, Any]:
    """Convenience function for encryption."""
    return get_security_manager().encrypt_sensitive_data(data)

def secure_decrypt(encrypted_dict: Dict[str, Any]) -> Any:
    """Convenience function for decryption."""
    return get_security_manager().decrypt_sensitive_data(encrypted_dict)