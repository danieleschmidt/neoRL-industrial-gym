"""Security utilities for industrial RL applications."""

import hashlib
import hmac
import numpy as np
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .monitoring.logger import get_logger


class SecurityManager:
    """Manages security for industrial RL systems."""
    
    def __init__(self, secret_key: Optional[str] = None):
        """Initialize security manager.
        
        Args:
            secret_key: Secret key for HMAC operations
        """
        self.secret_key = secret_key or "default-neorl-industrial-key"
        self.logger = get_logger("security")
        
        # Track security events
        self.security_events = []
        
    def validate_input_array(
        self,
        array: np.ndarray,
        expected_shape: Optional[tuple] = None,
        expected_dtype: Optional[np.dtype] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_nan: bool = False,
        allow_inf: bool = False,
    ) -> bool:
        """Validate input array for security and safety.
        
        Args:
            array: Input array to validate
            expected_shape: Expected shape (None to skip check)
            expected_dtype: Expected dtype (None to skip check)
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_nan: Whether to allow NaN values
            allow_inf: Whether to allow infinite values
            
        Returns:
            True if array is valid
            
        Raises:
            SecurityError: If array fails security validation
        """
        try:
            # Basic type check
            if not isinstance(array, np.ndarray):
                raise SecurityError(f"Input must be numpy array, got {type(array)}")
                
            # Shape validation
            if expected_shape is not None and array.shape != expected_shape:
                raise SecurityError(f"Invalid shape: expected {expected_shape}, got {array.shape}")
                
            # Dtype validation
            if expected_dtype is not None and array.dtype != expected_dtype:
                raise SecurityError(f"Invalid dtype: expected {expected_dtype}, got {array.dtype}")
                
            # Check for NaN values
            if not allow_nan and np.any(np.isnan(array)):
                raise SecurityError("Array contains NaN values")
                
            # Check for infinite values
            if not allow_inf and np.any(np.isinf(array)):
                raise SecurityError("Array contains infinite values")
                
            # Value range validation
            if min_value is not None and np.any(array < min_value):
                raise SecurityError(f"Array contains values below minimum {min_value}")
                
            if max_value is not None and np.any(array > max_value):
                raise SecurityError(f"Array contains values above maximum {max_value}")
                
            # Check for unreasonably large arrays (potential DoS)
            if array.size > 1e8:  # 100M elements
                self.logger.warning(f"Very large array detected: {array.size} elements")
                
            return True
            
        except SecurityError:
            self._log_security_event("INPUT_VALIDATION_FAILED", {
                "array_shape": str(array.shape) if isinstance(array, np.ndarray) else "unknown",
                "array_dtype": str(array.dtype) if isinstance(array, np.ndarray) else "unknown",
            })
            raise
            
    def sanitize_string(self, input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input to prevent injection attacks.
        
        Args:
            input_str: Input string to sanitize
            max_length: Maximum allowed string length
            
        Returns:
            Sanitized string
            
        Raises:
            SecurityError: If string is malicious
        """
        if not isinstance(input_str, str):
            raise SecurityError(f"Input must be string, got {type(input_str)}")
            
        # Length check
        if len(input_str) > max_length:
            raise SecurityError(f"String too long: {len(input_str)} > {max_length}")
            
        # Check for potential code injection patterns
        dangerous_patterns = [
            r"<script.*?>.*?</script>",  # Script tags
            r"javascript:",              # JavaScript URLs
            r"on\w+\s*=",               # Event handlers
            r"eval\s*\(",               # eval() calls
            r"exec\s*\(",               # exec() calls
            r"__import__",              # Python imports
            r"subprocess",              # Subprocess calls
            r"os\.",                    # OS module calls
            r"sys\.",                   # Sys module calls
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                self._log_security_event("INJECTION_ATTEMPT", {
                    "pattern": pattern,
                    "input_preview": input_str[:100],
                })
                raise SecurityError(f"Potentially malicious pattern detected: {pattern}")
                
        # Remove/escape special characters
        sanitized = re.sub(r'[<>&"\']', '', input_str)
        sanitized = sanitized.strip()
        
        return sanitized
        
    def validate_file_path(self, file_path: Union[str, Path], allowed_dirs: List[str]) -> Path:
        """Validate file path for security.
        
        Args:
            file_path: File path to validate
            allowed_dirs: List of allowed directory prefixes
            
        Returns:
            Validated Path object
            
        Raises:
            SecurityError: If path is not allowed
        """
        path = Path(file_path).resolve()
        
        # Check for path traversal attempts
        if ".." in str(file_path):
            raise SecurityError("Path traversal detected")
            
        # Check if path is within allowed directories
        allowed = False
        for allowed_dir in allowed_dirs:
            allowed_path = Path(allowed_dir).resolve()
            try:
                path.relative_to(allowed_path)
                allowed = True
                break
            except ValueError:
                continue
                
        if not allowed:
            self._log_security_event("UNAUTHORIZED_PATH_ACCESS", {
                "requested_path": str(path),
                "allowed_dirs": allowed_dirs,
            })
            raise SecurityError(f"Path not in allowed directories: {path}")
            
        return path
        
    def hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data for logging/storage.
        
        Args:
            data: Sensitive data to hash
            
        Returns:
            SHA-256 hash of the data
        """
        return hashlib.sha256(data.encode()).hexdigest()
        
    def create_hmac(self, message: str) -> str:
        """Create HMAC for message integrity.
        
        Args:
            message: Message to create HMAC for
            
        Returns:
            HMAC digest
        """
        return hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
    def verify_hmac(self, message: str, signature: str) -> bool:
        """Verify HMAC signature.
        
        Args:
            message: Original message
            signature: HMAC signature to verify
            
        Returns:
            True if signature is valid
        """
        expected = self.create_hmac(message)
        return hmac.compare_digest(expected, signature)
        
    def validate_hyperparameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate hyperparameters for safety.
        
        Args:
            params: Hyperparameter dictionary
            
        Returns:
            Validated parameters
            
        Raises:
            SecurityError: If parameters are unsafe
        """
        validated = {}
        
        for key, value in params.items():
            # Sanitize parameter names
            safe_key = self.sanitize_string(key, max_length=100)
            
            # Validate specific parameters
            if "learning_rate" in key.lower():
                if not isinstance(value, (int, float)) or value <= 0 or value > 1:
                    raise SecurityError(f"Invalid learning rate: {value}")
                    
            elif "batch_size" in key.lower():
                if not isinstance(value, int) or value <= 0 or value > 10000:
                    raise SecurityError(f"Invalid batch size: {value}")
                    
            elif "epochs" in key.lower():
                if not isinstance(value, int) or value <= 0 or value > 100000:
                    raise SecurityError(f"Invalid epoch count: {value}")
                    
            # Check for extremely large numeric values
            if isinstance(value, (int, float)):
                if abs(value) > 1e10:
                    self.logger.warning(f"Very large parameter value: {key}={value}")
                    
            validated[safe_key] = value
            
        return validated
        
    def _log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log a security event."""
        event = {
            "event_type": event_type,
            "details": details,
            "timestamp": __import__("time").time(),
        }
        
        self.security_events.append(event)
        
        # Log to security logger
        self.logger.safety_event(
            event_type=event_type,
            severity="HIGH",
            details=details,
        )
        
    def get_security_summary(self) -> Dict[str, Any]:
        """Get summary of security events."""
        event_counts = {}
        for event in self.security_events:
            event_type = event["event_type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
        return {
            "total_events": len(self.security_events),
            "event_types": event_counts,
            "recent_events": self.security_events[-10:] if self.security_events else [],
        }


class SecurityError(Exception):
    """Raised for security-related errors."""
    pass


# Global security manager
_security_manager = None


def get_security_manager(secret_key: Optional[str] = None) -> SecurityManager:
    """Get global security manager instance.
    
    Args:
        secret_key: Secret key for HMAC operations
        
    Returns:
        SecurityManager instance
    """
    global _security_manager
    
    if _security_manager is None:
        _security_manager = SecurityManager(secret_key)
        
    return _security_manager


def secure_validate_array(
    array: np.ndarray,
    **kwargs
) -> bool:
    """Convenience function for array validation.
    
    Args:
        array: Array to validate
        **kwargs: Arguments for validate_input_array
        
    Returns:
        True if array is valid
    """
    return get_security_manager().validate_input_array(array, **kwargs)