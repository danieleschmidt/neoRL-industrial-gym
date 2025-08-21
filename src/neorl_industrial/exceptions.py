"""Custom exceptions for neoRL-industrial."""


class NeoRLIndustrialException(Exception):
    """Base exception for neoRL-industrial."""
    pass


class SafetyViolationError(NeoRLIndustrialException):
    """Raised when safety constraints are critically violated."""
    
    def __init__(self, message: str, constraint_name: str = None, violation_count: int = 0):
        super().__init__(message)
        self.constraint_name = constraint_name
        self.violation_count = violation_count


class EmergencyShutdownError(NeoRLIndustrialException):
    """Raised when emergency shutdown is triggered."""
    
    def __init__(self, message: str, trigger_reason: str = None):
        super().__init__(message)
        self.trigger_reason = trigger_reason


class AgentNotTrainedError(NeoRLIndustrialException):
    """Raised when trying to use an untrained agent."""
    pass


class ConfigurationError(NeoRLIndustrialException):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: str = None):
        super().__init__(message)
        self.config_key = config_key


class DatasetError(NeoRLIndustrialException):
    """Raised when dataset operations fail."""
    
    def __init__(self, message: str, dataset_name: str = None):
        super().__init__(message)
        self.dataset_name = dataset_name


class EnvironmentError(NeoRLIndustrialException):
    """Raised when environment operations fail."""
    
    def __init__(self, message: str, env_name: str = None):
        super().__init__(message)
        self.env_name = env_name


class TrainingError(NeoRLIndustrialException):
    """Raised when training fails."""
    
    def __init__(self, message: str, agent_type: str = None, epoch: int = None):
        super().__init__(message)
        self.agent_type = agent_type
        self.epoch = epoch


class NetworkError(NeoRLIndustrialException):
    """Raised when neural network operations fail."""
    
    def __init__(self, message: str, network_name: str = None):
        super().__init__(message)
        self.network_name = network_name


class ValidationError(NeoRLIndustrialException):
    """Raised when validation fails."""
    
    def __init__(self, message: str, validation_type: str = None):
        super().__init__(message)
        self.validation_type = validation_type


class RecoveryError(NeoRLIndustrialException):
    """Raised when error recovery fails."""
    pass