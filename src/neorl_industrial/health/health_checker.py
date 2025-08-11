"""Comprehensive health checking system for industrial RL environments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

from ..core.types import SafetyConstraint
from ..exceptions import HealthCheckError, SystemError
from ..security import SecurityManager


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class SystemHealth:
    """System health status container."""
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: float
    component: str
    
    @property
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        return self.status == HealthStatus.HEALTHY
    
    @property
    def requires_attention(self) -> bool:
        """Check if system requires attention."""
        return self.status in (HealthStatus.WARNING, HealthStatus.CRITICAL)


class HealthChecker:
    """Comprehensive system health checker for industrial RL systems."""
    
    def __init__(
        self,
        check_dependencies: bool = True,
        check_security: bool = True,
        check_storage: bool = True,
        check_memory: bool = True,
        security_manager: Optional[SecurityManager] = None,
    ) -> None:
        """Initialize health checker.
        
        Args:
            check_dependencies: Check external dependencies
            check_security: Perform security health checks
            check_storage: Check storage and file system
            check_memory: Check memory usage
            security_manager: Optional security manager instance
        """
        self.check_dependencies = check_dependencies
        self.check_security = check_security
        self.check_storage = check_storage
        self.check_memory = check_memory
        self.security_manager = security_manager or SecurityManager()
        
        self._health_history: List[SystemHealth] = []
    
    def check_all(self) -> Dict[str, SystemHealth]:
        """Perform comprehensive health check.
        
        Returns:
            Dictionary of health check results by component
        """
        checks = {}
        
        # Core dependency checks
        if self.check_dependencies:
            checks["jax"] = self._check_jax_health()
            checks["mlflow"] = self._check_mlflow_health()
            checks["python_env"] = self._check_python_environment()
        
        # Security checks
        if self.check_security:
            checks["security"] = self._check_security_health()
            checks["permissions"] = self._check_file_permissions()
        
        # System resource checks
        if self.check_storage:
            checks["storage"] = self._check_storage_health()
            checks["datasets"] = self._check_dataset_availability()
            
        if self.check_memory:
            checks["memory"] = self._check_memory_health()
            checks["gpu_memory"] = self._check_gpu_memory()
        
        # Industrial-specific checks
        checks["safety_systems"] = self._check_safety_systems()
        checks["environment_registry"] = self._check_environment_health()
        checks["agent_registry"] = self._check_agent_health()
        
        # Store health history
        for health in checks.values():
            self._health_history.append(health)
            
        # Keep only last 100 entries
        if len(self._health_history) > 100:
            self._health_history = self._health_history[-100:]
        
        return checks
    
    def _check_jax_health(self) -> SystemHealth:
        """Check JAX installation and functionality."""
        if not JAX_AVAILABLE:
            return SystemHealth(
                status=HealthStatus.CRITICAL,
                message="JAX not available - required for agent training",
                details={"jax_installed": False},
                timestamp=time.time(),
                component="jax"
            )
        
        try:
            # Test basic JAX operations
            test_array = jnp.array([1.0, 2.0, 3.0])
            result = jnp.sum(test_array)
            
            # Check available devices
            devices = jax.devices()
            device_info = {
                "devices": [str(device) for device in devices],
                "device_count": len(devices),
                "default_backend": jax.default_backend(),
            }
            
            return SystemHealth(
                status=HealthStatus.HEALTHY,
                message=f"JAX working correctly with {len(devices)} device(s)",
                details={
                    "jax_installed": True,
                    "test_result": float(result),
                    **device_info
                },
                timestamp=time.time(),
                component="jax"
            )
            
        except Exception as e:
            return SystemHealth(
                status=HealthStatus.CRITICAL,
                message=f"JAX functionality test failed: {e}",
                details={"jax_installed": True, "error": str(e)},
                timestamp=time.time(),
                component="jax"
            )
    
    def _check_mlflow_health(self) -> SystemHealth:
        """Check MLflow availability and tracking."""
        if not MLFLOW_AVAILABLE:
            return SystemHealth(
                status=HealthStatus.WARNING,
                message="MLflow not available - experiment tracking disabled",
                details={"mlflow_installed": False},
                timestamp=time.time(),
                component="mlflow"
            )
        
        try:
            # Test MLflow functionality
            tracking_uri = mlflow.get_tracking_uri()
            current_experiment = mlflow.get_experiment_by_name("Default")
            
            return SystemHealth(
                status=HealthStatus.HEALTHY,
                message="MLflow working correctly",
                details={
                    "mlflow_installed": True,
                    "tracking_uri": tracking_uri,
                    "default_experiment_exists": current_experiment is not None,
                },
                timestamp=time.time(),
                component="mlflow"
            )
            
        except Exception as e:
            return SystemHealth(
                status=HealthStatus.WARNING,
                message=f"MLflow functionality test failed: {e}",
                details={"mlflow_installed": True, "error": str(e)},
                timestamp=time.time(),
                component="mlflow"
            )
    
    def _check_python_environment(self) -> SystemHealth:
        """Check Python environment health."""
        import sys
        import platform
        
        try:
            python_version = sys.version
            platform_info = platform.platform()
            
            # Check Python version compatibility
            version_info = sys.version_info
            if version_info < (3, 8):
                status = HealthStatus.CRITICAL
                message = f"Python {version_info.major}.{version_info.minor} not supported"
            elif version_info >= (3, 12):
                status = HealthStatus.WARNING
                message = f"Python {version_info.major}.{version_info.minor} not fully tested"
            else:
                status = HealthStatus.HEALTHY
                message = "Python environment compatible"
            
            return SystemHealth(
                status=status,
                message=message,
                details={
                    "python_version": python_version,
                    "platform": platform_info,
                    "executable": sys.executable,
                },
                timestamp=time.time(),
                component="python_env"
            )
            
        except Exception as e:
            return SystemHealth(
                status=HealthStatus.CRITICAL,
                message=f"Python environment check failed: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                component="python_env"
            )
    
    def _check_security_health(self) -> SystemHealth:
        """Check security configuration."""
        try:
            security_status = self.security_manager.check_system_security()
            
            if security_status["overall_risk"] == "LOW":
                status = HealthStatus.HEALTHY
                message = "Security configuration healthy"
            elif security_status["overall_risk"] == "MEDIUM":
                status = HealthStatus.WARNING
                message = "Security configuration has minor issues"
            else:
                status = HealthStatus.CRITICAL
                message = "Security configuration has critical issues"
                
            return SystemHealth(
                status=status,
                message=message,
                details=security_status,
                timestamp=time.time(),
                component="security"
            )
            
        except Exception as e:
            return SystemHealth(
                status=HealthStatus.WARNING,
                message=f"Security health check failed: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                component="security"
            )
    
    def _check_file_permissions(self) -> SystemHealth:
        """Check file system permissions."""
        try:
            # Check write permissions in common directories
            test_dirs = [
                Path.home() / ".neorl_industrial",
                Path("/tmp"),
                Path.cwd(),
            ]
            
            permission_results = {}
            all_accessible = True
            
            for test_dir in test_dirs:
                try:
                    test_dir.mkdir(parents=True, exist_ok=True)
                    test_file = test_dir / "health_check_test.tmp"
                    
                    # Test write
                    test_file.write_text("test")
                    # Test read
                    content = test_file.read_text()
                    # Cleanup
                    test_file.unlink()
                    
                    permission_results[str(test_dir)] = {
                        "readable": True,
                        "writable": True,
                        "accessible": True,
                    }
                    
                except PermissionError:
                    permission_results[str(test_dir)] = {
                        "readable": False,
                        "writable": False, 
                        "accessible": False,
                    }
                    all_accessible = False
                except Exception as e:
                    permission_results[str(test_dir)] = {
                        "readable": False,
                        "writable": False,
                        "accessible": False,
                        "error": str(e),
                    }
                    all_accessible = False
            
            status = HealthStatus.HEALTHY if all_accessible else HealthStatus.WARNING
            message = (
                "File permissions healthy" if all_accessible 
                else "Some directories not accessible"
            )
            
            return SystemHealth(
                status=status,
                message=message,
                details=permission_results,
                timestamp=time.time(),
                component="permissions"
            )
            
        except Exception as e:
            return SystemHealth(
                status=HealthStatus.WARNING,
                message=f"Permission check failed: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                component="permissions"
            )
    
    def _check_storage_health(self) -> SystemHealth:
        """Check storage availability and usage."""
        try:
            import shutil
            
            # Check disk space
            usage = shutil.disk_usage("/")
            total_gb = usage.total / (1024**3)
            free_gb = usage.free / (1024**3)
            used_percent = (usage.used / usage.total) * 100
            
            # Determine health based on free space
            if free_gb < 1.0:  # Less than 1GB
                status = HealthStatus.CRITICAL
                message = f"Critically low disk space: {free_gb:.1f}GB remaining"
            elif free_gb < 5.0:  # Less than 5GB
                status = HealthStatus.WARNING
                message = f"Low disk space: {free_gb:.1f}GB remaining"
            else:
                status = HealthStatus.HEALTHY
                message = f"Sufficient disk space: {free_gb:.1f}GB remaining"
            
            return SystemHealth(
                status=status,
                message=message,
                details={
                    "total_gb": round(total_gb, 2),
                    "free_gb": round(free_gb, 2),
                    "used_percent": round(used_percent, 1),
                },
                timestamp=time.time(),
                component="storage"
            )
            
        except Exception as e:
            return SystemHealth(
                status=HealthStatus.WARNING,
                message=f"Storage check failed: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                component="storage"
            )
    
    def _check_dataset_availability(self) -> SystemHealth:
        """Check dataset availability and integrity."""
        try:
            dataset_dir = Path.home() / ".neorl_industrial" / "datasets"
            
            if not dataset_dir.exists():
                return SystemHealth(
                    status=HealthStatus.WARNING,
                    message="Dataset directory not found - datasets need to be downloaded",
                    details={
                        "dataset_dir": str(dataset_dir),
                        "exists": False,
                        "datasets_found": [],
                    },
                    timestamp=time.time(),
                    component="datasets"
                )
            
            # Check for dataset files
            dataset_files = list(dataset_dir.glob("*.h5"))
            dataset_count = len(dataset_files)
            
            if dataset_count == 0:
                status = HealthStatus.WARNING
                message = "No datasets found - download required for training"
            elif dataset_count < 7:  # Expected 7 environments
                status = HealthStatus.WARNING
                message = f"Only {dataset_count}/7 expected datasets found"
            else:
                status = HealthStatus.HEALTHY
                message = f"All {dataset_count} datasets available"
            
            return SystemHealth(
                status=status,
                message=message,
                details={
                    "dataset_dir": str(dataset_dir),
                    "exists": True,
                    "datasets_found": [f.name for f in dataset_files],
                    "dataset_count": dataset_count,
                },
                timestamp=time.time(),
                component="datasets"
            )
            
        except Exception as e:
            return SystemHealth(
                status=HealthStatus.WARNING,
                message=f"Dataset check failed: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                component="datasets"
            )
    
    def _check_memory_health(self) -> SystemHealth:
        """Check system memory usage."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            used_percent = memory.percent
            
            # Determine health based on available memory
            if available_gb < 0.5:  # Less than 500MB
                status = HealthStatus.CRITICAL
                message = f"Critically low memory: {available_gb:.1f}GB available"
            elif available_gb < 2.0:  # Less than 2GB
                status = HealthStatus.WARNING
                message = f"Low memory: {available_gb:.1f}GB available"
            else:
                status = HealthStatus.HEALTHY
                message = f"Sufficient memory: {available_gb:.1f}GB available"
            
            return SystemHealth(
                status=status,
                message=message,
                details={
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(available_gb, 2),
                    "used_percent": round(used_percent, 1),
                },
                timestamp=time.time(),
                component="memory"
            )
            
        except ImportError:
            return SystemHealth(
                status=HealthStatus.WARNING,
                message="psutil not available - cannot check memory",
                details={"psutil_installed": False},
                timestamp=time.time(),
                component="memory"
            )
        except Exception as e:
            return SystemHealth(
                status=HealthStatus.WARNING,
                message=f"Memory check failed: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                component="memory"
            )
    
    def _check_gpu_memory(self) -> SystemHealth:
        """Check GPU memory if available."""
        if not JAX_AVAILABLE:
            return SystemHealth(
                status=HealthStatus.WARNING,
                message="JAX not available - cannot check GPU memory",
                details={"jax_available": False},
                timestamp=time.time(),
                component="gpu_memory"
            )
        
        try:
            devices = jax.devices()
            gpu_devices = [d for d in devices if d.device_kind == "gpu"]
            
            if not gpu_devices:
                return SystemHealth(
                    status=HealthStatus.HEALTHY,
                    message="No GPU devices found - using CPU",
                    details={"gpu_devices": [], "using_cpu": True},
                    timestamp=time.time(),
                    component="gpu_memory"
                )
            
            # For GPU devices, we can't easily check memory without running operations
            # This is a simplified check
            gpu_info = {
                "gpu_count": len(gpu_devices),
                "gpu_devices": [str(d) for d in gpu_devices],
            }
            
            return SystemHealth(
                status=HealthStatus.HEALTHY,
                message=f"GPU memory check passed - {len(gpu_devices)} GPU(s) available",
                details=gpu_info,
                timestamp=time.time(),
                component="gpu_memory"
            )
            
        except Exception as e:
            return SystemHealth(
                status=HealthStatus.WARNING,
                message=f"GPU memory check failed: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                component="gpu_memory"
            )
    
    def _check_safety_systems(self) -> SystemHealth:
        """Check safety constraint systems."""
        try:
            # Test safety constraint functionality
            from ..core.types import SafetyConstraint
            
            # Create test constraint
            test_constraint = SafetyConstraint(
                name="test_temperature",
                type="range",
                min_value=0.0,
                max_value=100.0,
                violation_penalty=-10.0
            )
            
            # Test constraint evaluation
            test_passed = True
            error_details = {}
            
            try:
                # Test normal value
                normal_violation = test_constraint.check_violation(50.0)
                if normal_violation != 0.0:
                    test_passed = False
                    error_details["normal_value"] = "Should not violate"
                
                # Test violation
                high_violation = test_constraint.check_violation(150.0)
                if high_violation <= 0.0:
                    test_passed = False
                    error_details["high_value"] = "Should violate"
                
            except Exception as e:
                test_passed = False
                error_details["constraint_test"] = str(e)
            
            if test_passed:
                status = HealthStatus.HEALTHY
                message = "Safety constraint systems operational"
            else:
                status = HealthStatus.CRITICAL
                message = "Safety constraint systems not functioning properly"
            
            return SystemHealth(
                status=status,
                message=message,
                details={
                    "constraint_test_passed": test_passed,
                    "error_details": error_details,
                },
                timestamp=time.time(),
                component="safety_systems"
            )
            
        except Exception as e:
            return SystemHealth(
                status=HealthStatus.CRITICAL,
                message=f"Safety system check failed: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                component="safety_systems"
            )
    
    def _check_environment_health(self) -> SystemHealth:
        """Check industrial environment registry."""
        try:
            from ..environments import ENVIRONMENT_REGISTRY
            
            registered_envs = list(ENVIRONMENT_REGISTRY.keys())
            env_count = len(registered_envs)
            
            # Expected minimum environments
            expected_envs = [
                "ChemicalReactor-v0",
                "PowerGrid-v0", 
                "RobotAssembly-v0"
            ]
            
            missing_envs = [env for env in expected_envs if env not in registered_envs]
            
            if missing_envs:
                status = HealthStatus.WARNING
                message = f"Missing {len(missing_envs)} expected environments"
            else:
                status = HealthStatus.HEALTHY
                message = f"All {env_count} environments registered correctly"
            
            return SystemHealth(
                status=status,
                message=message,
                details={
                    "registered_environments": registered_envs,
                    "environment_count": env_count,
                    "missing_environments": missing_envs,
                },
                timestamp=time.time(),
                component="environment_registry"
            )
            
        except Exception as e:
            return SystemHealth(
                status=HealthStatus.WARNING,
                message=f"Environment registry check failed: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                component="environment_registry"
            )
    
    def _check_agent_health(self) -> SystemHealth:
        """Check agent registry and availability."""
        try:
            from ..agents import AGENT_REGISTRY
            
            registered_agents = list(AGENT_REGISTRY.keys())
            agent_count = len(registered_agents)
            
            # Expected minimum agents
            expected_agents = ["CQL", "IQL", "TD3BC"]
            missing_agents = [agent for agent in expected_agents if agent not in registered_agents]
            
            if missing_agents:
                status = HealthStatus.WARNING
                message = f"Missing {len(missing_agents)} expected agents"
            else:
                status = HealthStatus.HEALTHY
                message = f"All {agent_count} agents registered correctly"
            
            return SystemHealth(
                status=status,
                message=message,
                details={
                    "registered_agents": registered_agents,
                    "agent_count": agent_count,
                    "missing_agents": missing_agents,
                },
                timestamp=time.time(),
                component="agent_registry"
            )
            
        except Exception as e:
            return SystemHealth(
                status=HealthStatus.WARNING,
                message=f"Agent registry check failed: {e}",
                details={"error": str(e)},
                timestamp=time.time(),
                component="agent_registry"
            )
    
    def get_overall_health(self, health_results: Dict[str, SystemHealth]) -> SystemHealth:
        """Calculate overall system health from individual checks."""
        # Count status levels
        critical_count = sum(1 for h in health_results.values() if h.status == HealthStatus.CRITICAL)
        warning_count = sum(1 for h in health_results.values() if h.status == HealthStatus.WARNING)
        healthy_count = sum(1 for h in health_results.values() if h.status == HealthStatus.HEALTHY)
        
        # Determine overall status
        if critical_count > 0:
            status = HealthStatus.CRITICAL
            message = f"System has {critical_count} critical issue(s)"
        elif warning_count > 0:
            status = HealthStatus.WARNING  
            message = f"System has {warning_count} warning(s)"
        else:
            status = HealthStatus.HEALTHY
            message = "All systems healthy"
        
        critical_components = [
            name for name, health in health_results.items() 
            if health.status == HealthStatus.CRITICAL
        ]
        warning_components = [
            name for name, health in health_results.items() 
            if health.status == HealthStatus.WARNING
        ]
        
        return SystemHealth(
            status=status,
            message=message,
            details={
                "total_checks": len(health_results),
                "healthy_count": healthy_count,
                "warning_count": warning_count,
                "critical_count": critical_count,
                "critical_components": critical_components,
                "warning_components": warning_components,
            },
            timestamp=time.time(),
            component="overall"
        )
    
    def get_health_history(self, component: Optional[str] = None, limit: int = 10) -> List[SystemHealth]:
        """Get health check history.
        
        Args:
            component: Optional component filter
            limit: Maximum number of entries to return
            
        Returns:
            List of health check results
        """
        history = self._health_history
        
        if component:
            history = [h for h in history if h.component == component]
        
        # Return most recent entries
        return history[-limit:]


def check_system_health(
    check_dependencies: bool = True,
    check_security: bool = True, 
    check_storage: bool = True,
    check_memory: bool = True,
) -> Dict[str, SystemHealth]:
    """Convenience function to check system health.
    
    Args:
        check_dependencies: Check external dependencies
        check_security: Perform security health checks
        check_storage: Check storage and file system
        check_memory: Check memory usage
        
    Returns:
        Dictionary of health check results
    """
    checker = HealthChecker(
        check_dependencies=check_dependencies,
        check_security=check_security,
        check_storage=check_storage,
        check_memory=check_memory,
    )
    return checker.check_all()


def create_health_report(health_results: Dict[str, SystemHealth]) -> str:
    """Create formatted health report.
    
    Args:
        health_results: Health check results
        
    Returns:
        Formatted health report string
    """
    lines = []
    lines.append("=== SYSTEM HEALTH REPORT ===")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Overall summary
    checker = HealthChecker()
    overall_health = checker.get_overall_health(health_results)
    
    status_emoji = {
        HealthStatus.HEALTHY: "✅",
        HealthStatus.WARNING: "⚠️", 
        HealthStatus.CRITICAL: "❌",
        HealthStatus.UNKNOWN: "❔",
    }
    
    lines.append(f"OVERALL STATUS: {status_emoji[overall_health.status]} {overall_health.status.value.upper()}")
    lines.append(f"MESSAGE: {overall_health.message}")
    lines.append("")
    
    # Individual component results
    lines.append("COMPONENT DETAILS:")
    lines.append("-" * 50)
    
    for component, health in sorted(health_results.items()):
        emoji = status_emoji[health.status]
        lines.append(f"{emoji} {component.upper()}: {health.message}")
        
        # Add important details
        if health.status != HealthStatus.HEALTHY and health.details:
            for key, value in health.details.items():
                if key in ("error", "missing_environments", "missing_agents", "critical_components"):
                    lines.append(f"  └─ {key}: {value}")
        lines.append("")
    
    # Summary statistics
    lines.append("SUMMARY:")
    lines.append(f"  Total Checks: {overall_health.details['total_checks']}")
    lines.append(f"  Healthy: {overall_health.details['healthy_count']}")
    lines.append(f"  Warnings: {overall_health.details['warning_count']}")
    lines.append(f"  Critical: {overall_health.details['critical_count']}")
    
    if overall_health.details['critical_components']:
        lines.append(f"  Critical Components: {', '.join(overall_health.details['critical_components'])}")
    if overall_health.details['warning_components']:
        lines.append(f"  Warning Components: {', '.join(overall_health.details['warning_components'])}")
    
    lines.append("")
    lines.append("=== END HEALTH REPORT ===")
    
    return "\n".join(lines)