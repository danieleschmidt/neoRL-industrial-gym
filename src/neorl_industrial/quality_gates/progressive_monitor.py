"""Progressive Quality Monitor - Continuous quality monitoring during development."""

import time
import threading
import logging
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, Future

from .quality_metrics import QualityMetrics, QualityThresholds
from .gate_executor import QualityGateExecutor

logger = logging.getLogger(__name__)


@dataclass
class QualityEvent:
    """Represents a quality-related event."""
    timestamp: float
    event_type: str  # 'file_change', 'quality_check', 'threshold_violation'
    file_path: Optional[str] = None
    metrics: Optional[QualityMetrics] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileSnapshot:
    """Snapshot of a file for change detection."""
    path: str
    hash: str
    size: int
    modified_time: float


class ProgressiveQualityMonitor:
    """
    Progressive Quality Monitor provides real-time quality monitoring
    throughout the development lifecycle.
    
    Features:
    - File system watching for immediate quality checks
    - Progressive quality gate execution based on development phase
    - Adaptive thresholds based on project maturity
    - Real-time quality metrics dashboard
    - Automatic quality degradation prevention
    """
    
    def __init__(
        self,
        project_root: Path,
        thresholds: Optional[QualityThresholds] = None,
        check_interval: float = 5.0,
        enable_real_time: bool = True
    ):
        self.project_root = Path(project_root)
        self.thresholds = thresholds or QualityThresholds()
        self.check_interval = check_interval
        self.enable_real_time = enable_real_time
        
        # Core components
        self.gate_executor = QualityGateExecutor(project_root)
        self.quality_history: List[QualityEvent] = []
        self.file_snapshots: Dict[str, FileSnapshot] = {}
        
        # Monitoring state
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="quality-monitor")
        
        # Quality metrics
        self.current_metrics = QualityMetrics()
        self.last_full_check = 0.0
        self.check_lock = threading.RLock()
        
        # Event callbacks
        self.event_handlers: List[Callable[[QualityEvent], None]] = []
        
    def add_event_handler(self, handler: Callable[[QualityEvent], None]) -> None:
        """Add an event handler for quality events."""
        self.event_handlers.append(handler)
        
    def start_monitoring(self) -> None:
        """Start progressive quality monitoring."""
        if self.is_running:
            logger.warning("Quality monitor is already running")
            return
            
        logger.info("Starting progressive quality monitoring...")
        self.is_running = True
        
        # Initial scan
        self._initial_scan()
        
        if self.enable_real_time:
            self.monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="quality-monitor",
                daemon=True
            )
            self.monitor_thread.start()
            
        logger.info("Progressive quality monitoring started")
        
    def stop_monitoring(self) -> None:
        """Stop progressive quality monitoring."""
        if not self.is_running:
            return
            
        logger.info("Stopping progressive quality monitoring...")
        self.is_running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            
        self.executor.shutdown(wait=True)
        logger.info("Progressive quality monitoring stopped")
        
    def _initial_scan(self) -> None:
        """Perform initial scan of the project."""
        logger.info("Performing initial project scan...")
        
        # Scan all Python files
        python_files = list(self.project_root.rglob("*.py"))
        logger.info(f"Found {len(python_files)} Python files")
        
        # Create initial snapshots
        for py_file in python_files:
            try:
                self._create_file_snapshot(py_file)
            except Exception as e:
                logger.warning(f"Failed to create snapshot for {py_file}: {e}")
                
        # Run initial quality check
        self._run_quality_check(trigger="initial_scan")
        
    def _create_file_snapshot(self, file_path: Path) -> FileSnapshot:
        """Create a snapshot of a file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        content = file_path.read_text(encoding='utf-8')
        file_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        snapshot = FileSnapshot(
            path=str(file_path.relative_to(self.project_root)),
            hash=file_hash,
            size=len(content),
            modified_time=file_path.stat().st_mtime
        )
        
        self.file_snapshots[snapshot.path] = snapshot
        return snapshot
        
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Quality monitoring loop started")
        
        while self.is_running:
            try:
                self._check_for_changes()
                
                # Periodic full quality check
                if time.time() - self.last_full_check > 60.0:  # Every minute
                    self._run_quality_check(trigger="periodic")
                    self.last_full_check = time.time()
                    
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10.0)  # Back off on error
                
        logger.info("Quality monitoring loop ended")
        
    def _check_for_changes(self) -> None:
        """Check for file changes and trigger quality checks."""
        changed_files = []
        
        # Check existing files
        for file_path, snapshot in list(self.file_snapshots.items()):
            full_path = self.project_root / file_path
            
            if not full_path.exists():
                # File deleted
                del self.file_snapshots[file_path]
                self._emit_event(QualityEvent(
                    timestamp=time.time(),
                    event_type="file_deleted",
                    file_path=file_path
                ))
                continue
                
            current_mtime = full_path.stat().st_mtime
            if current_mtime > snapshot.modified_time:
                # File modified
                try:
                    new_snapshot = self._create_file_snapshot(full_path)
                    if new_snapshot.hash != snapshot.hash:
                        changed_files.append(file_path)
                        self._emit_event(QualityEvent(
                            timestamp=time.time(),
                            event_type="file_changed",
                            file_path=file_path,
                            details={
                                "old_size": snapshot.size,
                                "new_size": new_snapshot.size,
                                "size_delta": new_snapshot.size - snapshot.size
                            }
                        ))
                except Exception as e:
                    logger.warning(f"Failed to update snapshot for {file_path}: {e}")
                    
        # Check for new files
        python_files = list(self.project_root.rglob("*.py"))
        for py_file in python_files:
            rel_path = str(py_file.relative_to(self.project_root))
            if rel_path not in self.file_snapshots:
                try:
                    self._create_file_snapshot(py_file)
                    changed_files.append(rel_path)
                    self._emit_event(QualityEvent(
                        timestamp=time.time(),
                        event_type="file_added",
                        file_path=rel_path
                    ))
                except Exception as e:
                    logger.warning(f"Failed to create snapshot for new file {py_file}: {e}")
                    
        # Trigger quality check if files changed
        if changed_files:
            logger.info(f"Detected changes in {len(changed_files)} files, triggering quality check")
            self._run_quality_check(trigger="file_changes", changed_files=changed_files)
            
    def _run_quality_check(self, trigger: str, changed_files: Optional[List[str]] = None) -> None:
        """Run quality checks based on current state."""
        with self.check_lock:
            logger.info(f"Running quality check (trigger: {trigger})")
            
            try:
                # Run progressive quality gates
                results = self.gate_executor.run_progressive_gates(
                    phase=self._determine_project_phase(),
                    changed_files=changed_files
                )
                
                # Update current metrics
                self.current_metrics = QualityMetrics.from_gate_results(results)
                
                # Check for threshold violations
                violations = self._check_thresholds()
                
                # Emit quality check event
                self._emit_event(QualityEvent(
                    timestamp=time.time(),
                    event_type="quality_check",
                    metrics=self.current_metrics,
                    details={
                        "trigger": trigger,
                        "changed_files": changed_files or [],
                        "violations": violations,
                        "results": results
                    }
                ))
                
                # Handle violations
                if violations:
                    self._handle_violations(violations)
                    
                logger.info(f"Quality check completed (score: {self.current_metrics.overall_score:.1f})")
                
            except Exception as e:
                logger.error(f"Quality check failed: {e}")
                self._emit_event(QualityEvent(
                    timestamp=time.time(),
                    event_type="quality_check_failed",
                    details={"error": str(e), "trigger": trigger}
                ))
                
    def _determine_project_phase(self) -> str:
        """Determine current project development phase."""
        # Simple heuristic based on file count and test coverage
        python_files = len(list(self.project_root.rglob("*.py")))
        test_files = len(list(self.project_root.rglob("test_*.py"))) + \
                    len(list(self.project_root.rglob("*_test.py")))
        
        test_ratio = test_files / max(python_files, 1)
        
        if python_files < 10:
            return "prototype"
        elif test_ratio < 0.1:
            return "development"
        elif test_ratio < 0.3:
            return "testing"
        else:
            return "production"
            
    def _check_thresholds(self) -> List[Dict[str, Any]]:
        """Check current metrics against thresholds."""
        violations = []
        
        if self.current_metrics.code_coverage < self.thresholds.min_code_coverage:
            violations.append({
                "type": "code_coverage",
                "current": self.current_metrics.code_coverage,
                "threshold": self.thresholds.min_code_coverage,
                "severity": "high"
            })
            
        if self.current_metrics.test_pass_rate < self.thresholds.min_test_pass_rate:
            violations.append({
                "type": "test_pass_rate",
                "current": self.current_metrics.test_pass_rate,
                "threshold": self.thresholds.min_test_pass_rate,
                "severity": "high"
            })
            
        if self.current_metrics.security_score < self.thresholds.min_security_score:
            violations.append({
                "type": "security_score",
                "current": self.current_metrics.security_score,
                "threshold": self.thresholds.min_security_score,
                "severity": "critical"
            })
            
        if self.current_metrics.performance_score < self.thresholds.min_performance_score:
            violations.append({
                "type": "performance_score",
                "current": self.current_metrics.performance_score,
                "threshold": self.thresholds.min_performance_score,
                "severity": "medium"
            })
            
        return violations
        
    def _handle_violations(self, violations: List[Dict[str, Any]]) -> None:
        """Handle quality threshold violations."""
        for violation in violations:
            severity = violation.get("severity", "medium")
            violation_type = violation["type"]
            
            logger.warning(f"Quality violation: {violation_type} "
                         f"({violation['current']:.1f} < {violation['threshold']:.1f}) "
                         f"[{severity}]")
                         
            # Emit violation event
            self._emit_event(QualityEvent(
                timestamp=time.time(),
                event_type="threshold_violation",
                details=violation
            ))
            
            # Auto-remediation for critical violations
            if severity == "critical":
                self._auto_remediate_violation(violation)
                
    def _auto_remediate_violation(self, violation: Dict[str, Any]) -> None:
        """Attempt automatic remediation of critical violations."""
        violation_type = violation["type"]
        
        logger.info(f"Attempting auto-remediation for {violation_type}")
        
        try:
            if violation_type == "security_score":
                # Run security fixes
                self.gate_executor.run_security_fixes()
            elif violation_type == "code_coverage":
                # Generate missing tests (placeholder)
                logger.info("Code coverage violation detected - consider adding tests")
            elif violation_type == "performance_score":
                # Run performance optimizations
                self.gate_executor.run_performance_optimizations()
                
            logger.info(f"Auto-remediation completed for {violation_type}")
            
        except Exception as e:
            logger.error(f"Auto-remediation failed for {violation_type}: {e}")
            
    def _emit_event(self, event: QualityEvent) -> None:
        """Emit a quality event to all handlers."""
        self.quality_history.append(event)
        
        # Limit history size
        if len(self.quality_history) > 1000:
            self.quality_history = self.quality_history[-800:]
            
        # Notify handlers
        for handler in self.event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.warning(f"Event handler failed: {e}")
                
    def get_quality_status(self) -> Dict[str, Any]:
        """Get current quality status."""
        return {
            "current_metrics": self.current_metrics,
            "is_monitoring": self.is_running,
            "last_check": self.last_full_check,
            "files_tracked": len(self.file_snapshots),
            "recent_events": self.quality_history[-10:] if self.quality_history else [],
            "project_phase": self._determine_project_phase(),
            "violations": self._check_thresholds()
        }
        
    def export_quality_report(self, output_path: Path) -> None:
        """Export detailed quality report."""
        report = {
            "timestamp": time.time(),
            "project_root": str(self.project_root),
            "quality_status": self.get_quality_status(),
            "thresholds": {
                "min_code_coverage": self.thresholds.min_code_coverage,
                "min_test_pass_rate": self.thresholds.min_test_pass_rate,
                "min_security_score": self.thresholds.min_security_score,
                "min_performance_score": self.thresholds.min_performance_score
            },
            "event_history": [
                {
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "file_path": event.file_path,
                    "details": event.details
                }
                for event in self.quality_history
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Quality report exported to {output_path}")
        
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()