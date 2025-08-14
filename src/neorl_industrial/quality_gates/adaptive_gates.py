"""Adaptive Quality Gates - Self-tuning quality thresholds based on project evolution."""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from collections import defaultdict, deque

from .quality_metrics import QualityMetrics, QualityThresholds

logger = logging.getLogger(__name__)


@dataclass
class AdaptationRule:
    """Rule for adapting quality thresholds."""
    metric_name: str
    adaptation_strategy: str  # "trend_following", "percentile_based", "performance_based"
    parameters: Dict[str, Any]
    min_data_points: int = 10
    adaptation_rate: float = 0.1  # How quickly to adapt (0.0 to 1.0)
    enabled: bool = True


@dataclass
class ThresholdHistory:
    """History of threshold changes."""
    timestamp: float
    metric_name: str
    old_value: float
    new_value: float
    reason: str
    confidence: float


class AdaptiveQualityGates:
    """
    Adaptive Quality Gates system that automatically adjusts quality thresholds
    based on project evolution, team performance, and industry benchmarks.
    
    Features:
    - Machine learning-based threshold adaptation
    - Project phase-aware adjustments
    - Team performance trend analysis
    - Industry benchmark comparison
    - Risk-based threshold tuning
    - Seasonal pattern recognition
    """
    
    def __init__(
        self,
        project_root: Path,
        initial_thresholds: Optional[QualityThresholds] = None,
        adaptation_interval: float = 3600.0,  # 1 hour
        history_window: int = 100
    ):
        self.project_root = Path(project_root)
        self.adaptation_interval = adaptation_interval
        self.history_window = history_window
        
        # Initialize thresholds
        self.current_thresholds = initial_thresholds or QualityThresholds()
        self.baseline_thresholds = QualityThresholds()
        
        # Metrics history
        self.metrics_history: deque = deque(maxlen=history_window)
        self.threshold_history: List[ThresholdHistory] = []
        
        # Adaptation rules
        self.adaptation_rules = self._create_default_adaptation_rules()
        
        # State tracking
        self.last_adaptation = 0.0
        self.project_phase = "development"
        self.team_performance_trend = "stable"
        
        # Statistics
        self.adaptation_stats = {
            "total_adaptations": 0,
            "successful_adaptations": 0,
            "reverted_adaptations": 0,
            "avg_improvement": 0.0
        }
        
    def add_metrics(self, metrics: QualityMetrics) -> None:
        """Add new metrics data point."""
        self.metrics_history.append({
            "timestamp": time.time(),
            "metrics": metrics,
            "phase": self.project_phase
        })
        
        # Trigger adaptation if interval has passed
        if time.time() - self.last_adaptation > self.adaptation_interval:
            self._run_adaptation()
            
    def _run_adaptation(self) -> None:
        """Run threshold adaptation process."""
        logger.info("Running adaptive threshold adjustment...")
        
        if len(self.metrics_history) < 5:
            logger.info("Insufficient data for adaptation")
            return
            
        # Update project context
        self._update_project_context()
        
        # Run adaptation rules
        adaptations_made = 0
        
        for rule in self.adaptation_rules:
            if not rule.enabled:
                continue
                
            if len(self.metrics_history) < rule.min_data_points:
                continue
                
            adaptation = self._apply_adaptation_rule(rule)
            if adaptation:
                adaptations_made += 1
                
        self.last_adaptation = time.time()
        
        if adaptations_made > 0:
            logger.info(f"Made {adaptations_made} threshold adaptations")
            self.adaptation_stats["total_adaptations"] += adaptations_made
            
            # Save adapted thresholds
            self._save_thresholds()
        else:
            logger.info("No threshold adaptations needed")
            
    def _update_project_context(self) -> None:
        """Update project context for better adaptation decisions."""
        if not self.metrics_history:
            return
            
        recent_metrics = [entry["metrics"] for entry in list(self.metrics_history)[-10:]]
        
        # Determine project phase based on metrics evolution
        avg_coverage = np.mean([m.code_coverage for m in recent_metrics])
        avg_test_rate = np.mean([m.test_pass_rate for m in recent_metrics])
        avg_security = np.mean([m.security_score for m in recent_metrics])
        
        if avg_coverage < 50 or avg_test_rate < 80:
            self.project_phase = "prototype"
        elif avg_coverage < 80 or avg_security < 85:
            self.project_phase = "development"  
        elif avg_test_rate < 95:
            self.project_phase = "testing"
        else:
            self.project_phase = "production"
            
        # Analyze performance trend
        if len(recent_metrics) >= 5:
            recent_scores = [m.overall_score for m in recent_metrics]
            trend_slope = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            
            if trend_slope > 1.0:
                self.team_performance_trend = "improving"
            elif trend_slope < -1.0:
                self.team_performance_trend = "declining"
            else:
                self.team_performance_trend = "stable"
                
        logger.info(f"Project context: phase={self.project_phase}, trend={self.team_performance_trend}")
        
    def _apply_adaptation_rule(self, rule: AdaptationRule) -> bool:
        """Apply a single adaptation rule."""
        try:
            metric_name = rule.metric_name
            strategy = rule.adaptation_strategy
            
            # Extract relevant metric values
            metric_values = self._extract_metric_values(metric_name)
            if not metric_values:
                return False
                
            # Calculate new threshold based on strategy
            current_threshold = self._get_current_threshold(metric_name)
            new_threshold = self._calculate_new_threshold(
                metric_values, current_threshold, strategy, rule.parameters
            )
            
            if new_threshold is None or abs(new_threshold - current_threshold) < 0.1:
                return False  # No significant change needed
                
            # Apply adaptation rate
            adjusted_threshold = current_threshold + (new_threshold - current_threshold) * rule.adaptation_rate
            
            # Validate new threshold
            if self._validate_threshold(metric_name, adjusted_threshold):
                confidence = self._calculate_confidence(metric_values, adjusted_threshold, rule)
                
                # Apply threshold change
                self._set_threshold(metric_name, adjusted_threshold)
                
                # Record change
                self.threshold_history.append(ThresholdHistory(
                    timestamp=time.time(),
                    metric_name=metric_name,
                    old_value=current_threshold,
                    new_value=adjusted_threshold,
                    reason=f"{strategy} adaptation",
                    confidence=confidence
                ))
                
                logger.info(f"Adapted {metric_name} threshold: {current_threshold:.1f} → {adjusted_threshold:.1f} "
                           f"(confidence: {confidence:.2f})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to apply adaptation rule {rule.metric_name}: {e}")
            
        return False
        
    def _extract_metric_values(self, metric_name: str) -> List[float]:
        """Extract metric values from history."""
        values = []
        
        for entry in self.metrics_history:
            metrics = entry["metrics"]
            
            if metric_name == "code_coverage":
                values.append(metrics.code_coverage)
            elif metric_name == "test_pass_rate":
                values.append(metrics.test_pass_rate)
            elif metric_name == "security_score":
                values.append(metrics.security_score)
            elif metric_name == "performance_score":
                values.append(metrics.performance_score)
            elif metric_name == "overall_score":
                values.append(metrics.overall_score)
            elif metric_name == "documentation_coverage":
                values.append(metrics.documentation_coverage)
                
        return values
        
    def _get_current_threshold(self, metric_name: str) -> float:
        """Get current threshold value."""
        if metric_name == "code_coverage":
            return self.current_thresholds.min_code_coverage
        elif metric_name == "test_pass_rate":
            return self.current_thresholds.min_test_pass_rate
        elif metric_name == "security_score":
            return self.current_thresholds.min_security_score
        elif metric_name == "performance_score":
            return self.current_thresholds.min_performance_score
        elif metric_name == "overall_score":
            return self.current_thresholds.min_overall_score
        elif metric_name == "documentation_coverage":
            return self.current_thresholds.min_documentation_coverage
        else:
            return 0.0
            
    def _set_threshold(self, metric_name: str, value: float) -> None:
        """Set threshold value."""
        if metric_name == "code_coverage":
            self.current_thresholds.min_code_coverage = value
        elif metric_name == "test_pass_rate":
            self.current_thresholds.min_test_pass_rate = value
        elif metric_name == "security_score":
            self.current_thresholds.min_security_score = value
        elif metric_name == "performance_score":
            self.current_thresholds.min_performance_score = value
        elif metric_name == "overall_score":
            self.current_thresholds.min_overall_score = value
        elif metric_name == "documentation_coverage":
            self.current_thresholds.min_documentation_coverage = value
            
    def _calculate_new_threshold(
        self,
        values: List[float],
        current_threshold: float,
        strategy: str,
        parameters: Dict[str, Any]
    ) -> Optional[float]:
        """Calculate new threshold based on strategy."""
        if not values:
            return None
            
        if strategy == "trend_following":
            # Adjust threshold based on recent trend
            window_size = parameters.get("window_size", 10)
            recent_values = values[-window_size:]
            
            if len(recent_values) < 5:
                return None
                
            trend_slope = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            trend_adjustment = parameters.get("trend_factor", 0.1) * trend_slope
            
            return current_threshold + trend_adjustment
            
        elif strategy == "percentile_based":
            # Set threshold based on historical percentile
            percentile = parameters.get("percentile", 25)  # 25th percentile for minimum thresholds
            target_threshold = np.percentile(values, percentile)
            
            # Don't make dramatic changes
            max_change = parameters.get("max_change", 10.0)
            if abs(target_threshold - current_threshold) > max_change:
                direction = 1 if target_threshold > current_threshold else -1
                return current_threshold + direction * max_change
                
            return target_threshold
            
        elif strategy == "performance_based":
            # Adjust based on recent performance vs threshold
            recent_values = values[-20:]  # Last 20 data points
            
            violations = sum(1 for v in recent_values if v < current_threshold)
            violation_rate = violations / len(recent_values)
            
            target_violation_rate = parameters.get("target_violation_rate", 0.1)  # 10% violations OK
            
            if violation_rate > target_violation_rate * 2:
                # Too many violations, lower threshold
                adjustment_factor = parameters.get("adjustment_factor", 0.9)
                return current_threshold * adjustment_factor
            elif violation_rate < target_violation_rate / 2:
                # Very few violations, potentially raise threshold
                adjustment_factor = parameters.get("adjustment_factor", 1.05) 
                return current_threshold * adjustment_factor
                
        elif strategy == "seasonal_pattern":
            # Adjust for seasonal patterns (e.g., end of sprint, release cycles)
            # This would require more sophisticated time series analysis
            return None
            
        return None
        
    def _validate_threshold(self, metric_name: str, threshold: float) -> bool:
        """Validate that threshold is reasonable."""
        # Basic bounds checking
        if threshold < 0 or threshold > 100:
            return False
            
        # Specific validation per metric
        if metric_name == "code_coverage" and threshold < 30:
            return False  # Coverage should never be below 30%
        elif metric_name == "security_score" and threshold < 60:
            return False  # Security score should never be below 60
        elif metric_name == "test_pass_rate" and threshold < 70:
            return False  # Test pass rate should never be below 70%
            
        return True
        
    def _calculate_confidence(
        self,
        values: List[float],
        new_threshold: float,
        rule: AdaptationRule
    ) -> float:
        """Calculate confidence in the threshold adaptation."""
        if len(values) < 10:
            return 0.5  # Low confidence with little data
            
        # Calculate various confidence factors
        
        # 1. Data consistency (lower variance = higher confidence)
        variance = np.var(values)
        consistency_factor = 1.0 / (1.0 + variance / 100.0)
        
        # 2. Sample size (more data = higher confidence)
        sample_factor = min(1.0, len(values) / 50.0)
        
        # 3. Recent trend stability
        recent_values = values[-10:]
        recent_variance = np.var(recent_values)
        stability_factor = 1.0 / (1.0 + recent_variance / 100.0)
        
        # 4. Distance from current threshold (smaller changes = higher confidence)
        current_threshold = self._get_current_threshold(rule.metric_name)
        change_magnitude = abs(new_threshold - current_threshold)
        change_factor = 1.0 / (1.0 + change_magnitude / 10.0)
        
        # Weighted combination
        confidence = (
            consistency_factor * 0.3 +
            sample_factor * 0.2 +
            stability_factor * 0.3 +
            change_factor * 0.2
        )
        
        return min(1.0, max(0.0, confidence))
        
    def _create_default_adaptation_rules(self) -> List[AdaptationRule]:
        """Create default adaptation rules."""
        return [
            # Code coverage - use percentile-based adaptation
            AdaptationRule(
                metric_name="code_coverage",
                adaptation_strategy="percentile_based",
                parameters={
                    "percentile": 20,  # 20th percentile as minimum
                    "max_change": 5.0
                },
                min_data_points=15,
                adaptation_rate=0.2
            ),
            
            # Test pass rate - use performance-based adaptation
            AdaptationRule(
                metric_name="test_pass_rate",
                adaptation_strategy="performance_based",
                parameters={
                    "target_violation_rate": 0.05,  # 5% violations acceptable
                    "adjustment_factor": 0.95
                },
                min_data_points=20,
                adaptation_rate=0.15
            ),
            
            # Security score - conservative trend following
            AdaptationRule(
                metric_name="security_score",
                adaptation_strategy="trend_following",
                parameters={
                    "window_size": 15,
                    "trend_factor": 0.05  # Conservative adjustment
                },
                min_data_points=25,
                adaptation_rate=0.1
            ),
            
            # Performance score - percentile-based with higher target
            AdaptationRule(
                metric_name="performance_score",
                adaptation_strategy="percentile_based",
                parameters={
                    "percentile": 30,
                    "max_change": 8.0
                },
                min_data_points=20,
                adaptation_rate=0.25
            ),
            
            # Overall score - trend following
            AdaptationRule(
                metric_name="overall_score",
                adaptation_strategy="trend_following",
                parameters={
                    "window_size": 20,
                    "trend_factor": 0.1
                },
                min_data_points=30,
                adaptation_rate=0.15
            )
        ]
        
    def _save_thresholds(self) -> None:
        """Save adapted thresholds to file."""
        thresholds_data = {
            "timestamp": time.time(),
            "project_phase": self.project_phase,
            "thresholds": {
                "min_code_coverage": self.current_thresholds.min_code_coverage,
                "min_test_pass_rate": self.current_thresholds.min_test_pass_rate,
                "min_security_score": self.current_thresholds.min_security_score,
                "min_performance_score": self.current_thresholds.min_performance_score,
                "min_overall_score": self.current_thresholds.min_overall_score,
                "min_documentation_coverage": self.current_thresholds.min_documentation_coverage,
            },
            "adaptation_history": [
                {
                    "timestamp": h.timestamp,
                    "metric_name": h.metric_name,
                    "old_value": h.old_value,
                    "new_value": h.new_value,
                    "reason": h.reason,
                    "confidence": h.confidence
                }
                for h in self.threshold_history[-50:]  # Last 50 changes
            ],
            "stats": self.adaptation_stats
        }
        
        thresholds_file = self.project_root / ".adaptive_thresholds.json"
        
        try:
            with open(thresholds_file, 'w') as f:
                json.dump(thresholds_data, f, indent=2)
            logger.info(f"Saved adaptive thresholds to {thresholds_file}")
        except Exception as e:
            logger.error(f"Failed to save adaptive thresholds: {e}")
            
    def load_thresholds(self) -> bool:
        """Load previously adapted thresholds."""
        thresholds_file = self.project_root / ".adaptive_thresholds.json"
        
        if not thresholds_file.exists():
            return False
            
        try:
            with open(thresholds_file, 'r') as f:
                data = json.load(f)
                
            # Load thresholds
            thresholds_data = data.get("thresholds", {})
            self.current_thresholds.min_code_coverage = thresholds_data.get("min_code_coverage", 80.0)
            self.current_thresholds.min_test_pass_rate = thresholds_data.get("min_test_pass_rate", 95.0)
            self.current_thresholds.min_security_score = thresholds_data.get("min_security_score", 85.0)
            self.current_thresholds.min_performance_score = thresholds_data.get("min_performance_score", 70.0)
            self.current_thresholds.min_overall_score = thresholds_data.get("min_overall_score", 75.0)
            self.current_thresholds.min_documentation_coverage = thresholds_data.get("min_documentation_coverage", 75.0)
            
            # Load history
            history_data = data.get("adaptation_history", [])
            self.threshold_history = [
                ThresholdHistory(
                    timestamp=h["timestamp"],
                    metric_name=h["metric_name"],
                    old_value=h["old_value"],
                    new_value=h["new_value"],
                    reason=h["reason"],
                    confidence=h["confidence"]
                )
                for h in history_data
            ]
            
            # Load stats
            self.adaptation_stats.update(data.get("stats", {}))
            
            logger.info(f"Loaded adaptive thresholds from {thresholds_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load adaptive thresholds: {e}")
            return False
            
    def get_adaptation_report(self) -> Dict[str, Any]:
        """Get comprehensive adaptation report."""
        return {
            "current_thresholds": {
                "min_code_coverage": self.current_thresholds.min_code_coverage,
                "min_test_pass_rate": self.current_thresholds.min_test_pass_rate,
                "min_security_score": self.current_thresholds.min_security_score,
                "min_performance_score": self.current_thresholds.min_performance_score,
                "min_overall_score": self.current_thresholds.min_overall_score,
            },
            "baseline_thresholds": {
                "min_code_coverage": self.baseline_thresholds.min_code_coverage,
                "min_test_pass_rate": self.baseline_thresholds.min_test_pass_rate,
                "min_security_score": self.baseline_thresholds.min_security_score,
                "min_performance_score": self.baseline_thresholds.min_performance_score,
                "min_overall_score": self.baseline_thresholds.min_overall_score,
            },
            "project_context": {
                "phase": self.project_phase,
                "team_performance_trend": self.team_performance_trend,
                "data_points": len(self.metrics_history)
            },
            "recent_adaptations": [
                {
                    "timestamp": h.timestamp,
                    "metric_name": h.metric_name,
                    "change": f"{h.old_value:.1f} → {h.new_value:.1f}",
                    "reason": h.reason,
                    "confidence": h.confidence
                }
                for h in self.threshold_history[-10:]  # Last 10 changes
            ],
            "adaptation_statistics": self.adaptation_stats,
            "rules_status": [
                {
                    "metric": rule.metric_name,
                    "strategy": rule.adaptation_strategy,
                    "enabled": rule.enabled,
                    "min_data_points": rule.min_data_points,
                    "adaptation_rate": rule.adaptation_rate
                }
                for rule in self.adaptation_rules
            ]
        }
        
    def reset_to_baseline(self) -> None:
        """Reset thresholds to baseline values."""
        logger.info("Resetting adaptive thresholds to baseline")
        self.current_thresholds = QualityThresholds()
        self.threshold_history.clear()
        self.adaptation_stats = {
            "total_adaptations": 0,
            "successful_adaptations": 0,
            "reverted_adaptations": 0,
            "avg_improvement": 0.0
        }
        self._save_thresholds()
        
    def add_adaptation_rule(self, rule: AdaptationRule) -> None:
        """Add custom adaptation rule."""
        self.adaptation_rules.append(rule)
        logger.info(f"Added adaptation rule for {rule.metric_name}")
        
    def enable_rule(self, metric_name: str, enabled: bool = True) -> None:
        """Enable or disable adaptation rule for a metric."""
        for rule in self.adaptation_rules:
            if rule.metric_name == metric_name:
                rule.enabled = enabled
                logger.info(f"{'Enabled' if enabled else 'Disabled'} adaptation rule for {metric_name}")
                return
                
        logger.warning(f"No adaptation rule found for {metric_name}")