"""Quality metrics and thresholds for progressive quality monitoring."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import math


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics."""
    
    # Code quality metrics
    code_coverage: float = 0.0
    test_pass_rate: float = 0.0
    cyclomatic_complexity: float = 0.0
    code_duplication: float = 0.0
    
    # Security metrics
    security_score: float = 100.0
    vulnerability_count: int = 0
    high_severity_vulnerabilities: int = 0
    
    # Performance metrics
    performance_score: float = 100.0
    response_time_p95: float = 0.0
    memory_usage_mb: float = 0.0
    throughput_ops_per_sec: float = 0.0
    
    # Documentation metrics
    documentation_coverage: float = 0.0
    api_documentation_complete: bool = False
    readme_quality_score: float = 0.0
    
    # Maintainability metrics
    maintainability_index: float = 0.0
    technical_debt_hours: float = 0.0
    code_smells: int = 0
    
    # Process metrics
    build_success_rate: float = 0.0
    deployment_frequency: float = 0.0
    lead_time_hours: float = 0.0
    
    # Computed metrics
    overall_score: float = field(init=False, default=0.0)
    quality_trend: str = field(init=False, default="stable")
    risk_level: str = field(init=False, default="low")
    
    def __post_init__(self):
        """Compute derived metrics."""
        self.overall_score = self._calculate_overall_score()
        self.risk_level = self._calculate_risk_level()
        
    def _calculate_overall_score(self) -> float:
        """Calculate overall quality score (0-100)."""
        weights = {
            'code_quality': 0.25,
            'security': 0.25,
            'performance': 0.15,
            'documentation': 0.10,
            'maintainability': 0.15,
            'process': 0.10
        }
        
        # Code quality component
        code_quality_score = (
            self.code_coverage * 0.4 +
            self.test_pass_rate * 0.3 +
            max(0, 100 - self.cyclomatic_complexity * 2) * 0.2 +
            max(0, 100 - self.code_duplication) * 0.1
        )
        
        # Security component
        security_component = min(self.security_score, 100.0)
        
        # Performance component  
        performance_component = min(self.performance_score, 100.0)
        
        # Documentation component
        documentation_component = (
            self.documentation_coverage * 0.6 +
            (100.0 if self.api_documentation_complete else 0.0) * 0.2 +
            self.readme_quality_score * 0.2
        )
        
        # Maintainability component
        maintainability_component = max(0, min(100, self.maintainability_index))
        
        # Process component
        process_component = (
            self.build_success_rate * 0.6 +
            min(100, self.deployment_frequency * 10) * 0.2 +
            max(0, 100 - self.lead_time_hours) * 0.2
        )
        
        # Weighted sum
        overall = (
            code_quality_score * weights['code_quality'] +
            security_component * weights['security'] +
            performance_component * weights['performance'] +
            documentation_component * weights['documentation'] +
            maintainability_component * weights['maintainability'] +
            process_component * weights['process']
        )
        
        return max(0.0, min(100.0, overall))
        
    def _calculate_risk_level(self) -> str:
        """Calculate risk level based on metrics."""
        if (self.high_severity_vulnerabilities > 0 or 
            self.security_score < 60 or
            self.test_pass_rate < 70):
            return "high"
        elif (self.overall_score < 70 or
              self.code_coverage < 60 or
              self.vulnerability_count > 5):
            return "medium"
        else:
            return "low"
            
    def update_trend(self, previous_metrics: Optional['QualityMetrics']) -> None:
        """Update quality trend based on previous metrics."""
        if not previous_metrics:
            self.quality_trend = "stable"
            return
            
        score_delta = self.overall_score - previous_metrics.overall_score
        
        if score_delta > 2.0:
            self.quality_trend = "improving"
        elif score_delta < -2.0:
            self.quality_trend = "declining"
        else:
            self.quality_trend = "stable"
            
    @classmethod
    def from_gate_results(cls, gate_results: Dict[str, Any]) -> 'QualityMetrics':
        """Create quality metrics from quality gate results."""
        metrics = cls()
        
        # Extract metrics from gate results
        if "Unit Tests" in gate_results:
            test_result = gate_results["Unit Tests"].details
            if "tests_passed" in test_result and "tests_failed" in test_result:
                total_tests = test_result["tests_passed"] + test_result["tests_failed"]
                if total_tests > 0:
                    metrics.test_pass_rate = (test_result["tests_passed"] / total_tests) * 100
            
            if "coverage_percent" in test_result:
                metrics.code_coverage = test_result.get("coverage_percent", 0.0)
                
        if "Security Scan" in gate_results:
            security_result = gate_results["Security Scan"].details
            metrics.high_severity_vulnerabilities = security_result.get("high_severity_issues", 0)
            metrics.vulnerability_count = security_result.get("total_issues", 0)
            
            # Calculate security score
            if metrics.high_severity_vulnerabilities == 0:
                metrics.security_score = max(70, 100 - metrics.vulnerability_count * 5)
            else:
                metrics.security_score = max(30, 70 - metrics.high_severity_vulnerabilities * 15)
                
        if "Performance Tests" in gate_results:
            perf_result = gate_results["Performance Tests"].details
            perf_metrics = perf_result.get("performance_metrics", {})
            
            if "steps_per_second" in perf_metrics:
                metrics.throughput_ops_per_sec = perf_metrics["steps_per_second"]
                # Score based on throughput (assuming 1000+ is excellent)
                metrics.performance_score = min(100, metrics.throughput_ops_per_sec / 10)
                
            if "memory_mb" in perf_metrics:
                metrics.memory_usage_mb = perf_metrics["memory_mb"]
                
        if "Documentation Check" in gate_results:
            doc_result = gate_results["Documentation Check"].details
            metrics.documentation_coverage = doc_result.get("docstring_coverage_percent", 0.0)
            metrics.api_documentation_complete = len(doc_result.get("missing_docs", [])) == 0
            
            # Simple README quality score
            present_docs = len(doc_result.get("present_docs", []))
            total_expected = present_docs + len(doc_result.get("missing_docs", []))
            if total_expected > 0:
                metrics.readme_quality_score = (present_docs / total_expected) * 100
                
        if "Code Style" in gate_results:
            style_result = gate_results["Code Style"].details
            issues = style_result.get("ruff_issues", 0)
            # Convert issues to quality score
            if issues == 0:
                code_style_score = 100
            else:
                code_style_score = max(50, 100 - issues * 2)
                
            # Use for maintainability index
            metrics.maintainability_index = code_style_score
            
        # Set build success rate based on overall gate success
        passed_gates = sum(1 for result in gate_results.values() 
                          if result.status == "passed")
        total_gates = len(gate_results)
        if total_gates > 0:
            metrics.build_success_rate = (passed_gates / total_gates) * 100
            
        # Recalculate overall score
        metrics.overall_score = metrics._calculate_overall_score()
        metrics.risk_level = metrics._calculate_risk_level()
        
        return metrics
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "code_coverage": self.code_coverage,
            "test_pass_rate": self.test_pass_rate,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "code_duplication": self.code_duplication,
            "security_score": self.security_score,
            "vulnerability_count": self.vulnerability_count,
            "high_severity_vulnerabilities": self.high_severity_vulnerabilities,
            "performance_score": self.performance_score,
            "response_time_p95": self.response_time_p95,
            "memory_usage_mb": self.memory_usage_mb,
            "throughput_ops_per_sec": self.throughput_ops_per_sec,
            "documentation_coverage": self.documentation_coverage,
            "api_documentation_complete": self.api_documentation_complete,
            "readme_quality_score": self.readme_quality_score,
            "maintainability_index": self.maintainability_index,
            "technical_debt_hours": self.technical_debt_hours,
            "code_smells": self.code_smells,
            "build_success_rate": self.build_success_rate,
            "deployment_frequency": self.deployment_frequency,
            "lead_time_hours": self.lead_time_hours,
            "overall_score": self.overall_score,
            "quality_trend": self.quality_trend,
            "risk_level": self.risk_level
        }


@dataclass
class QualityThresholds:
    """Quality thresholds for different project phases."""
    
    # Minimum acceptable values
    min_code_coverage: float = 80.0
    min_test_pass_rate: float = 95.0
    min_security_score: float = 85.0
    min_performance_score: float = 70.0
    min_documentation_coverage: float = 75.0
    min_overall_score: float = 75.0
    
    # Maximum acceptable values
    max_cyclomatic_complexity: float = 10.0
    max_code_duplication: float = 5.0
    max_technical_debt_hours: float = 40.0
    max_vulnerability_count: int = 3
    max_high_severity_vulnerabilities: int = 0
    
    # Performance thresholds
    max_response_time_p95: float = 200.0  # milliseconds
    max_memory_usage_mb: float = 512.0
    min_throughput_ops_per_sec: float = 100.0
    
    @classmethod
    def for_phase(cls, phase: str) -> 'QualityThresholds':
        """Get quality thresholds for a specific development phase."""
        if phase == "prototype":
            return cls(
                min_code_coverage=50.0,
                min_test_pass_rate=80.0,
                min_security_score=70.0,
                min_performance_score=50.0,
                min_documentation_coverage=30.0,
                min_overall_score=60.0,
                max_vulnerability_count=10,
                max_technical_debt_hours=100.0
            )
        elif phase == "development":
            return cls(
                min_code_coverage=70.0,
                min_test_pass_rate=90.0,
                min_security_score=80.0,
                min_performance_score=60.0,
                min_documentation_coverage=50.0,
                min_overall_score=70.0,
                max_vulnerability_count=5,
                max_technical_debt_hours=60.0
            )
        elif phase == "testing":
            return cls(
                min_code_coverage=85.0,
                min_test_pass_rate=95.0,
                min_security_score=90.0,
                min_performance_score=75.0,
                min_documentation_coverage=80.0,
                min_overall_score=80.0,
                max_vulnerability_count=2,
                max_technical_debt_hours=30.0
            )
        else:  # production
            return cls(
                min_code_coverage=90.0,
                min_test_pass_rate=98.0,
                min_security_score=95.0,
                min_performance_score=85.0,
                min_documentation_coverage=90.0,
                min_overall_score=85.0,
                max_vulnerability_count=0,
                max_technical_debt_hours=20.0
            )
            
    def is_passing(self, metrics: QualityMetrics) -> bool:
        """Check if metrics meet all thresholds."""
        return (
            metrics.code_coverage >= self.min_code_coverage and
            metrics.test_pass_rate >= self.min_test_pass_rate and
            metrics.security_score >= self.min_security_score and
            metrics.performance_score >= self.min_performance_score and
            metrics.documentation_coverage >= self.min_documentation_coverage and
            metrics.overall_score >= self.min_overall_score and
            metrics.cyclomatic_complexity <= self.max_cyclomatic_complexity and
            metrics.code_duplication <= self.max_code_duplication and
            metrics.technical_debt_hours <= self.max_technical_debt_hours and
            metrics.vulnerability_count <= self.max_vulnerability_count and
            metrics.high_severity_vulnerabilities <= self.max_high_severity_vulnerabilities
        )
        
    def get_violations(self, metrics: QualityMetrics) -> List[Dict[str, Any]]:
        """Get list of threshold violations."""
        violations = []
        
        if metrics.code_coverage < self.min_code_coverage:
            violations.append({
                "metric": "code_coverage",
                "current": metrics.code_coverage,
                "threshold": self.min_code_coverage,
                "type": "minimum"
            })
            
        if metrics.test_pass_rate < self.min_test_pass_rate:
            violations.append({
                "metric": "test_pass_rate",
                "current": metrics.test_pass_rate,
                "threshold": self.min_test_pass_rate,
                "type": "minimum"
            })
            
        if metrics.security_score < self.min_security_score:
            violations.append({
                "metric": "security_score",
                "current": metrics.security_score,
                "threshold": self.min_security_score,
                "type": "minimum"
            })
            
        if metrics.performance_score < self.min_performance_score:
            violations.append({
                "metric": "performance_score",
                "current": metrics.performance_score,
                "threshold": self.min_performance_score,
                "type": "minimum"
            })
            
        if metrics.high_severity_vulnerabilities > self.max_high_severity_vulnerabilities:
            violations.append({
                "metric": "high_severity_vulnerabilities",
                "current": metrics.high_severity_vulnerabilities,
                "threshold": self.max_high_severity_vulnerabilities,
                "type": "maximum"
            })
            
        return violations