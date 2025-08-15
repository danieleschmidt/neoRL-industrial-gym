#!/usr/bin/env python3
"""Test script for Progressive Quality Gates system."""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from neorl_industrial.quality_gates import (
    ProgressiveQualityMonitor,
    QualityMetrics,
    QualityThresholds,
    QualityGateExecutor,
    RealTimeQualityMonitor,
    AdaptiveQualityGates
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_quality_metrics():
    """Test quality metrics calculation."""
    logger.info("Testing quality metrics...")
    
    metrics = QualityMetrics(
        code_coverage=85.0,
        test_pass_rate=95.0,
        security_score=90.0,
        performance_score=75.0,
        documentation_coverage=80.0
    )
    
    print(f"Overall Score: {metrics.overall_score:.1f}")
    print(f"Risk Level: {metrics.risk_level}")
    print(f"Quality Trend: {metrics.quality_trend}")
    
    assert metrics.overall_score > 0
    assert metrics.risk_level in ["low", "medium", "high"]
    print("✅ Quality metrics test passed")


def test_quality_thresholds():
    """Test quality thresholds."""
    logger.info("Testing quality thresholds...")
    
    # Test phase-specific thresholds
    prototype_thresholds = QualityThresholds.for_phase("prototype")
    production_thresholds = QualityThresholds.for_phase("production")
    
    print(f"Prototype coverage threshold: {prototype_thresholds.min_code_coverage}")
    print(f"Production coverage threshold: {production_thresholds.min_code_coverage}")
    
    assert prototype_thresholds.min_code_coverage < production_thresholds.min_code_coverage
    print("✅ Quality thresholds test passed")


def test_gate_executor():
    """Test quality gate executor."""
    logger.info("Testing quality gate executor...")
    
    project_root = Path(__file__).parent
    executor = QualityGateExecutor(project_root)
    
    # Run development phase gates
    results = executor.run_progressive_gates(phase="development")
    
    print(f"Executed {len(results)} quality gates")
    for gate_name, result in results.items():
        print(f"  {gate_name}: {result.status} (score: {result.score:.1f})")
        
    assert len(results) > 0
    print("✅ Gate executor test passed")


def test_adaptive_gates():
    """Test adaptive quality gates."""
    logger.info("Testing adaptive gates...")
    
    project_root = Path(__file__).parent
    adaptive = AdaptiveQualityGates(project_root)
    
    # Add some mock metrics data
    for i in range(20):
        metrics = QualityMetrics(
            code_coverage=75.0 + i * 0.5,  # Improving trend
            test_pass_rate=90.0 + i * 0.2,
            security_score=85.0,
            performance_score=70.0 + i * 0.1
        )
        adaptive.add_metrics(metrics)
        
    report = adaptive.get_adaptation_report()
    print(f"Adaptation rules: {len(report['rules_status'])}")
    print(f"Data points: {report['project_context']['data_points']}")
    print(f"Project phase: {report['project_context']['phase']}")
    
    assert len(report['rules_status']) > 0
    print("✅ Adaptive gates test passed")


def test_progressive_monitor():
    """Test progressive quality monitor."""
    logger.info("Testing progressive monitor...")
    
    project_root = Path(__file__).parent
    
    # Create monitor but don't start real-time monitoring
    monitor = ProgressiveQualityMonitor(
        project_root=project_root,
        enable_real_time=False
    )
    
    # Test initial scan
    monitor._initial_scan()
    
    status = monitor.get_quality_status()
    print(f"Files tracked: {status['files_tracked']}")
    print(f"Current score: {status['current_metrics'].overall_score:.1f}")
    print(f"Project phase: {status['project_phase']}")
    
    assert status['files_tracked'] > 0
    print("✅ Progressive monitor test passed")


def test_integration():
    """Test integration between components."""
    logger.info("Testing component integration...")
    
    project_root = Path(__file__).parent
    
    # Initialize components
    thresholds = QualityThresholds()
    executor = QualityGateExecutor(project_root)
    adaptive = AdaptiveQualityGates(project_root, initial_thresholds=thresholds)
    
    # Run quality gates
    results = executor.run_progressive_gates(phase="development")
    
    # Create metrics from results
    metrics = QualityMetrics.from_gate_results(results)
    
    # Add to adaptive system
    adaptive.add_metrics(metrics)
    
    print(f"Integration test - Overall score: {metrics.overall_score:.1f}")
    
    report = adaptive.get_adaptation_report()
    print(f"Adaptive system processed metrics successfully")
    
    assert metrics.overall_score >= 0
    print("✅ Integration test passed")


def main():
    """Run all tests."""
    print("🧪 Testing Progressive Quality Gates System")
    print("=" * 50)
    
    tests = [
        test_quality_metrics,
        test_quality_thresholds, 
        test_gate_executor,
        test_adaptive_gates,
        test_progressive_monitor,
        test_integration
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except Exception as e:
            logger.error(f"Test {test_func.__name__} failed: {e}")
            failed += 1
            print()
            
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All Progressive Quality Gates tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())