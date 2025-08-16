"""Integration tests for research components and academic validation."""

import pytest
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any

import neorl_industrial as ni
from neorl_industrial.agents.safety_critical import RiskAwareCQLAgent, ConstrainedIQLAgent
from neorl_industrial.benchmarks.industrial_benchmarks import (
    IndustrialBenchmarkSuite, 
    BenchmarkConfig
)
from neorl_industrial.core.types import SafetyConstraint
from neorl_industrial.data.industrial_data_loader import create_plc_loader
from neorl_industrial.experiments.research_validation import ResearchValidationFramework


class TestAdvancedEnvironments:
    """Test high-fidelity industrial environments."""
    
    def test_advanced_chemical_reactor_creation(self):
        """Test advanced chemical reactor environment creation."""
        env = ni.make('AdvancedChemicalReactor-v0')
        
        assert env is not None
        assert env.state_dim == 20
        assert env.action_dim == 6
        
        # Test reset
        obs, info = env.reset()
        assert obs.shape == (20,)
        assert isinstance(info, dict)
        
        # Test step
        action = jnp.zeros(6)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        assert next_obs.shape == (20,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)
        
    def test_advanced_power_grid_creation(self):
        """Test advanced power grid environment creation."""
        env = ni.make('AdvancedPowerGrid-v0')
        
        assert env is not None
        assert env.state_dim == 32
        assert env.action_dim == 8
        
        # Test reset
        obs, info = env.reset()
        assert obs.shape == (32,)
        
        # Test step with valid action
        action = jnp.array([30.0, 25.0, 20.0, 28.0, 1.0, 1.0, 0.0, 0.0])
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        assert next_obs.shape == (32,)
        assert isinstance(reward, float)
        
    def test_environment_safety_constraints(self):
        """Test safety constraint evaluation in environments."""
        env = ni.make('AdvancedChemicalReactor-v0')
        
        # Test safety metrics
        obs, _ = env.reset()
        safety_metrics = env.get_safety_metrics()
        
        assert hasattr(safety_metrics, 'total_violations')
        assert hasattr(safety_metrics, 'safety_score')
        assert hasattr(safety_metrics, 'constraint_satisfaction')
        assert isinstance(safety_metrics.total_violations, int)
        assert isinstance(safety_metrics.safety_score, float)


class TestSafetyCriticalAgents:
    """Test novel safety-critical RL algorithms."""
    
    def test_risk_aware_cql_creation(self):
        """Test Risk-Aware CQL agent creation and basic functionality."""
        safety_constraints = [
            SafetyConstraint(
                name="test_constraint",
                constraint_fn=lambda state: state[0] < 400.0,
                violation_penalty=-100.0
            )
        ]
        
        agent = RiskAwareCQLAgent(
            state_dim=20,
            action_dim=6,
            safety_constraints=safety_constraints
        )
        
        assert agent is not None
        assert len(agent.safety_constraints) == 1
        assert hasattr(agent, 'safety_critic')
        assert hasattr(agent, 'risk_estimator')
        
        # Test safety violation probability computation
        state = jnp.ones(20)
        action = jnp.ones(6)
        
        violation_prob = agent.compute_safety_violation_probability(state, action)
        assert isinstance(violation_prob, jnp.ndarray)
        assert jnp.all(violation_prob >= 0.0)
        assert jnp.all(violation_prob <= 1.0)
        
    def test_constrained_iql_creation(self):
        """Test Constrained IQL agent creation and functionality."""
        safety_constraints = [
            SafetyConstraint(
                name="test_constraint",
                constraint_fn=lambda state: state[0] < 400.0,
                violation_penalty=-100.0
            )
        ]
        
        agent = ConstrainedIQLAgent(
            state_dim=20,
            action_dim=6,
            safety_constraints=safety_constraints
        )
        
        assert agent is not None
        assert hasattr(agent, 'lagrange_multipliers')
        assert hasattr(agent, 'constraint_predictor')
        
        # Test constraint violation computation
        state = jnp.ones(20)
        action = jnp.ones(6)
        
        violations = agent.compute_safety_violation_probability(state, action)
        assert isinstance(violations, jnp.ndarray)
        
    def test_safe_action_generation(self):
        """Test safe action generation functionality."""
        from neorl_industrial.agents.safety_critical import SafeEnsembleAgent
        
        safety_constraints = [
            SafetyConstraint(
                name="test_constraint",
                constraint_fn=lambda state: jnp.all(state > -10.0),
                violation_penalty=-100.0
            )
        ]
        
        agent = SafeEnsembleAgent(
            state_dim=10,
            action_dim=3,
            safety_constraints=safety_constraints,
            n_models=3
        )
        
        state = jnp.ones(10)
        preferred_action = jnp.ones(3) * 5.0  # Potentially unsafe
        
        safe_action, metadata = agent.get_safe_action(state, preferred_action)
        
        assert safe_action.shape == (3,)
        assert isinstance(metadata, dict)
        assert "violation_prob" in metadata
        assert "uncertainty" in metadata


class TestIndustrialDataIntegration:
    """Test real-world data integration capabilities."""
    
    def test_plc_loader_creation(self):
        """Test PLC data loader creation (mock)."""
        # This would normally require actual PLC connection
        # For testing, we check the factory function
        try:
            loader = create_plc_loader("192.168.1.100", port=4840)
            assert loader is not None
            assert hasattr(loader, 'connector')
        except Exception as e:
            # Expected in test environment without actual PLC
            assert "connection" in str(e).lower() or "import" in str(e).lower()
            
    def test_data_quality_validation(self):
        """Test data quality validation functionality."""
        from neorl_industrial.data.industrial_data_loader import DataQualityMetrics
        
        # Create sample quality metrics
        metrics = DataQualityMetrics(
            completeness=0.95,
            timeliness=0.88,
            accuracy=0.92,
            consistency=0.90,
            integrity=0.98
        )
        
        assert metrics.completeness == 0.95
        assert metrics.timeliness == 0.88
        assert metrics.accuracy == 0.92
        assert metrics.consistency == 0.90
        assert metrics.integrity == 0.98


class TestBenchmarkingSuite:
    """Test comprehensive benchmarking framework."""
    
    def test_benchmark_config_creation(self):
        """Test benchmark configuration."""
        config = BenchmarkConfig(
            n_episodes=10,
            n_seeds=3,
            confidence_level=0.95
        )
        
        assert config.n_episodes == 10
        assert config.n_seeds == 3
        assert config.confidence_level == 0.95
        
    def test_benchmark_suite_creation(self):
        """Test benchmark suite creation."""
        config = BenchmarkConfig(n_episodes=5, n_seeds=2)
        suite = IndustrialBenchmarkSuite(config)
        
        assert suite is not None
        assert len(suite.benchmarks) == 4  # Safety, Performance, Scalability, Robustness
        
    def test_benchmark_evaluation_structure(self):
        """Test benchmark evaluation structure (without full run)."""
        from neorl_industrial.benchmarks.industrial_benchmarks import BenchmarkResult
        
        # Test result structure
        result = BenchmarkResult(
            benchmark_name="Test",
            agent_name="TestAgent",
            environment_name="TestEnv",
            mean_return=100.0,
            std_return=10.0,
            median_return=98.0,
            min_return=80.0,
            max_return=120.0,
            safety_violations=5,
            violation_rate=0.05,
            safety_score=95.0,
            episode_length=200.0,
            convergence_time=50.0,
            sample_efficiency=0.5,
            confidence_interval=(90.0, 110.0),
            statistical_significance=0.01,
            metadata={}
        )
        
        assert result.benchmark_name == "Test"
        assert result.mean_return == 100.0
        assert result.safety_violations == 5


class TestResearchValidation:
    """Test research validation framework."""
    
    def test_validation_framework_creation(self):
        """Test research validation framework initialization."""
        framework = ResearchValidationFramework(
            experiment_name="test_experiment",
            random_seed=42
        )
        
        assert framework.experiment_name == "test_experiment"
        assert framework.random_seed == 42
        assert framework.output_dir.exists()
        
    def test_hypothesis_validation_structure(self):
        """Test hypothesis validation structure."""
        framework = ResearchValidationFramework("test", random_seed=42)
        
        # Test H1 validation (simplified)
        result = framework._validate_h1_quality_gates()
        
        assert "hypothesis" in result
        assert "statistical_tests" in result
        assert "conclusion" in result
        assert result["hypothesis"] == "H1_progressive_quality_gates"
        
    def test_statistical_analysis_components(self):
        """Test statistical analysis components."""
        framework = ResearchValidationFramework("test", random_seed=42)
        
        # Test H4 validation (performance)
        result = framework._validate_h4_jax_performance()
        
        assert "performance_metrics" in result
        assert "statistical_tests" in result
        assert "jax_pytorch_speedup" in result["performance_metrics"]
        assert result["performance_metrics"]["jax_pytorch_speedup"] > 1.0


class TestSystemIntegration:
    """Test end-to-end system integration."""
    
    def test_complete_workflow_simulation(self):
        """Test complete research workflow (simplified)."""
        # 1. Create environment
        env = ni.make('AdvancedChemicalReactor-v0')
        
        # 2. Create safety-aware agent
        safety_constraints = [
            SafetyConstraint(
                name="temperature",
                constraint_fn=lambda state: state[0] < 400.0,
                violation_penalty=-100.0
            )
        ]
        
        agent = RiskAwareCQLAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            safety_constraints=safety_constraints
        )
        
        # 3. Test environment interaction
        obs, _ = env.reset()
        action = jnp.zeros(env.action_dim)  # Safe action
        
        # Test safety assessment
        violation_prob = agent.compute_safety_violation_probability(obs, action)
        assert isinstance(violation_prob, jnp.ndarray)
        
        # Test environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert next_obs.shape == obs.shape
        
        # Test safety metrics
        safety_metrics = env.get_safety_metrics()
        assert hasattr(safety_metrics, 'total_violations')
        
    def test_quality_gates_integration(self):
        """Test quality gates system integration."""
        from neorl_industrial.quality_gates import AdaptiveQualityGates
        
        # Create quality gates system
        gates = AdaptiveQualityGates(
            initial_thresholds={"performance": 80.0, "safety": 90.0}
        )
        
        # Test threshold adaptation
        metrics = jnp.array([85.0, 88.0, 92.0, 87.0, 90.0])
        gates.update_thresholds(metrics, strategy="percentile")
        
        assert hasattr(gates, 'thresholds')
        
    def test_academic_reproducibility(self):
        """Test academic reproducibility features."""
        # Test deterministic behavior with fixed seed
        np.random.seed(42)
        
        # Create identical environments
        env1 = ni.make('AdvancedChemicalReactor-v0')
        env2 = ni.make('AdvancedChemicalReactor-v0')
        
        # Reset with same seed
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        # Should produce identical observations
        assert jnp.allclose(obs1, obs2, atol=1e-6)
        
        # Test same action produces same result
        action = jnp.ones(6) * 0.1
        
        next_obs1, reward1, _, _, _ = env1.step(action)
        next_obs2, reward2, _, _, _ = env2.step(action)
        
        assert jnp.allclose(next_obs1, next_obs2, atol=1e-6)
        assert abs(reward1 - reward2) < 1e-6


@pytest.mark.integration
class TestFullSystemValidation:
    """Full system integration tests for research publication."""
    
    def test_research_paper_reproducibility(self):
        """Test research paper results reproducibility."""
        # This would run the full validation protocol
        # For testing, we verify the structure exists
        
        framework = ResearchValidationFramework("reproducibility_test", random_seed=42)
        
        # Test hypothesis validation exists
        assert hasattr(framework, 'run_hypothesis_validation')
        assert hasattr(framework, 'run_comprehensive_validation')
        
        # Test output generation
        assert hasattr(framework, '_generate_publication_summary')
        
    def test_benchmark_paper_metrics(self):
        """Test that benchmark produces paper-quality metrics."""
        config = BenchmarkConfig(n_episodes=5, n_seeds=2)  # Reduced for testing
        suite = IndustrialBenchmarkSuite(config)
        
        # Verify benchmark structure matches paper requirements
        assert len(suite.benchmarks) == 4
        
        benchmark_names = [b.__class__.__name__ for b in suite.benchmarks]
        expected_names = ["SafetyBenchmark", "PerformanceBenchmark", 
                         "ScalabilityBenchmark", "RobustnessBenchmark"]
        
        for name in expected_names:
            assert name in benchmark_names


if __name__ == "__main__":
    # Run tests when script executed directly
    pytest.main([__file__, "-v"])