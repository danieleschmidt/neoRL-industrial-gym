"""Integration tests for environment and agent interactions."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from tests.fixtures.industrial_data import IndustrialDataGenerator, IndustrialDataset


class MockEnvironment:
    """Mock industrial environment for testing."""
    
    def __init__(self, env_name: str, state_dim: int, action_dim: int):
        self.env_name = env_name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.current_state = np.zeros(state_dim, dtype=np.float32)
        self.episode_step = 0
        self.max_episode_steps = 1000
        self.safety_violations = []
        
    def reset(self):
        """Reset environment to initial state."""
        self.current_state = np.random.randn(self.state_dim).astype(np.float32)
        self.episode_step = 0
        self.safety_violations.clear()
        return self.current_state
    
    def step(self, action: np.ndarray):
        """Take a step in the environment."""
        # Simple dynamics simulation
        noise = np.random.randn(self.state_dim) * 0.01
        self.current_state += 0.1 * action[:self.state_dim] + noise
        
        # Compute reward
        reward = -np.sum(np.square(action)) * 0.01  # Energy penalty
        reward += -np.sum(np.square(self.current_state)) * 0.001  # State penalty
        
        # Check safety constraints
        safety_violation = self._check_safety_constraints()
        if safety_violation:
            reward -= 10.0  # Large penalty for safety violation
            self.safety_violations.append(self.episode_step)
        
        # Check termination
        self.episode_step += 1
        done = (self.episode_step >= self.max_episode_steps) or safety_violation
        timeout = self.episode_step >= self.max_episode_steps
        
        info = {
            "safety_violation": safety_violation,
            "timeout": timeout,
            "episode_step": self.episode_step
        }
        
        return self.current_state.copy(), reward, done, info
    
    def _check_safety_constraints(self) -> bool:
        """Check if current state violates safety constraints."""
        # Mock safety checks based on environment type
        if "ChemicalReactor" in self.env_name:
            # Temperature and pressure limits
            if self.state_dim >= 2:
                temp_violation = self.current_state[0] > 5.0 or self.current_state[0] < -5.0
                pressure_violation = self.current_state[1] > 3.0
                return temp_violation or pressure_violation
        
        elif "RobotAssembly" in self.env_name:
            # Joint limit violations
            joint_limit = 2.0
            if self.state_dim >= 7:
                joint_violations = np.any(np.abs(self.current_state[:7]) > joint_limit)
                return joint_violations
        
        # Generic safety check - extreme states
        return np.any(np.abs(self.current_state) > 10.0)
    
    def get_dataset(self, quality: str = "medium", size: int = 1000):
        """Get offline dataset for the environment."""
        generator = IndustrialDataGenerator()
        
        if "ChemicalReactor" in self.env_name:
            return generator.generate_chemical_reactor_data(size)
        elif "RobotAssembly" in self.env_name:
            return generator.generate_robot_assembly_data(size)
        else:
            # Generic dataset
            observations = np.random.randn(size, self.state_dim).astype(np.float32)
            actions = np.random.randn(size, self.action_dim).astype(np.float32)
            rewards = np.random.randn(size).astype(np.float32)
            
            return {
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "next_observations": observations,  # Simplified
                "terminals": np.random.choice([True, False], size, p=[0.01, 0.99]),
                "timeouts": np.random.choice([True, False], size, p=[0.05, 0.95]),
            }


class MockAgent:
    """Mock RL agent for testing."""
    
    def __init__(self, state_dim: int, action_dim: int, algorithm: str = "CQL"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.algorithm = algorithm
        self.training_step = 0
        self.evaluation_step = 0
        
        # Mock policy parameters
        self.policy_weights = np.random.randn(state_dim, action_dim) * 0.1
        
    def predict(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action given state."""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        # Simple linear policy for testing
        action = np.dot(state, self.policy_weights)
        
        if not deterministic:
            action += np.random.randn(*action.shape) * 0.1
        
        return action.squeeze()
    
    def train_step(self, batch: dict) -> dict:
        """Perform one training step."""
        self.training_step += 1
        
        # Mock training metrics
        metrics = {
            "loss": np.random.rand() * 0.1 + 0.01,
            "q_value": np.random.rand() * 10,
            "policy_loss": np.random.rand() * 0.05,
            "step": self.training_step
        }
        
        # Simulate policy improvement
        self.policy_weights += np.random.randn(*self.policy_weights.shape) * 0.001
        
        return metrics
    
    def evaluate(self, env: MockEnvironment, n_episodes: int = 10) -> dict:
        """Evaluate agent performance."""
        episode_returns = []
        episode_lengths = []
        safety_violations = []
        
        for _ in range(n_episodes):
            state = env.reset()
            episode_return = 0
            episode_length = 0
            episode_safety_violations = 0
            
            done = False
            while not done:
                action = self.predict(state, deterministic=True)
                next_state, reward, done, info = env.step(action)
                
                episode_return += reward
                episode_length += 1
                
                if info.get("safety_violation", False):
                    episode_safety_violations += 1
                
                state = next_state
                
                # Prevent infinite episodes
                if episode_length > 2000:
                    break
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            safety_violations.append(episode_safety_violations)
        
        return {
            "mean_return": np.mean(episode_returns),
            "std_return": np.std(episode_returns),
            "mean_length": np.mean(episode_lengths),
            "safety_violations": np.sum(safety_violations),
            "safety_violation_rate": np.mean([v > 0 for v in safety_violations]),
            "episodes": n_episodes
        }


@pytest.fixture
def chemical_reactor_env():
    """Chemical reactor environment fixture."""
    return MockEnvironment("ChemicalReactor-v0", state_dim=12, action_dim=3)


@pytest.fixture
def robot_assembly_env():
    """Robot assembly environment fixture."""
    return MockEnvironment("RobotAssembly-v0", state_dim=24, action_dim=7)


@pytest.fixture
def cql_agent():
    """CQL agent fixture."""
    return MockAgent(state_dim=12, action_dim=3, algorithm="CQL")


@pytest.fixture
def robot_agent():
    """Robot control agent fixture."""
    return MockAgent(state_dim=24, action_dim=7, algorithm="TD3+BC")


class TestEnvironmentIntegration:
    """Test environment behavior and properties."""
    
    @pytest.mark.integration
    def test_environment_reset(self, chemical_reactor_env):
        """Test environment reset functionality."""
        initial_state = chemical_reactor_env.reset()
        
        assert isinstance(initial_state, np.ndarray)
        assert initial_state.shape == (12,)
        assert chemical_reactor_env.episode_step == 0
        assert len(chemical_reactor_env.safety_violations) == 0
    
    @pytest.mark.integration
    def test_environment_step(self, chemical_reactor_env):
        """Test environment step functionality."""
        state = chemical_reactor_env.reset()
        action = np.random.randn(3).astype(np.float32)
        
        next_state, reward, done, info = chemical_reactor_env.step(action)
        
        assert isinstance(next_state, np.ndarray)
        assert next_state.shape == (12,)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert "safety_violation" in info
        assert "episode_step" in info
    
    @pytest.mark.integration
    def test_episode_termination(self, chemical_reactor_env):
        """Test episode termination conditions."""
        chemical_reactor_env.max_episode_steps = 10  # Short episode for testing
        state = chemical_reactor_env.reset()
        
        episode_length = 0
        done = False
        
        while not done and episode_length < 20:  # Safety limit
            action = np.random.randn(3).astype(np.float32)
            state, reward, done, info = chemical_reactor_env.step(action)
            episode_length += 1
        
        assert done
        assert episode_length <= 10  # Should terminate within max steps
    
    @pytest.mark.integration
    def test_safety_constraint_violations(self, chemical_reactor_env):
        """Test safety constraint violation detection."""
        state = chemical_reactor_env.reset()
        
        # Create action that should cause safety violation
        extreme_action = np.array([10.0, 10.0, 10.0])  # Very large action
        
        violations_detected = 0
        for _ in range(100):  # Try multiple times
            state, reward, done, info = chemical_reactor_env.step(extreme_action)
            if info.get("safety_violation", False):
                violations_detected += 1
                break
            if done:
                state = chemical_reactor_env.reset()
        
        # Should detect at least one violation with extreme actions
        assert violations_detected > 0
    
    @pytest.mark.integration
    def test_dataset_generation(self, chemical_reactor_env):
        """Test offline dataset generation."""
        dataset = chemical_reactor_env.get_dataset(quality="medium", size=500)
        
        if isinstance(dataset, IndustrialDataset):
            assert dataset.observations.shape == (500, 12)
            assert dataset.actions.shape == (500, 3)
            assert dataset.rewards.shape == (500,)
            assert "environment" in dataset.metadata
        else:
            # Legacy format
            assert dataset["observations"].shape == (500, 12)
            assert dataset["actions"].shape == (500, 3)
            assert dataset["rewards"].shape == (500,)


class TestAgentEnvironmentInteraction:
    """Test agent-environment interaction patterns."""
    
    @pytest.mark.integration
    def test_agent_evaluation(self, chemical_reactor_env, cql_agent):
        """Test agent evaluation in environment."""
        eval_results = cql_agent.evaluate(chemical_reactor_env, n_episodes=5)
        
        assert "mean_return" in eval_results
        assert "safety_violations" in eval_results
        assert "safety_violation_rate" in eval_results
        assert eval_results["episodes"] == 5
        assert isinstance(eval_results["mean_return"], (int, float))
    
    @pytest.mark.integration
    def test_training_loop_integration(self, chemical_reactor_env, cql_agent):
        """Test basic training loop integration."""
        # Get dataset
        dataset = chemical_reactor_env.get_dataset(size=100)
        
        if isinstance(dataset, IndustrialDataset):
            batch = {
                "observations": dataset.observations[:32],
                "actions": dataset.actions[:32],
                "rewards": dataset.rewards[:32],
                "next_observations": dataset.next_observations[:32],
                "terminals": dataset.terminals[:32]
            }
        else:
            batch = {
                "observations": dataset["observations"][:32],
                "actions": dataset["actions"][:32],
                "rewards": dataset["rewards"][:32],
                "next_observations": dataset["next_observations"][:32],
                "terminals": dataset["terminals"][:32]
            }
        
        # Perform training steps
        initial_step = cql_agent.training_step
        metrics = cql_agent.train_step(batch)
        
        assert cql_agent.training_step == initial_step + 1
        assert "loss" in metrics
        assert "step" in metrics
    
    @pytest.mark.integration
    def test_robot_control_integration(self, robot_assembly_env, robot_agent):
        """Test robot control specific integration."""
        state = robot_assembly_env.reset()
        assert state.shape == (24,)  # Full robot state
        
        # Test action prediction
        action = robot_agent.predict(state)
        assert action.shape == (7,)  # Joint torques
        
        # Test environment step
        next_state, reward, done, info = robot_assembly_env.step(action)
        assert next_state.shape == (24,)
    
    @pytest.mark.integration
    def test_safety_monitoring_integration(self, chemical_reactor_env, cql_agent):
        """Test integration of safety monitoring during operation."""
        eval_results = cql_agent.evaluate(chemical_reactor_env, n_episodes=10)
        
        # Safety monitoring should be active
        assert "safety_violations" in eval_results
        assert "safety_violation_rate" in eval_results
        assert 0 <= eval_results["safety_violation_rate"] <= 1
        
        # Check that safety violations affect performance
        if eval_results["safety_violations"] > 0:
            # If there were violations, performance should be impacted
            assert eval_results["mean_return"] < 100  # Arbitrary threshold


class TestMultiEnvironmentIntegration:
    """Test integration across multiple environment types."""
    
    @pytest.mark.integration
    @pytest.mark.parametrize("env_config", [
        ("ChemicalReactor-v0", 12, 3),
        ("RobotAssembly-v0", 24, 7),
        ("HVACControl-v0", 18, 5),
    ])
    def test_agent_adaptability(self, env_config):
        """Test agent adaptation to different environment types."""
        env_name, state_dim, action_dim = env_config
        
        env = MockEnvironment(env_name, state_dim, action_dim)
        agent = MockAgent(state_dim, action_dim)
        
        # Test that agent can interact with different environments
        state = env.reset()
        action = agent.predict(state)
        next_state, reward, done, info = env.step(action)
        
        assert next_state.shape == (state_dim,)
        assert action.shape == (action_dim,)
        assert isinstance(reward, (int, float))
    
    @pytest.mark.integration
    def test_cross_environment_transfer(self):
        """Test potential for cross-environment knowledge transfer."""
        # Create similar environments
        env1 = MockEnvironment("ChemicalReactor-v0", 12, 3)
        env2 = MockEnvironment("ChemicalReactor-v1", 12, 3)  # Similar but different
        
        # Train agent on first environment
        agent = MockAgent(12, 3)
        dataset1 = env1.get_dataset(size=50)
        
        if isinstance(dataset1, IndustrialDataset):
            batch = {
                "observations": dataset1.observations[:32],
                "actions": dataset1.actions[:32],
                "rewards": dataset1.rewards[:32],
                "next_observations": dataset1.next_observations[:32],
                "terminals": dataset1.terminals[:32]
            }
        else:
            batch = {k: v[:32] for k, v in dataset1.items()}
        
        # Training steps
        for _ in range(10):
            agent.train_step(batch)
        
        # Test on both environments
        eval1 = agent.evaluate(env1, n_episodes=3)
        eval2 = agent.evaluate(env2, n_episodes=3)
        
        # Both evaluations should complete successfully
        assert "mean_return" in eval1
        assert "mean_return" in eval2


@pytest.mark.integration
@pytest.mark.slow
def test_long_episode_integration():
    """Test integration over long episodes."""
    env = MockEnvironment("ChemicalReactor-v0", 12, 3)
    agent = MockAgent(12, 3)
    
    env.max_episode_steps = 5000  # Long episode
    
    state = env.reset()
    total_reward = 0
    steps = 0
    
    done = False
    while not done and steps < 5000:
        action = agent.predict(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
    
    assert steps > 100  # Should run for reasonable duration
    assert isinstance(total_reward, (int, float))


@pytest.mark.integration
@pytest.mark.performance
def test_integration_performance():
    """Test performance of integrated system."""
    import time
    
    env = MockEnvironment("ChemicalReactor-v0", 12, 3)
    agent = MockAgent(12, 3)
    
    # Time evaluation
    start_time = time.time()
    eval_results = agent.evaluate(env, n_episodes=20)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    # Should complete evaluation in reasonable time
    assert elapsed_time < 10.0  # 10 seconds max for 20 episodes
    assert eval_results["episodes"] == 20