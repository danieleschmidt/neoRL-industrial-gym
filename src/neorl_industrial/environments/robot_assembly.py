"""Robot assembly environment with safety constraints."""

import numpy as np
from typing import Dict, List, Optional

from .base import IndustrialEnv
from ..core.types import Array, StateArray, ActionArray, SafetyConstraint


def force_constraint(state: StateArray, action: ActionArray) -> bool:
    """Ensure contact forces don't exceed safe limits."""
    # Assume state[18:21] are contact forces in x, y, z
    contact_forces = state[18:21]
    max_force = 50.0  # Newtons
    return np.all(np.abs(contact_forces) < max_force)


def collision_constraint(state: StateArray, action: ActionArray) -> bool:
    """Check for potential collisions."""
    # Simplified collision check based on position
    position = state[0:3]
    # Define safe workspace boundaries
    workspace_min = np.array([-0.5, -0.5, 0.0])
    workspace_max = np.array([0.5, 0.5, 0.8])
    return np.all((position >= workspace_min) & (position <= workspace_max))


def velocity_constraint(state: StateArray, action: ActionArray) -> bool:
    """Ensure velocities don't exceed safe limits."""
    velocities = state[7:14]  # Joint velocities
    max_velocity = 2.0  # rad/s
    return np.all(np.abs(velocities) < max_velocity)


class RobotAssemblyEnv(IndustrialEnv):
    """Robot assembly environment.
    
    State space (24D):
    - [0:3]: End-effector position (x, y, z)
    - [3:7]: End-effector orientation (quaternion)
    - [7:14]: Joint positions (7-DOF robot)
    - [14:18]: End-effector velocity (linear + angular)
    - [18:21]: Contact forces (x, y, z)
    - [21:24]: Assembly status (part alignment, insertion depth, completion)
    
    Action space (7D): Joint velocity commands
    
    Rewards based on:
    - Task completion
    - Force compliance
    - Trajectory smoothness
    - Safety constraint satisfaction
    """
    
    def __init__(self, **kwargs):
        safety_constraints = [
            SafetyConstraint(
                name="force_limits",
                check_fn=force_constraint,
                penalty=-100.0,
                critical=True
            ),
            SafetyConstraint(
                name="collision_avoidance",
                check_fn=collision_constraint,
                penalty=-200.0,
                critical=True
            ),
            SafetyConstraint(
                name="velocity_limits",
                check_fn=velocity_constraint,
                penalty=-50.0,
                critical=False
            ),
        ]
        
        super().__init__(
            state_dim=24,
            action_dim=7,
            safety_constraints=safety_constraints,
            **kwargs
        )
        
        # Robot parameters
        self.link_lengths = np.array([0.3, 0.3, 0.25, 0.25, 0.15, 0.1, 0.05])
        self.joint_limits_low = np.array([-np.pi] * 7)
        self.joint_limits_high = np.array([np.pi] * 7)
        
        # Assembly task parameters
        self.target_position = np.array([0.3, 0.0, 0.4])  # Target assembly point
        self.insertion_depth = 0.05  # Required insertion depth
        self.alignment_tolerance = 0.005  # 5mm tolerance
    
    def _forward_kinematics(self, joint_positions: Array) -> tuple:
        """Simple forward kinematics for 7-DOF robot."""
        # Simplified FK - in practice would use proper DH parameters
        x = 0.0
        y = 0.0
        z = 0.0
        
        for i, (angle, length) in enumerate(zip(joint_positions, self.link_lengths)):
            if i % 2 == 0:  # Even joints contribute to x,z
                x += length * np.cos(angle)
                z += length * np.sin(angle)
            else:  # Odd joints contribute to y
                y += length * np.sin(angle)
        
        # Simplified orientation (would be proper rotation matrix)
        quat = np.array([0, 0, 0, 1])  # Identity quaternion
        
        return np.array([x, y, z]), quat
    
    def _get_initial_state(self) -> StateArray:
        """Initialize robot assembly state."""
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Random initial joint positions
        joint_positions = np.random.uniform(
            self.joint_limits_low * 0.5,
            self.joint_limits_high * 0.5,
            7
        )
        
        # Forward kinematics
        position, orientation = self._forward_kinematics(joint_positions)
        
        state[0:3] = position
        state[3:7] = orientation
        state[7:14] = joint_positions
        # Velocities start at zero
        state[14:18] = 0.0
        # No initial contact forces
        state[18:21] = 0.0
        # Assembly status: alignment, depth, completion
        state[21:24] = [0.0, 0.0, 0.0]
        
        return state
    
    def _dynamics(self, state: StateArray, action: ActionArray) -> StateArray:
        """Robot dynamics simulation."""
        next_state = state.copy()
        
        # Current joint positions and velocities
        joint_positions = state[7:14]
        joint_velocities = action  # Direct velocity control
        
        # Integrate joint positions
        new_joint_positions = joint_positions + joint_velocities * self.dt
        new_joint_positions = np.clip(
            new_joint_positions,
            self.joint_limits_low,
            self.joint_limits_high
        )
        
        # Forward kinematics for new position
        new_position, new_orientation = self._forward_kinematics(new_joint_positions)
        
        # End-effector velocity (simplified)
        old_position = state[0:3]
        ee_velocity = (new_position - old_position) / self.dt
        
        # Contact force simulation (simplified)
        distance_to_target = np.linalg.norm(new_position - self.target_position)
        if distance_to_target < 0.01:  # In contact
            # Simulate contact forces
            normal_force = max(0, 0.01 - distance_to_target) * 1000  # Spring model
            contact_forces = np.array([0, 0, -normal_force])
        else:
            contact_forces = np.zeros(3)
        
        # Assembly status
        alignment_error = np.linalg.norm(new_position[:2] - self.target_position[:2])
        alignment_score = max(0, 1.0 - alignment_error / self.alignment_tolerance)
        
        insertion_depth = max(0, self.target_position[2] - new_position[2])
        depth_score = min(1.0, insertion_depth / self.insertion_depth)
        
        completion_score = alignment_score * depth_score
        
        # Update state
        next_state[0:3] = new_position
        next_state[3:7] = new_orientation
        next_state[7:14] = new_joint_positions
        next_state[14:18] = np.concatenate([ee_velocity, [0]])  # Simplified angular vel
        next_state[18:21] = contact_forces
        next_state[21:24] = [alignment_score, depth_score, completion_score]
        
        return next_state
    
    def _compute_reward(self, state: StateArray, action: ActionArray, next_state: StateArray) -> float:
        """Compute reward for assembly task."""
        position = next_state[0:3]
        contact_forces = next_state[18:21]
        assembly_status = next_state[21:24]
        
        # Task completion reward
        completion_reward = 100 * assembly_status[2]  # Completion score
        
        # Distance reward
        distance_to_target = np.linalg.norm(position - self.target_position)
        distance_reward = -10 * distance_to_target
        
        # Force compliance reward
        force_magnitude = np.linalg.norm(contact_forces)
        if force_magnitude > 30:  # Excessive force penalty
            force_reward = -50 * (force_magnitude - 30)
        else:
            force_reward = 0
        
        # Action smoothness
        action_penalty = -0.1 * np.sum(action**2)
        
        # Velocity penalty (encourage smooth motion)
        velocities = next_state[14:18]
        velocity_penalty = -0.5 * np.sum(velocities**2)
        
        total_reward = (
            completion_reward + distance_reward + force_reward +
            action_penalty + velocity_penalty
        )
        
        return float(total_reward)
    
    def _is_done(self, state: StateArray) -> bool:
        """Check termination conditions."""
        assembly_status = state[21:24]
        contact_forces = state[18:21]
        position = state[0:3]
        
        # Task completed
        if assembly_status[2] > 0.95:
            return True
        
        # Excessive contact forces
        if np.any(np.abs(contact_forces) > 80):
            return True
        
        # Out of workspace
        workspace_min = np.array([-0.6, -0.6, -0.1])
        workspace_max = np.array([0.6, 0.6, 0.9])
        if not np.all((position >= workspace_min) & (position <= workspace_max)):
            return True
        
        return False
    
    def get_dataset(self, quality: str = "mixed") -> Dict[str, Array]:
        """Generate or load offline dataset."""
        n_samples = {
            "expert": 120000,
            "medium": 180000,
            "mixed": 250000,
            "random": 100000,
        }[quality]
        
        observations = []
        actions = []
        rewards = []
        terminals = []
        
        for _ in range(n_samples // 1000):
            obs, _ = self.reset()
            done = False
            episode_length = 0
            
            while not done and episode_length < 1000:
                if quality == "expert":
                    # Simple PD controller toward target
                    current_pos = obs[0:3]
                    error = self.target_position - current_pos
                    joint_vels = obs[7:14]
                    
                    # Proportional control
                    kp = 2.0
                    action = kp * error[:3]
                    # Add joint damping
                    action = np.concatenate([action, -0.1 * joint_vels[3:]])
                    action = action[:7]
                elif quality == "random":
                    action = np.random.uniform(-1, 1, self.action_dim)
                else:
                    # Mixed quality
                    if np.random.rand() < 0.7:
                        # Decent policy
                        current_pos = obs[0:3]
                        error = self.target_position - current_pos
                        action = 1.0 * error[:3]
                        action = np.concatenate([action, np.random.uniform(-0.5, 0.5, 4)])
                    else:
                        action = np.random.uniform(-0.8, 0.8, self.action_dim)
                
                action = np.clip(action, -2.0, 2.0)
                
                observations.append(obs.copy())
                actions.append(action)
                
                obs, reward, terminated, truncated, _ = self.step(action)
                rewards.append(reward)
                terminals.append(terminated)
                
                done = terminated or truncated
                episode_length += 1
        
        return {
            "observations": np.array(observations, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "terminals": np.array(terminals, dtype=bool),
        }
