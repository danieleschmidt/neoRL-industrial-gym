"""Industry-specific connectors for real-world deployment and ecosystem integration."""

import time
import json
import asyncio
import threading
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Protocol
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import concurrent.futures
from collections import deque, defaultdict
from enum import Enum

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    JAX_AVAILABLE = False

from ..monitoring.logger import get_logger
from ..core.types import Array


class IndustryType(Enum):
    """Supported industry types."""
    CHEMICAL = "chemical"
    MANUFACTURING = "manufacturing"  
    ENERGY = "energy"
    AUTOMOTIVE = "automotive"
    AEROSPACE = "aerospace"
    PHARMACEUTICAL = "pharmaceutical"
    FOOD_BEVERAGE = "food_beverage"
    TEXTILE = "textile"
    MINING = "mining"
    OIL_GAS = "oil_gas"


class CommunicationProtocol(Enum):
    """Industrial communication protocols."""
    MODBUS = "modbus"
    OPC_UA = "opc_ua"
    MQTT = "mqtt"
    ETHERNET_IP = "ethernet_ip"
    PROFINET = "profinet"
    CANBUS = "canbus"
    HTTP_REST = "http_rest"
    WEBSOCKET = "websocket"


@dataclass
class IndustryConfig:
    """Configuration for industry-specific integration."""
    # Industry specification
    industry_type: IndustryType
    plant_identifier: str
    communication_protocol: CommunicationProtocol
    
    # Connection parameters
    host: str = "localhost"
    port: int = 502
    timeout: float = 5.0
    retry_attempts: int = 3
    
    # Data mapping
    state_variables: Dict[str, str] = None  # Maps RL state to PLC variables
    action_variables: Dict[str, str] = None  # Maps RL actions to PLC variables
    safety_variables: Dict[str, str] = None  # Critical safety variables
    
    # Safety configuration
    emergency_stop_enabled: bool = True
    safety_check_interval: float = 1.0
    max_action_deviation: float = 0.1
    
    # Performance optimization
    batch_communication: bool = True
    async_operations: bool = True
    caching_enabled: bool = True
    cache_duration: float = 0.1
    
    # Compliance
    data_logging_enabled: bool = True
    audit_trail: bool = True
    regulatory_compliance: List[str] = None  # ISO, FDA, etc.


class IndustrialConnector(ABC):
    """Abstract base class for industrial system connectors."""
    
    def __init__(self, config: IndustryConfig):
        self.config = config
        self.logger = get_logger(f"{config.industry_type.value}_connector")
        self.connection_active = False
        
        # Data caching
        self.state_cache = {}
        self.cache_timestamps = {}
        
        # Safety monitoring
        self.safety_violations = deque(maxlen=1000)
        self.emergency_stop_triggered = False
        
        # Performance metrics
        self.communication_metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'connection_errors': 0,
            'average_latency': 0.0,
            'uptime': 0.0
        }
        
        # Initialize industry-specific defaults
        self._initialize_industry_defaults()
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to industrial system."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from industrial system."""
        pass
    
    @abstractmethod
    def read_state(self) -> Dict[str, Any]:
        """Read current state from industrial system."""
        pass
    
    @abstractmethod
    def write_action(self, action: Dict[str, Any]) -> bool:
        """Write action to industrial system."""
        pass
    
    @abstractmethod
    def emergency_stop(self) -> bool:
        """Trigger emergency stop."""
        pass
    
    def _initialize_industry_defaults(self):
        """Initialize industry-specific default configurations."""
        if self.config.state_variables is None:
            self.config.state_variables = self._get_default_state_variables()
        
        if self.config.action_variables is None:
            self.config.action_variables = self._get_default_action_variables()
        
        if self.config.safety_variables is None:
            self.config.safety_variables = self._get_default_safety_variables()
        
        if self.config.regulatory_compliance is None:
            self.config.regulatory_compliance = self._get_default_compliance_standards()
    
    def _get_default_state_variables(self) -> Dict[str, str]:
        """Get default state variable mappings for industry type."""
        defaults = {
            IndustryType.CHEMICAL: {
                'temperature': 'AI_Temperature_001',
                'pressure': 'AI_Pressure_001',
                'flow_rate': 'AI_Flow_001',
                'level': 'AI_Level_001',
                'ph': 'AI_pH_001'
            },
            IndustryType.MANUFACTURING: {
                'conveyor_speed': 'AI_ConveyorSpeed_001',
                'motor_torque': 'AI_MotorTorque_001',
                'position': 'AI_Position_001',
                'vibration': 'AI_Vibration_001',
                'power_consumption': 'AI_Power_001'
            },
            IndustryType.ENERGY: {
                'voltage': 'AI_Voltage_001',
                'current': 'AI_Current_001',
                'frequency': 'AI_Frequency_001',
                'power_factor': 'AI_PowerFactor_001',
                'load': 'AI_Load_001'
            }
        }
        
        return defaults.get(self.config.industry_type, {
            'sensor_1': 'AI_Sensor_001',
            'sensor_2': 'AI_Sensor_002',
            'sensor_3': 'AI_Sensor_003'
        })
    
    def _get_default_action_variables(self) -> Dict[str, str]:
        """Get default action variable mappings for industry type."""
        defaults = {
            IndustryType.CHEMICAL: {
                'heater_setpoint': 'AO_HeaterSetpoint_001',
                'valve_position': 'AO_ValvePosition_001',
                'pump_speed': 'AO_PumpSpeed_001'
            },
            IndustryType.MANUFACTURING: {
                'motor_speed': 'AO_MotorSpeed_001',
                'actuator_position': 'AO_ActuatorPos_001',
                'gripper_force': 'AO_GripperForce_001'
            },
            IndustryType.ENERGY: {
                'generator_setpoint': 'AO_GenSetpoint_001',
                'transformer_tap': 'AO_TransformerTap_001',
                'switch_position': 'DO_Switch_001'
            }
        }
        
        return defaults.get(self.config.industry_type, {
            'actuator_1': 'AO_Actuator_001',
            'actuator_2': 'AO_Actuator_002',
            'actuator_3': 'AO_Actuator_003'
        })
    
    def _get_default_safety_variables(self) -> Dict[str, str]:
        """Get default safety variable mappings for industry type."""
        defaults = {
            IndustryType.CHEMICAL: {
                'emergency_stop': 'DI_EmergencyStop_001',
                'safety_valve': 'DO_SafetyValve_001',
                'alarm_active': 'DI_Alarm_001',
                'interlock': 'DI_Interlock_001'
            },
            IndustryType.MANUFACTURING: {
                'emergency_stop': 'DI_EmergencyStop_001',
                'safety_door': 'DI_SafetyDoor_001',
                'light_curtain': 'DI_LightCurtain_001',
                'safety_relay': 'DO_SafetyRelay_001'
            }
        }
        
        return defaults.get(self.config.industry_type, {
            'emergency_stop': 'DI_EmergencyStop_001',
            'safety_system': 'DI_SafetySystem_001'
        })
    
    def _get_default_compliance_standards(self) -> List[str]:
        """Get default compliance standards for industry type."""
        defaults = {
            IndustryType.CHEMICAL: ['ISO 9001', 'ISO 14001', 'OSHA Process Safety'],
            IndustryType.PHARMACEUTICAL: ['FDA CFR 21 Part 11', 'ISO 9001', 'GMP'],
            IndustryType.AUTOMOTIVE: ['ISO/TS 16949', 'ISO 26262', 'ISO 9001'],
            IndustryType.AEROSPACE: ['AS9100', 'ISO 9001', 'FAA Part 145'],
            IndustryType.FOOD_BEVERAGE: ['HACCP', 'FDA Food Safety', 'ISO 22000']
        }
        
        return defaults.get(self.config.industry_type, ['ISO 9001'])
    
    def check_safety_conditions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check safety conditions and constraints."""
        safety_status = {
            'overall_safe': True,
            'violations': [],
            'warnings': [],
            'emergency_action_required': False
        }
        
        # Check industry-specific safety conditions
        if self.config.industry_type == IndustryType.CHEMICAL:
            safety_status.update(self._check_chemical_safety(state))
        elif self.config.industry_type == IndustryType.MANUFACTURING:
            safety_status.update(self._check_manufacturing_safety(state))
        elif self.config.industry_type == IndustryType.ENERGY:
            safety_status.update(self._check_energy_safety(state))
        
        # Record violations
        if safety_status['violations']:
            violation_record = {
                'timestamp': time.time(),
                'violations': safety_status['violations'],
                'state': state.copy(),
                'plant_id': self.config.plant_identifier
            }
            self.safety_violations.append(violation_record)
        
        # Trigger emergency stop if critical
        if safety_status['emergency_action_required'] and self.config.emergency_stop_enabled:
            self.emergency_stop_triggered = True
            self.emergency_stop()
        
        return safety_status
    
    def _check_chemical_safety(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check chemical industry specific safety conditions."""
        violations = []
        warnings = []
        emergency_required = False
        
        # Temperature safety
        if 'temperature' in state:
            temp = state['temperature']
            if temp > 400:  # Critical temperature
                violations.append(f'Temperature critical: {temp}°C > 400°C')
                emergency_required = True
            elif temp > 350:  # Warning temperature
                warnings.append(f'Temperature warning: {temp}°C > 350°C')
        
        # Pressure safety
        if 'pressure' in state:
            pressure = state['pressure']
            if pressure > 15:  # Critical pressure
                violations.append(f'Pressure critical: {pressure} bar > 15 bar')
                emergency_required = True
            elif pressure > 12:  # Warning pressure
                warnings.append(f'Pressure warning: {pressure} bar > 12 bar')
        
        return {
            'violations': violations,
            'warnings': warnings,
            'emergency_action_required': emergency_required,
            'overall_safe': not violations
        }
    
    def _check_manufacturing_safety(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check manufacturing industry specific safety conditions."""
        violations = []
        warnings = []
        emergency_required = False
        
        # Vibration safety
        if 'vibration' in state:
            vibration = state['vibration']
            if vibration > 10:  # Critical vibration
                violations.append(f'Vibration critical: {vibration} mm/s > 10 mm/s')
                emergency_required = True
            elif vibration > 7:  # Warning vibration
                warnings.append(f'Vibration warning: {vibration} mm/s > 7 mm/s')
        
        # Power consumption safety
        if 'power_consumption' in state:
            power = state['power_consumption']
            if power > 2000:  # Critical power
                violations.append(f'Power consumption critical: {power} W > 2000 W')
                emergency_required = True
        
        return {
            'violations': violations,
            'warnings': warnings,
            'emergency_action_required': emergency_required,
            'overall_safe': not violations
        }
    
    def _check_energy_safety(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Check energy industry specific safety conditions."""
        violations = []
        warnings = []
        emergency_required = False
        
        # Voltage safety
        if 'voltage' in state:
            voltage = state['voltage']
            if voltage > 500 or voltage < 200:  # Critical voltage range
                violations.append(f'Voltage critical: {voltage} V outside safe range')
                emergency_required = True
        
        # Frequency safety
        if 'frequency' in state:
            frequency = state['frequency']
            if abs(frequency - 50) > 2:  # Critical frequency deviation
                violations.append(f'Frequency critical: {frequency} Hz deviates > 2 Hz from 50 Hz')
                emergency_required = True
        
        return {
            'violations': violations,
            'warnings': warnings,
            'emergency_action_required': emergency_required,
            'overall_safe': not violations
        }


class ModbusConnector(IndustrialConnector):
    """Modbus industrial connector implementation."""
    
    def __init__(self, config: IndustryConfig):
        super().__init__(config)
        self.client = None
        self.register_map = {}
        
        # Modbus-specific initialization
        self._initialize_register_map()
    
    def _initialize_register_map(self):
        """Initialize Modbus register mapping."""
        # Map variable names to Modbus addresses
        register_counter = 1
        
        for var_name in self.config.state_variables.keys():
            self.register_map[var_name] = register_counter
            register_counter += 1
        
        for var_name in self.config.action_variables.keys():
            self.register_map[var_name] = register_counter
            register_counter += 1
    
    def connect(self) -> bool:
        """Establish Modbus connection."""
        try:
            # Simulated Modbus connection
            self.logger.info(f"Connecting to Modbus device at {self.config.host}:{self.config.port}")
            
            # In real implementation, would use pymodbus:
            # from pymodbus.client.sync import ModbusTcpClient
            # self.client = ModbusTcpClient(self.config.host, port=self.config.port)
            # connection_result = self.client.connect()
            
            # Simulated successful connection
            self.client = {"connected": True}  # Mock client
            self.connection_active = True
            
            self.logger.info("Modbus connection established")
            return True
            
        except Exception as e:
            self.logger.error(f"Modbus connection failed: {e}")
            self.communication_metrics['connection_errors'] += 1
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Modbus device."""
        try:
            if self.client:
                # In real implementation: self.client.close()
                self.client = None
            
            self.connection_active = False
            self.logger.info("Modbus connection closed")
            return True
            
        except Exception as e:
            self.logger.error(f"Modbus disconnect failed: {e}")
            return False
    
    def read_state(self) -> Dict[str, Any]:
        """Read state from Modbus registers."""
        if not self.connection_active:
            return {}
        
        start_time = time.time()
        state = {}
        
        try:
            # Check cache first
            if self.config.caching_enabled:
                cache_valid = self._is_cache_valid()
                if cache_valid:
                    return self.state_cache
            
            # Read state variables
            for var_name, plc_address in self.config.state_variables.items():
                register_addr = self.register_map.get(var_name, 1)
                
                # In real implementation:
                # result = self.client.read_holding_registers(register_addr, 1)
                # value = result.registers[0] if not result.isError() else 0
                
                # Simulated sensor readings
                value = self._simulate_sensor_reading(var_name)
                state[var_name] = value
            
            # Update cache
            if self.config.caching_enabled:
                self.state_cache = state.copy()
                self.cache_timestamps['state'] = time.time()
            
            # Update metrics
            self.communication_metrics['messages_received'] += 1
            latency = time.time() - start_time
            self._update_latency_metric(latency)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Modbus read failed: {e}")
            self.communication_metrics['connection_errors'] += 1
            return {}
    
    def write_action(self, action: Dict[str, Any]) -> bool:
        """Write action to Modbus registers."""
        if not self.connection_active:
            return False
        
        try:
            start_time = time.time()
            
            # Safety check before writing
            current_state = self.read_state()
            safety_status = self.check_safety_conditions(current_state)
            
            if not safety_status['overall_safe']:
                self.logger.warning(f"Action blocked due to safety violations: {safety_status['violations']}")
                return False
            
            # Write action variables
            for var_name, value in action.items():
                if var_name in self.config.action_variables:
                    plc_address = self.config.action_variables[var_name]
                    register_addr = self.register_map.get(var_name, 100)
                    
                    # Convert and validate value
                    modbus_value = self._convert_to_modbus_value(value)
                    
                    # In real implementation:
                    # self.client.write_register(register_addr, modbus_value)
                    
                    # Simulated write
                    self.logger.debug(f"Writing {var_name}={modbus_value} to register {register_addr}")
            
            # Update metrics
            self.communication_metrics['messages_sent'] += 1
            latency = time.time() - start_time
            self._update_latency_metric(latency)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Modbus write failed: {e}")
            self.communication_metrics['connection_errors'] += 1
            return False
    
    def emergency_stop(self) -> bool:
        """Trigger emergency stop via Modbus."""
        try:
            self.logger.critical("EMERGENCY STOP TRIGGERED")
            
            # Write emergency stop signal
            emergency_register = self.register_map.get('emergency_stop', 999)
            
            # In real implementation:
            # self.client.write_register(emergency_register, 1)
            
            # Simulated emergency stop
            self.emergency_stop_triggered = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Emergency stop failed: {e}")
            return False
    
    def _simulate_sensor_reading(self, var_name: str) -> float:
        """Simulate realistic sensor readings."""
        base_values = {
            'temperature': 300.0,
            'pressure': 5.0,
            'flow_rate': 50.0,
            'level': 0.7,
            'ph': 7.0,
            'conveyor_speed': 1.5,
            'motor_torque': 75.0,
            'position': 0.5,
            'vibration': 2.0,
            'power_consumption': 800.0,
            'voltage': 230.0,
            'current': 5.0,
            'frequency': 50.0,
            'power_factor': 0.85,
            'load': 0.6
        }
        
        base_value = base_values.get(var_name, 1.0)
        
        # Add realistic noise and variation
        import random
        noise = random.uniform(-0.05, 0.05)  # ±5% noise
        return base_value * (1 + noise)
    
    def _convert_to_modbus_value(self, value: float) -> int:
        """Convert floating point value to Modbus register value."""
        # Scale and convert to integer (typical for Modbus)
        scaled_value = int(value * 100)  # 2 decimal places
        return max(0, min(65535, scaled_value))  # Clamp to 16-bit range
    
    def _is_cache_valid(self) -> bool:
        """Check if cached state is still valid."""
        if 'state' not in self.cache_timestamps:
            return False
        
        cache_age = time.time() - self.cache_timestamps['state']
        return cache_age < self.config.cache_duration
    
    def _update_latency_metric(self, latency: float):
        """Update average latency metric."""
        current_avg = self.communication_metrics['average_latency']
        message_count = (self.communication_metrics['messages_sent'] + 
                        self.communication_metrics['messages_received'])
        
        if message_count > 0:
            self.communication_metrics['average_latency'] = (
                (current_avg * (message_count - 1) + latency) / message_count
            )


class OPCUAConnector(IndustrialConnector):
    """OPC UA industrial connector implementation."""
    
    def __init__(self, config: IndustryConfig):
        super().__init__(config)
        self.client = None
        self.node_map = {}
        
        # OPC UA specific initialization
        self._initialize_node_map()
    
    def _initialize_node_map(self):
        """Initialize OPC UA node mapping."""
        # Map variable names to OPC UA node IDs
        node_counter = 1000
        
        for var_name in self.config.state_variables.keys():
            self.node_map[var_name] = f"ns=2;i={node_counter}"
            node_counter += 1
        
        for var_name in self.config.action_variables.keys():
            self.node_map[var_name] = f"ns=2;i={node_counter}"
            node_counter += 1
    
    def connect(self) -> bool:
        """Establish OPC UA connection."""
        try:
            self.logger.info(f"Connecting to OPC UA server at opc.tcp://{self.config.host}:{self.config.port}")
            
            # In real implementation, would use opcua library:
            # from opcua import Client
            # self.client = Client(f"opc.tcp://{self.config.host}:{self.config.port}")
            # self.client.connect()
            
            # Simulated successful connection
            self.client = {"connected": True, "endpoint": f"opc.tcp://{self.config.host}:{self.config.port}"}
            self.connection_active = True
            
            self.logger.info("OPC UA connection established")
            return True
            
        except Exception as e:
            self.logger.error(f"OPC UA connection failed: {e}")
            self.communication_metrics['connection_errors'] += 1
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from OPC UA server."""
        try:
            if self.client:
                # In real implementation: self.client.disconnect()
                self.client = None
            
            self.connection_active = False
            self.logger.info("OPC UA connection closed")
            return True
            
        except Exception as e:
            self.logger.error(f"OPC UA disconnect failed: {e}")
            return False
    
    def read_state(self) -> Dict[str, Any]:
        """Read state from OPC UA nodes."""
        if not self.connection_active:
            return {}
        
        start_time = time.time()
        state = {}
        
        try:
            # Check cache first
            if self.config.caching_enabled:
                cache_valid = self._is_cache_valid()
                if cache_valid:
                    return self.state_cache
            
            # Read state variables
            for var_name, plc_address in self.config.state_variables.items():
                node_id = self.node_map.get(var_name)
                
                # In real implementation:
                # node = self.client.get_node(node_id)
                # value = node.get_value()
                
                # Simulated sensor readings
                value = self._simulate_sensor_reading(var_name)
                state[var_name] = value
            
            # Update cache
            if self.config.caching_enabled:
                self.state_cache = state.copy()
                self.cache_timestamps['state'] = time.time()
            
            # Update metrics
            self.communication_metrics['messages_received'] += 1
            latency = time.time() - start_time
            self._update_latency_metric(latency)
            
            return state
            
        except Exception as e:
            self.logger.error(f"OPC UA read failed: {e}")
            self.communication_metrics['connection_errors'] += 1
            return {}
    
    def write_action(self, action: Dict[str, Any]) -> bool:
        """Write action to OPC UA nodes."""
        if not self.connection_active:
            return False
        
        try:
            start_time = time.time()
            
            # Safety check before writing
            current_state = self.read_state()
            safety_status = self.check_safety_conditions(current_state)
            
            if not safety_status['overall_safe']:
                self.logger.warning(f"Action blocked due to safety violations: {safety_status['violations']}")
                return False
            
            # Write action variables
            for var_name, value in action.items():
                if var_name in self.config.action_variables:
                    node_id = self.node_map.get(var_name)
                    
                    # In real implementation:
                    # node = self.client.get_node(node_id)
                    # node.set_value(value)
                    
                    # Simulated write
                    self.logger.debug(f"Writing {var_name}={value} to node {node_id}")
            
            # Update metrics
            self.communication_metrics['messages_sent'] += 1
            latency = time.time() - start_time
            self._update_latency_metric(latency)
            
            return True
            
        except Exception as e:
            self.logger.error(f"OPC UA write failed: {e}")
            self.communication_metrics['connection_errors'] += 1
            return False
    
    def emergency_stop(self) -> bool:
        """Trigger emergency stop via OPC UA."""
        try:
            self.logger.critical("EMERGENCY STOP TRIGGERED")
            
            # Write emergency stop signal to all safety nodes
            emergency_node = self.node_map.get('emergency_stop')
            
            # In real implementation:
            # node = self.client.get_node(emergency_node)
            # node.set_value(True)
            
            # Simulated emergency stop
            self.emergency_stop_triggered = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Emergency stop failed: {e}")
            return False
    
    def _simulate_sensor_reading(self, var_name: str) -> float:
        """Simulate realistic sensor readings (same as Modbus)."""
        base_values = {
            'temperature': 300.0,
            'pressure': 5.0,
            'flow_rate': 50.0,
            'level': 0.7,
            'ph': 7.0,
            'conveyor_speed': 1.5,
            'motor_torque': 75.0,
            'position': 0.5,
            'vibration': 2.0,
            'power_consumption': 800.0,
            'voltage': 230.0,
            'current': 5.0,
            'frequency': 50.0,
            'power_factor': 0.85,
            'load': 0.6
        }
        
        base_value = base_values.get(var_name, 1.0)
        
        # Add realistic noise and variation
        import random
        noise = random.uniform(-0.05, 0.05)  # ±5% noise
        return base_value * (1 + noise)
    
    def _is_cache_valid(self) -> bool:
        """Check if cached state is still valid."""
        if 'state' not in self.cache_timestamps:
            return False
        
        cache_age = time.time() - self.cache_timestamps['state']
        return cache_age < self.config.cache_duration
    
    def _update_latency_metric(self, latency: float):
        """Update average latency metric."""
        current_avg = self.communication_metrics['average_latency']
        message_count = (self.communication_metrics['messages_sent'] + 
                        self.communication_metrics['messages_received'])
        
        if message_count > 0:
            self.communication_metrics['average_latency'] = (
                (current_avg * (message_count - 1) + latency) / message_count
            )


class IndustrialDeploymentManager:
    """Manager for industrial RL deployments across different industries."""
    
    def __init__(self):
        self.logger = get_logger("industrial_deployment_manager")
        
        # Active connections
        self.active_connectors = {}
        self.deployment_sessions = []
        
        # Performance tracking
        self.deployment_metrics = {
            'total_deployments': 0,
            'successful_connections': 0,
            'safety_incidents': 0,
            'uptime_percentage': 0.0,
            'average_performance': 0.0
        }
        
        # Regulatory compliance tracking
        self.compliance_records = defaultdict(list)
        
        self.logger.info("Industrial deployment manager initialized")
    
    def create_connector(self, config: IndustryConfig) -> IndustrialConnector:
        """Create appropriate connector based on communication protocol."""
        connector_classes = {
            CommunicationProtocol.MODBUS: ModbusConnector,
            CommunicationProtocol.OPC_UA: OPCUAConnector,
            # Additional protocols would be implemented here
        }
        
        connector_class = connector_classes.get(config.communication_protocol)
        if connector_class is None:
            raise ValueError(f"Unsupported communication protocol: {config.communication_protocol}")
        
        connector = connector_class(config)
        self.active_connectors[config.plant_identifier] = connector
        
        self.logger.info(f"Created {config.communication_protocol.value} connector for {config.plant_identifier}")
        return connector
    
    def deploy_rl_agent(self, agent, connector: IndustrialConnector, 
                       deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy RL agent to industrial system."""
        deployment_start = time.time()
        
        deployment_session = {
            'session_id': len(self.deployment_sessions),
            'plant_id': connector.config.plant_identifier,
            'industry_type': connector.config.industry_type.value,
            'start_time': deployment_start,
            'status': 'initializing',
            'performance_metrics': [],
            'safety_incidents': [],
            'compliance_events': []
        }
        
        try:
            # 1. Establish connection
            self.logger.info(f"Establishing connection to {connector.config.plant_identifier}")
            connection_success = connector.connect()
            
            if not connection_success:
                deployment_session['status'] = 'connection_failed'
                return deployment_session
            
            deployment_session['status'] = 'connected'
            self.deployment_metrics['successful_connections'] += 1
            
            # 2. Initial safety check
            initial_state = connector.read_state()
            safety_status = connector.check_safety_conditions(initial_state)
            
            if not safety_status['overall_safe']:
                self.logger.warning(f"Initial safety check failed: {safety_status['violations']}")
                deployment_session['status'] = 'safety_failure'
                deployment_session['safety_incidents'].append({
                    'timestamp': time.time(),
                    'type': 'initial_safety_check',
                    'violations': safety_status['violations']
                })
                return deployment_session
            
            # 3. Start deployment loop
            deployment_session['status'] = 'active'
            self.logger.info(f"Starting RL deployment for {connector.config.plant_identifier}")
            
            # Run deployment (simplified version)
            deployment_duration = deployment_config.get('duration', 3600)  # 1 hour default
            end_time = deployment_start + deployment_duration
            
            step_count = 0
            while time.time() < end_time and deployment_session['status'] == 'active':
                # Read current state
                current_state = connector.read_state()
                
                # Convert to RL state format
                rl_state = self._convert_to_rl_state(current_state, connector.config)
                
                # Get action from RL agent
                if hasattr(agent, 'get_action'):
                    rl_action = agent.get_action(rl_state)
                else:
                    # Default action if agent doesn't have get_action method
                    rl_action = [0.0] * len(connector.config.action_variables)
                
                # Convert to industrial action format
                industrial_action = self._convert_to_industrial_action(rl_action, connector.config)
                
                # Safety check
                safety_status = connector.check_safety_conditions(current_state)
                if not safety_status['overall_safe']:
                    self.logger.warning(f"Safety violation detected: {safety_status['violations']}")
                    deployment_session['safety_incidents'].append({
                        'timestamp': time.time(),
                        'step': step_count,
                        'violations': safety_status['violations'],
                        'state': current_state.copy()
                    })
                    
                    if safety_status['emergency_action_required']:
                        deployment_session['status'] = 'emergency_stopped'
                        break
                
                # Execute action
                action_success = connector.write_action(industrial_action)
                
                if not action_success:
                    self.logger.error("Action execution failed")
                    deployment_session['status'] = 'action_failure'
                    break
                
                # Record performance metrics
                performance_metric = {
                    'step': step_count,
                    'timestamp': time.time(),
                    'state': current_state.copy(),
                    'action': industrial_action.copy(),
                    'safety_status': safety_status['overall_safe']
                }
                deployment_session['performance_metrics'].append(performance_metric)
                
                step_count += 1
                
                # Sleep for control cycle
                control_cycle = deployment_config.get('control_cycle', 1.0)
                time.sleep(control_cycle)
            
            # 4. Finalize deployment
            deployment_session['end_time'] = time.time()
            deployment_session['total_steps'] = step_count
            deployment_session['duration'] = deployment_session['end_time'] - deployment_start
            
            if deployment_session['status'] == 'active':
                deployment_session['status'] = 'completed_successfully'
            
            # Disconnect
            connector.disconnect()
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            deployment_session['status'] = 'failed'
            deployment_session['error'] = str(e)
        
        finally:
            # Record deployment session
            self.deployment_sessions.append(deployment_session)
            self._update_deployment_metrics(deployment_session)
            
            # Generate compliance records
            self._generate_compliance_records(deployment_session, connector.config)
        
        return deployment_session
    
    def _convert_to_rl_state(self, industrial_state: Dict[str, Any], 
                           config: IndustryConfig) -> List[float]:
        """Convert industrial state to RL state format."""
        rl_state = []
        
        for var_name in config.state_variables.keys():
            value = industrial_state.get(var_name, 0.0)
            # Normalize based on typical ranges for each variable
            normalized_value = self._normalize_state_variable(var_name, value)
            rl_state.append(normalized_value)
        
        return rl_state
    
    def _normalize_state_variable(self, var_name: str, value: float) -> float:
        """Normalize state variable to [0, 1] range."""
        normalization_ranges = {
            'temperature': (200, 400),
            'pressure': (0, 20),
            'flow_rate': (0, 100),
            'level': (0, 1),
            'ph': (0, 14),
            'conveyor_speed': (0, 5),
            'motor_torque': (0, 150),
            'position': (0, 1),
            'vibration': (0, 20),
            'power_consumption': (0, 2000),
            'voltage': (0, 500),
            'current': (0, 20),
            'frequency': (45, 55),
            'power_factor': (0, 1),
            'load': (0, 1)
        }
        
        min_val, max_val = normalization_ranges.get(var_name, (0, 1))
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    def _convert_to_industrial_action(self, rl_action: List[float], 
                                    config: IndustryConfig) -> Dict[str, Any]:
        """Convert RL action to industrial action format."""
        industrial_action = {}
        
        for i, (var_name, plc_address) in enumerate(config.action_variables.items()):
            if i < len(rl_action):
                # Denormalize action value based on variable type
                denormalized_value = self._denormalize_action_variable(var_name, rl_action[i])
                industrial_action[var_name] = denormalized_value
        
        return industrial_action
    
    def _denormalize_action_variable(self, var_name: str, normalized_value: float) -> float:
        """Denormalize action variable from [0, 1] to actual range."""
        action_ranges = {
            'heater_setpoint': (250, 350),
            'valve_position': (0, 100),
            'pump_speed': (0, 100),
            'motor_speed': (0, 2000),
            'actuator_position': (0, 100),
            'gripper_force': (0, 1000),
            'generator_setpoint': (0, 1000),
            'transformer_tap': (-10, 10),
            'switch_position': (0, 1)
        }
        
        min_val, max_val = action_ranges.get(var_name, (0, 1))
        denormalized = min_val + normalized_value * (max_val - min_val)
        return denormalized
    
    def _update_deployment_metrics(self, deployment_session: Dict[str, Any]):
        """Update deployment performance metrics."""
        self.deployment_metrics['total_deployments'] += 1
        
        if deployment_session['status'] in ['completed_successfully', 'active']:
            # Calculate performance score
            safety_score = 1.0 - (len(deployment_session['safety_incidents']) / 
                                max(1, deployment_session.get('total_steps', 1)))
            performance_score = safety_score  # Simplified metric
            
            # Update average performance
            total_deps = self.deployment_metrics['total_deployments']
            current_avg = self.deployment_metrics['average_performance']
            self.deployment_metrics['average_performance'] = (
                (current_avg * (total_deps - 1) + performance_score) / total_deps
            )
        
        # Count safety incidents
        self.deployment_metrics['safety_incidents'] += len(deployment_session['safety_incidents'])
        
        # Calculate uptime (simplified)
        successful_deployments = sum(1 for session in self.deployment_sessions 
                                   if session['status'] == 'completed_successfully')
        self.deployment_metrics['uptime_percentage'] = (
            successful_deployments / max(1, len(self.deployment_sessions)) * 100
        )
    
    def _generate_compliance_records(self, deployment_session: Dict[str, Any], 
                                   config: IndustryConfig):
        """Generate regulatory compliance records."""
        for standard in config.regulatory_compliance:
            compliance_record = {
                'standard': standard,
                'plant_id': config.plant_identifier,
                'session_id': deployment_session['session_id'],
                'timestamp': time.time(),
                'compliance_status': 'compliant',  # Simplified assessment
                'safety_incidents': len(deployment_session['safety_incidents']),
                'total_steps': deployment_session.get('total_steps', 0),
                'duration': deployment_session.get('duration', 0)
            }
            
            # Assess compliance based on safety incidents
            if len(deployment_session['safety_incidents']) > 0:
                compliance_record['compliance_status'] = 'incident_reported'
                compliance_record['incident_details'] = deployment_session['safety_incidents']
            
            self.compliance_records[standard].append(compliance_record)
    
    def get_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment report."""
        return {
            'deployment_metrics': self.deployment_metrics.copy(),
            'active_connectors': len(self.active_connectors),
            'total_sessions': len(self.deployment_sessions),
            'recent_deployments': self.deployment_sessions[-5:] if self.deployment_sessions else [],
            'compliance_summary': {
                standard: {
                    'total_records': len(records),
                    'compliant_records': sum(1 for r in records if r['compliance_status'] == 'compliant'),
                    'incident_records': sum(1 for r in records if r['compliance_status'] == 'incident_reported')
                }
                for standard, records in self.compliance_records.items()
            },
            'connector_status': {
                plant_id: {
                    'connected': connector.connection_active,
                    'industry_type': connector.config.industry_type.value,
                    'protocol': connector.config.communication_protocol.value,
                    'metrics': connector.communication_metrics.copy()
                }
                for plant_id, connector in self.active_connectors.items()
            }
        }
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Generate safety-focused report."""
        all_incidents = []
        for session in self.deployment_sessions:
            all_incidents.extend(session['safety_incidents'])
        
        # Analyze incident patterns
        incident_types = defaultdict(int)
        for incident in all_incidents:
            for violation in incident['violations']:
                incident_types[violation.split(':')[0]] += 1
        
        return {
            'total_safety_incidents': len(all_incidents),
            'incidents_by_type': dict(incident_types),
            'recent_incidents': all_incidents[-10:] if all_incidents else [],
            'safety_performance': {
                'incident_rate': len(all_incidents) / max(1, sum(s.get('total_steps', 0) for s in self.deployment_sessions)),
                'emergency_stops': sum(1 for s in self.deployment_sessions if s['status'] == 'emergency_stopped'),
                'safety_score': 1.0 - (len(all_incidents) / max(1, self.deployment_metrics['total_deployments']))
            }
        }


# Global deployment manager
_global_deployment_manager = None


def get_deployment_manager() -> IndustrialDeploymentManager:
    """Get global industrial deployment manager."""
    global _global_deployment_manager
    if _global_deployment_manager is None:
        _global_deployment_manager = IndustrialDeploymentManager()
    return _global_deployment_manager


def deploy_to_industry(agent, industry_config: IndustryConfig, 
                      deployment_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function to deploy RL agent to industrial system."""
    manager = get_deployment_manager()
    connector = manager.create_connector(industry_config)
    
    deployment_config = deployment_config or {'duration': 3600, 'control_cycle': 1.0}
    
    return manager.deploy_rl_agent(agent, connector, deployment_config)