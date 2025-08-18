"""Contract testing configuration and fixtures for API validation."""

import json
import pytest
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

import numpy as np
import jsonschema


@dataclass
class ContractSchema:
    """Schema definition for API contracts."""
    name: str
    version: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    examples: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


class ContractValidator:
    """Validator for API contracts and data schemas."""
    
    def __init__(self):
        self.schemas = {}
        self.validation_results = []
    
    def register_schema(self, schema: ContractSchema):
        """Register a contract schema."""
        self.schemas[schema.name] = schema
    
    def validate_input(self, schema_name: str, data: Any) -> bool:
        """Validate input data against schema."""
        if schema_name not in self.schemas:
            raise ValueError(f"Schema {schema_name} not registered")
        
        schema = self.schemas[schema_name]
        try:
            jsonschema.validate(data, schema.input_schema)
            self.validation_results.append({
                "schema": schema_name,
                "type": "input",
                "valid": True,
                "data": data
            })
            return True
        except jsonschema.ValidationError as e:
            self.validation_results.append({
                "schema": schema_name,
                "type": "input",
                "valid": False,
                "data": data,
                "error": str(e)
            })
            return False
    
    def validate_output(self, schema_name: str, data: Any) -> bool:
        """Validate output data against schema."""
        if schema_name not in self.schemas:
            raise ValueError(f"Schema {schema_name} not registered")
        
        schema = self.schemas[schema_name]
        try:
            jsonschema.validate(data, schema.output_schema)
            self.validation_results.append({
                "schema": schema_name,
                "type": "output",
                "valid": True,
                "data": data
            })
            return True
        except jsonschema.ValidationError as e:
            self.validation_results.append({
                "schema": schema_name,
                "type": "output",
                "valid": False,
                "data": data,
                "error": str(e)
            })
            return False
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get detailed validation report."""
        total = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r["valid"])
        failed = total - passed
        
        return {
            "total_validations": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed / total if total > 0 else 0,
            "results": self.validation_results
        }


@pytest.fixture
def contract_validator():
    """Provide contract validation capabilities."""
    return ContractValidator()


@pytest.fixture
def environment_schemas():
    """Define contract schemas for environment interfaces."""
    schemas = {}
    
    # Base environment schema
    base_observation_schema = {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 1,
        "maxItems": 100
    }
    
    base_action_schema = {
        "type": "array", 
        "items": {"type": "number"},
        "minItems": 1,
        "maxItems": 20
    }
    
    # Environment step contract
    schemas["environment_step"] = ContractSchema(
        name="environment_step",
        version="1.0.0",
        input_schema={
            "type": "object",
            "properties": {
                "action": base_action_schema
            },
            "required": ["action"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "observation": base_observation_schema,
                "reward": {"type": "number"},
                "done": {"type": "boolean"},
                "info": {"type": "object"}
            },
            "required": ["observation", "reward", "done", "info"]
        },
        examples=[
            {
                "input": {"action": [0.5, -0.3, 0.8]},
                "output": {
                    "observation": [1.2, -0.5, 0.8, 2.1],
                    "reward": 10.5,
                    "done": False,
                    "info": {"safety_violation": False}
                }
            }
        ]
    )
    
    # Environment reset contract
    schemas["environment_reset"] = ContractSchema(
        name="environment_reset",
        version="1.0.0",
        input_schema={
            "type": "object",
            "properties": {
                "seed": {"type": ["integer", "null"]},
                "options": {"type": ["object", "null"]}
            }
        },
        output_schema={
            "type": "object",
            "properties": {
                "observation": base_observation_schema,
                "info": {"type": "object"}
            },
            "required": ["observation", "info"]
        },
        examples=[
            {
                "input": {"seed": 42},
                "output": {
                    "observation": [0.0, 0.0, 0.0, 0.0],
                    "info": {"episode_id": 1}
                }
            }
        ]
    )
    
    return schemas


@pytest.fixture
def agent_schemas():
    """Define contract schemas for agent interfaces."""
    schemas = {}
    
    # Agent predict contract
    schemas["agent_predict"] = ContractSchema(
        name="agent_predict",
        version="1.0.0",
        input_schema={
            "type": "object",
            "properties": {
                "observation": {
                    "type": "array",
                    "items": {"type": "number"}
                },
                "deterministic": {"type": "boolean"}
            },
            "required": ["observation"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "array",
                    "items": {"type": "number"}
                },
                "log_prob": {"type": ["number", "null"]},
                "value": {"type": ["number", "null"]}
            },
            "required": ["action"]
        },
        examples=[
            {
                "input": {
                    "observation": [1.0, 2.0, 3.0],
                    "deterministic": True
                },
                "output": {
                    "action": [0.5, -0.2],
                    "log_prob": -1.5,
                    "value": 10.2
                }
            }
        ]
    )
    
    # Agent train step contract
    schemas["agent_train_step"] = ContractSchema(
        name="agent_train_step", 
        version="1.0.0",
        input_schema={
            "type": "object",
            "properties": {
                "batch": {
                    "type": "object",
                    "properties": {
                        "observations": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}}
                        },
                        "actions": {
                            "type": "array", 
                            "items": {"type": "array", "items": {"type": "number"}}
                        },
                        "rewards": {
                            "type": "array",
                            "items": {"type": "number"}
                        },
                        "dones": {
                            "type": "array",
                            "items": {"type": "boolean"}
                        }
                    },
                    "required": ["observations", "actions", "rewards", "dones"]
                }
            },
            "required": ["batch"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "loss": {"type": "number", "minimum": 0},
                "metrics": {"type": "object"},
                "step": {"type": "integer", "minimum": 0}
            },
            "required": ["loss", "metrics", "step"]
        },
        examples=[]
    )
    
    return schemas


@pytest.fixture 
def safety_schemas():
    """Define contract schemas for safety system interfaces."""
    schemas = {}
    
    # Safety check contract
    schemas["safety_check"] = ContractSchema(
        name="safety_check",
        version="1.0.0",
        input_schema={
            "type": "object",
            "properties": {
                "state": {
                    "type": "array",
                    "items": {"type": "number"}
                },
                "action": {
                    "type": "array", 
                    "items": {"type": "number"}
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["state", "action"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "safe": {"type": "boolean"},
                "violations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "constraint": {"type": "string"},
                            "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                            "value": {"type": "number"},
                            "threshold": {"type": "number"}
                        }
                    }
                },
                "emergency_stop": {"type": "boolean"}
            },
            "required": ["safe", "violations", "emergency_stop"]
        },
        examples=[
            {
                "input": {
                    "state": [300.0, 5.0, 0.8],  # temp, pressure, flow
                    "action": [0.1, -0.2, 0.05],
                    "constraints": ["temperature", "pressure"]
                },
                "output": {
                    "safe": True,
                    "violations": [],
                    "emergency_stop": False
                }
            }
        ]
    )
    
    return schemas


@pytest.fixture
def dataset_schemas():
    """Define contract schemas for dataset interfaces."""
    schemas = {}
    
    # Dataset loading contract
    schemas["dataset_load"] = ContractSchema(
        name="dataset_load",
        version="1.0.0",
        input_schema={
            "type": "object",
            "properties": {
                "environment": {"type": "string"},
                "quality": {"type": "string", "enum": ["expert", "medium", "mixed", "random"]},
                "subset": {"type": ["string", "null"]},
                "normalize": {"type": "boolean"}
            },
            "required": ["environment", "quality"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "observations": {"type": "array"},
                "actions": {"type": "array"},
                "rewards": {"type": "array"},
                "terminals": {"type": "array"},
                "metadata": {
                    "type": "object",
                    "properties": {
                        "size": {"type": "integer", "minimum": 0},
                        "episode_count": {"type": "integer", "minimum": 0},
                        "mean_return": {"type": "number"},
                        "std_return": {"type": "number"}
                    }
                }
            },
            "required": ["observations", "actions", "rewards", "terminals", "metadata"]
        },
        examples=[]
    )
    
    return schemas


@pytest.fixture
def contract_test_data():
    """Generate test data for contract validation."""
    return {
        "valid_observations": [
            [1.0, 2.0, 3.0],
            [0.5, -1.2, 4.1, 2.3],
            list(np.random.randn(10).astype(float))
        ],
        "invalid_observations": [
            [],  # Empty array
            ["not", "numbers"],  # Wrong type
            [float('inf')],  # Invalid number
        ],
        "valid_actions": [
            [0.5, -0.3],
            [1.0, 0.0, -1.0, 0.5],
            list(np.random.randn(5).astype(float))
        ],
        "invalid_actions": [
            [],  # Empty array
            [None, 1.0],  # Null values
            "not_array",  # Wrong type
        ]
    }


def create_contract_tests(validator: ContractValidator, schemas: Dict[str, ContractSchema]):
    """Create contract test cases for all schemas."""
    test_cases = []
    
    for schema_name, schema in schemas.items():
        validator.register_schema(schema)
        
        # Test examples if provided
        for example in schema.examples:
            test_cases.append({
                "schema": schema_name,
                "type": "example",
                "input": example.get("input", {}),
                "expected_output": example.get("output", {})
            })
    
    return test_cases


@pytest.fixture
def contract_test_runner(contract_validator):
    """Provide utilities for running contract tests."""
    class ContractTestRunner:
        def __init__(self, validator):
            self.validator = validator
            self.results = []
        
        def run_input_validation(self, schema_name: str, test_data: List[Any]):
            """Run input validation tests."""
            for data in test_data:
                result = self.validator.validate_input(schema_name, data)
                self.results.append({
                    "schema": schema_name,
                    "type": "input",
                    "data": data,
                    "result": result
                })
        
        def run_output_validation(self, schema_name: str, test_data: List[Any]):
            """Run output validation tests.""" 
            for data in test_data:
                result = self.validator.validate_output(schema_name, data)
                self.results.append({
                    "schema": schema_name,
                    "type": "output",
                    "data": data,
                    "result": result
                })
        
        def get_summary(self) -> Dict[str, Any]:
            """Get test summary."""
            total = len(self.results)
            passed = sum(1 for r in self.results if r["result"])
            return {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "success_rate": passed / total if total > 0 else 0
            }
    
    return ContractTestRunner(contract_validator)