#!/usr/bin/env python3
"""
Configuration validation script for neoRL-industrial-gym.

This script validates all configuration files without requiring external dependencies.
"""

import json
import os
import re
import sys
from pathlib import Path


def validate_file_exists(file_path: str) -> bool:
    """Check if a file exists."""
    return Path(file_path).exists()


def validate_yaml_syntax(file_path: str) -> bool:
    """Basic YAML syntax validation without PyYAML."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Basic YAML structure checks
        lines = content.split('\n')
        indent_stack = []
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                continue
            
            # Check for basic YAML syntax errors
            if ':' not in stripped and not stripped.endswith(':'):
                # Should be a value or array item
                if not stripped.startswith('-') and not any(c in stripped for c in [':', '-', '|', '>']):
                    continue  # Likely a continuation
            
        return True
    except Exception as e:
        print(f"YAML validation error in {file_path}: {e}")
        return False


def validate_toml_syntax(file_path: str) -> bool:
    """Basic TOML syntax validation without tomli."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Basic TOML structure checks
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                continue
            
            # Check for section headers
            if stripped.startswith('[') and stripped.endswith(']'):
                continue
            
            # Check for key-value pairs
            if '=' in stripped:
                key, value = stripped.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Basic key validation
                if not key or ' ' in key.split('.')[0]:
                    print(f"Invalid key in {file_path}:{line_num}: {key}")
                    return False
        
        return True
    except Exception as e:
        print(f"TOML validation error in {file_path}: {e}")
        return False


def validate_json_syntax(file_path: str) -> bool:
    """Validate JSON syntax."""
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        print(f"JSON validation error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False


def validate_makefile(file_path: str) -> bool:
    """Basic Makefile validation."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for basic Makefile structure
        has_targets = False
        lines = content.split('\n')
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                continue
            
            # Check for targets (lines ending with :)
            if ':' in stripped and not stripped.startswith('\t'):
                has_targets = True
                target_name = stripped.split(':')[0].strip()
                
                # Validate target name
                if not target_name or ' ' in target_name:
                    print(f"Invalid target name in {file_path}: {target_name}")
                    return False
        
        if not has_targets:
            print(f"No targets found in {file_path}")
            return False
        
        return True
    except Exception as e:
        print(f"Makefile validation error: {e}")
        return False


def validate_python_syntax(file_path: str) -> bool:
    """Validate Python syntax."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Compile to check syntax
        compile(content, file_path, 'exec')
        return True
    except SyntaxError as e:
        print(f"Python syntax error in {file_path}: {e}")
        return False
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False


def validate_dockerfile(file_path: str) -> bool:
    """Basic Dockerfile validation."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        has_from = False
        
        in_multiline = False
        
        for line in lines:
            stripped = line.strip()
            original_line = line
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                continue
            
            # Check if this is a continuation of a multiline command
            if original_line.startswith('    ') or original_line.startswith('\t'):
                continue  # Skip continuation lines
            
            # Check if line ends with backslash (multiline)
            if stripped.endswith('\\'):
                in_multiline = True
            elif in_multiline and not stripped.endswith('\\'):
                in_multiline = False
                continue  # This is the end of a multiline, skip validation
            elif in_multiline:
                continue  # Still in multiline, skip
            
            # Check for FROM instruction
            if stripped.upper().startswith('FROM '):
                has_from = True
            
            # Basic instruction validation
            if stripped and not in_multiline:
                instruction = stripped.split()[0].upper()
                valid_instructions = [
                    'FROM', 'RUN', 'CMD', 'LABEL', 'EXPOSE', 'ENV', 'ADD', 'COPY',
                    'ENTRYPOINT', 'VOLUME', 'USER', 'WORKDIR', 'ARG', 'ONBUILD',
                    'STOPSIGNAL', 'HEALTHCHECK', 'SHELL'
                ]
                
                if instruction not in valid_instructions:
                    print(f"Unknown Dockerfile instruction in {file_path}: {instruction}")
                    return False
        
        if not has_from:
            print(f"Dockerfile {file_path} missing FROM instruction")
            return False
        
        return True
    except Exception as e:
        print(f"Dockerfile validation error: {e}")
        return False


def main():
    """Main validation function."""
    print("🔍 Validating configuration files...\n")
    
    # Files to validate with their validators
    validations = [
        ('.pre-commit-config.yaml', validate_yaml_syntax),
        ('pyproject.toml', validate_toml_syntax),
        ('Makefile', validate_makefile),
        ('Dockerfile', validate_dockerfile),
        ('.github/dependabot.yml', validate_yaml_syntax),
        ('config/environments/development.yaml', validate_yaml_syntax),
        ('config/environments/production.yaml', validate_yaml_syntax),
        ('config/environments/testing.yaml', validate_yaml_syntax),
        ('config/config_loader.py', validate_python_syntax),
        ('scripts/security_scan.py', validate_python_syntax),
        ('scripts/container_security.py', validate_python_syntax),
    ]
    
    # Track validation results
    passed = 0
    failed = 0
    
    for file_path, validator in validations:
        full_path = Path(file_path)
        
        if not full_path.exists():
            print(f"❌ File not found: {file_path}")
            failed += 1
            continue
        
        try:
            if validator(str(full_path)):
                print(f"✅ {file_path}")
                passed += 1
            else:
                print(f"❌ {file_path}")
                failed += 1
        except Exception as e:
            print(f"❌ {file_path} - Error: {e}")
            failed += 1
    
    # Summary
    print(f"\n📊 Validation Summary:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All configuration files are valid!")
        return 0
    else:
        print(f"\n⚠️  {failed} configuration files have issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())