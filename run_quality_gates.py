#!/usr/bin/env python3
"""Quality gates verification without external dependencies."""

import os
import sys
import subprocess
from pathlib import Path


def check_file_structure():
    """Verify project file structure."""
    print("🔍 Checking file structure...")
    
    required_files = [
        "src/neorl_industrial/__init__.py",
        "src/neorl_industrial/core/types.py",
        "src/neorl_industrial/environments/base.py",
        "src/neorl_industrial/environments/chemical_reactor.py", 
        "src/neorl_industrial/agents/base.py",
        "src/neorl_industrial/agents/cql.py",
        "src/neorl_industrial/datasets/loader.py",
        "src/neorl_industrial/utils.py",
        "examples/basic_usage.py",
        "pyproject.toml",
        "README.md",
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
    
    if missing:
        print("✗ Missing files:")
        for f in missing:
            print(f"  - {f}")
        return False
    else:
        print("✓ All required files present")
        return True


def check_code_quality():
    """Check code quality without external linters."""
    print("\n🔍 Checking code quality...")
    
    python_files = []
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    issues = []
    
    for file_path in python_files:
        with open(file_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                # Check for very long lines (>120 chars)
                if len(line) > 120:
                    issues.append(f"{file_path}:{i} - Line too long ({len(line)} chars)")
                
                # Check for TODO/FIXME comments
                if 'TODO' in line or 'FIXME' in line:
                    issues.append(f"{file_path}:{i} - TODO/FIXME found: {line.strip()}")
    
    if issues:
        print("✗ Code quality issues found:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        return False
    else:
        print("✓ Code quality checks passed")
        return True


def check_documentation():
    """Check documentation completeness."""
    print("\n🔍 Checking documentation...")
    
    # Check README structure
    readme_path = Path("README.md")
    if not readme_path.exists():
        print("✗ README.md not found")
        return False
    
    readme_content = readme_path.read_text()
    required_sections = [
        "# neoRL-industrial-gym",
        "## Overview", 
        "## Installation",
        "## Quick Start",
        "## Features",
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in readme_content:
            missing_sections.append(section)
    
    if missing_sections:
        print("✗ Missing README sections:")
        for section in missing_sections:
            print(f"  - {section}")
        return False
    
    # Check docstrings in key files
    key_files = [
        "src/neorl_industrial/__init__.py",
        "src/neorl_industrial/environments/base.py",
        "src/neorl_industrial/agents/base.py",
    ]
    
    undocumented = []
    for file_path in key_files:
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                content = f.read()
                if '"""' not in content:
                    undocumented.append(file_path)
    
    if undocumented:
        print("✗ Files missing docstrings:")
        for f in undocumented:
            print(f"  - {f}")
        return False
    
    print("✓ Documentation checks passed")
    return True


def check_security():
    """Basic security checks."""
    print("\n🔍 Checking security...")
    
    security_issues = []
    
    # Check for hardcoded secrets
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read().lower()
                    
                    # Simple patterns for potential secrets
                    suspicious_patterns = [
                        'password =',
                        'secret =', 
                        'api_key =',
                        'token =',
                        'private_key =',
                    ]
                    
                    for pattern in suspicious_patterns:
                        if pattern in content:
                            security_issues.append(f"{file_path}: Potential hardcoded secret ({pattern})")
    
    # Check imports for potentially dangerous modules
    dangerous_imports = ['subprocess', 'os.system', 'eval', 'exec']
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    for dangerous in dangerous_imports:
                        if f"import {dangerous}" in content or f"from {dangerous}" in content:
                            # Allow subprocess in specific contexts
                            if dangerous == 'subprocess' and 'scripts/' in file_path:
                                continue
                            security_issues.append(f"{file_path}: Potentially dangerous import ({dangerous})")
    
    if security_issues:
        print("⚠ Security issues found:")
        for issue in security_issues:
            print(f"  {issue}")
        return len(security_issues) == 0  # Return True only if no critical issues
    else:
        print("✓ Security checks passed")
        return True


def check_project_config():
    """Check project configuration files."""
    print("\n🔍 Checking project configuration...")
    
    # Check pyproject.toml
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("✗ pyproject.toml not found")
        return False
    
    pyproject_content = pyproject_path.read_text()
    
    required_sections = [
        "[build-system]",
        "[project]", 
        "dependencies",
    ]
    
    missing = []
    for section in required_sections:
        if section not in pyproject_content:
            missing.append(section)
    
    if missing:
        print("✗ Missing pyproject.toml sections:")
        for section in missing:
            print(f"  - {section}")
        return False
    
    print("✓ Project configuration valid")
    return True


def main():
    """Run all quality gates."""
    print("🚀 GENERATION 1 QUALITY GATES")
    print("=" * 50)
    
    checks = [
        ("File Structure", check_file_structure),
        ("Code Quality", check_code_quality),
        ("Documentation", check_documentation), 
        ("Security", check_security),
        ("Project Config", check_project_config),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} check failed with error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("QUALITY GATES SUMMARY:")
    
    passed = 0
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
        if result:
            passed += 1
    
    success_rate = passed / len(results)
    print(f"\nOverall: {passed}/{len(results)} checks passed ({success_rate:.1%})")
    
    if success_rate >= 0.8:  # 80% pass rate required
        print("🎉 GENERATION 1 QUALITY GATES PASSED!")
        print("✅ Ready to proceed to Generation 2")
        return 0
    else:
        print("❌ Quality gates failed. Please fix issues before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)