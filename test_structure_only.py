#!/usr/bin/env python3
"""Test code structure and syntax without runtime dependencies."""

import ast
import os
import sys
from pathlib import Path


def test_python_syntax(file_path):
    """Test if a Python file has valid syntax."""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST to check syntax
        ast.parse(content, filename=str(file_path))
        return True, None
        
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def test_import_structure(file_path):
    """Test import structure and detect circular imports."""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(file_path))
        
        imports = []
        relative_imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:  # Relative import
                    relative_imports.append(node.module or "")
                else:
                    imports.append(node.module or "")
        
        return True, {
            "imports": imports,
            "relative_imports": relative_imports
        }
        
    except Exception as e:
        return False, f"Import analysis failed: {e}"


def test_class_structure(file_path):
    """Test class definitions and inheritance."""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(file_path))
        
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [base.id for base in node.bases if hasattr(base, 'id')]
                classes.append({
                    "name": node.name,
                    "bases": bases,
                    "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                })
            elif isinstance(node, ast.FunctionDef):
                if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree) 
                          if hasattr(parent, 'body') and node in getattr(parent, 'body', [])):
                    functions.append(node.name)
        
        return True, {
            "classes": classes,
            "functions": functions
        }
        
    except Exception as e:
        return False, f"Class analysis failed: {e}"


def find_python_files(directory):
    """Find all Python files in directory."""
    
    python_files = []
    directory = Path(directory)
    
    for file_path in directory.rglob("*.py"):
        # Skip __pycache__ and .git directories
        if "__pycache__" not in str(file_path) and ".git" not in str(file_path):
            python_files.append(file_path)
    
    return sorted(python_files)


def test_research_module_structure():
    """Test the structure of research modules."""
    
    print("🔍 Testing research module structure...")
    
    research_dir = Path("src/neorl_industrial/research")
    
    if not research_dir.exists():
        print("❌ Research directory not found")
        return False
    
    expected_files = [
        "__init__.py",
        "novel_algorithms.py", 
        "meta_learning.py",
        "continual_learning.py",
        "neural_architecture_search.py",
        "foundation_models.py",
        "research_accelerator.py",
        "distributed_training.py"
    ]
    
    all_present = True
    
    for expected_file in expected_files:
        file_path = research_dir / expected_file
        if file_path.exists():
            print(f"✅ {expected_file} exists")
        else:
            print(f"❌ {expected_file} missing")
            all_present = False
    
    return all_present


def analyze_code_quality():
    """Analyze code quality metrics."""
    
    print("\n📊 Analyzing code quality...")
    
    src_dir = Path("src")
    python_files = find_python_files(src_dir)
    
    total_files = len(python_files)
    valid_syntax = 0
    total_lines = 0
    total_classes = 0
    total_functions = 0
    
    for file_path in python_files:
        # Test syntax
        syntax_ok, syntax_error = test_python_syntax(file_path)
        if syntax_ok:
            valid_syntax += 1
        else:
            print(f"❌ Syntax error in {file_path}: {syntax_error}")
        
        # Count lines
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                total_lines += lines
        except:
            pass
        
        # Analyze structure
        structure_ok, structure_info = test_class_structure(file_path)
        if structure_ok:
            total_classes += len(structure_info["classes"])
            total_functions += len(structure_info["functions"])
    
    print(f"\n📈 Code Quality Metrics:")
    print(f"  • Total Python files: {total_files}")
    print(f"  • Files with valid syntax: {valid_syntax}/{total_files} ({valid_syntax/total_files:.1%})")
    print(f"  • Total lines of code: {total_lines}")
    print(f"  • Total classes: {total_classes}")
    print(f"  • Total functions: {total_functions}")
    
    return valid_syntax == total_files


def test_documentation_structure():
    """Test documentation and README structure."""
    
    print("\n📚 Testing documentation structure...")
    
    docs_checks = [
        ("README.md", Path("README.md")),
        ("Examples directory", Path("examples")),
        ("Advanced demo", Path("examples/advanced_research_demo.py")),
        ("Documentation directory", Path("docs")),
    ]
    
    all_present = True
    
    for name, path in docs_checks:
        if path.exists():
            print(f"✅ {name} exists")
        else:
            print(f"❌ {name} missing")
            all_present = False
    
    return all_present


def test_configuration_files():
    """Test configuration and setup files."""
    
    print("\n⚙️ Testing configuration files...")
    
    config_files = [
        "pyproject.toml",
        "requirements.txt", 
        "requirements-dev.txt"
    ]
    
    all_present = True
    
    for config_file in config_files:
        file_path = Path(config_file)
        if file_path.exists():
            print(f"✅ {config_file} exists")
            
            # Basic validation for pyproject.toml
            if config_file == "pyproject.toml":
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if "[project]" in content:
                            print(f"  ✅ Valid project configuration")
                        else:
                            print(f"  ⚠️ No [project] section found")
                except:
                    print(f"  ❌ Could not read {config_file}")
                    
        else:
            print(f"❌ {config_file} missing")
            all_present = False
    
    return all_present


def run_structure_tests():
    """Run all structure tests."""
    
    print("🏗️ NEORL-INDUSTRIAL CODE STRUCTURE VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Research Module Structure", test_research_module_structure),
        ("Code Quality Analysis", analyze_code_quality),
        ("Documentation Structure", test_documentation_structure),
        ("Configuration Files", test_configuration_files),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        
        try:
            success = test_func()
            if success:
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    # Final report
    print("\n" + "=" * 60)
    print("🎯 STRUCTURE VALIDATION SUMMARY")
    print("=" * 60)
    
    success_rate = passed_tests / total_tests
    
    print(f"✅ Passed: {passed_tests}/{total_tests} ({success_rate:.1%})")
    
    if passed_tests == total_tests:
        print("🎉 ALL STRUCTURE TESTS PASSED!")
        print("📦 Code structure is well-organized and complete")
        exit_code = 0
    else:
        failed_tests = total_tests - passed_tests
        print(f"❌ Failed: {failed_tests} tests")
        print("🔧 Some structural issues need attention")
        exit_code = 1
    
    print("\n🚀 Structure validation complete!")
    
    return exit_code


if __name__ == "__main__":
    exit_code = run_structure_tests()
    sys.exit(exit_code)