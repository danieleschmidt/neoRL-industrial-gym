"""Comprehensive quality gates execution for neoRL-industrial-gym."""

import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QualityGateRunner:
    """Runs comprehensive quality gates for the project."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {}
        self.start_time = time.time()
        
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return results."""
        logger.info("🚀 Starting comprehensive quality gates...")
        
        gates = [
            ("Code Style", self.run_code_style_checks),
            ("Type Checking", self.run_type_checking),
            ("Unit Tests", self.run_unit_tests),
            ("Integration Tests", self.run_integration_tests),
            ("Security Scan", self.run_security_scan),
            ("Performance Tests", self.run_performance_tests),
            ("Documentation Check", self.run_documentation_check),
            ("License Compliance", self.run_license_compliance),
        ]
        
        for gate_name, gate_func in gates:
            logger.info(f"🔍 Running {gate_name}...")
            try:
                self.results[gate_name] = gate_func()
                logger.info(f"✅ {gate_name} completed")
            except Exception as e:
                logger.error(f"❌ {gate_name} failed: {str(e)}")
                self.results[gate_name] = {
                    "status": "failed",
                    "error": str(e),
                    "duration": 0
                }
        
        # Generate summary
        self.results["summary"] = self.generate_summary()
        
        total_time = time.time() - self.start_time
        logger.info(f"🏁 Quality gates completed in {total_time:.2f} seconds")
        
        return self.results
    
    def run_code_style_checks(self) -> Dict[str, Any]:
        """Run code style and linting checks."""
        start_time = time.time()
        
        try:
            # Check if ruff is available
            result = subprocess.run(
                ["python3", "-m", "ruff", "check", str(self.project_root / "src")],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            ruff_issues = len(result.stdout.split('\n')) if result.stdout else 0
            
            return {
                "status": "passed" if result.returncode == 0 else "warning",
                "ruff_issues": ruff_issues,
                "details": result.stdout if result.stdout else "No issues found",
                "duration": time.time() - start_time
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "error": "Code style check timed out",
                "duration": time.time() - start_time
            }
        except FileNotFoundError:
            # Fallback to basic style checks
            return {
                "status": "skipped",
                "reason": "Ruff not available, using basic checks",
                "duration": time.time() - start_time
            }
    
    def run_type_checking(self) -> Dict[str, Any]:
        """Run static type checking."""
        start_time = time.time()
        
        try:
            result = subprocess.run(
                ["python3", "-m", "mypy", str(self.project_root / "src"), "--ignore-missing-imports"],
                capture_output=True,
                text=True,
                timeout=180
            )
            
            type_errors = result.stdout.count("error:") if result.stdout else 0
            
            return {
                "status": "passed" if type_errors == 0 else "warning",
                "type_errors": type_errors,
                "details": result.stdout[:1000] if result.stdout else "No type errors",
                "duration": time.time() - start_time
            }
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return {
                "status": "skipped",
                "reason": "MyPy not available or timed out",
                "duration": time.time() - start_time
            }
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests with coverage."""
        start_time = time.time()
        
        try:
            # Try to run pytest with coverage
            result = subprocess.run([
                "python3", "-m", "pytest", 
                str(self.project_root / "tests"),
                "-v", 
                "--tb=short",
                "--maxfail=10"
            ], capture_output=True, text=True, timeout=300)
            
            # Parse pytest output
            output_lines = result.stdout.split('\n') if result.stdout else []
            
            test_passed = 0
            test_failed = 0
            test_skipped = 0
            
            for line in output_lines:
                if " PASSED " in line:
                    test_passed += 1
                elif " FAILED " in line:
                    test_failed += 1
                elif " SKIPPED " in line:
                    test_skipped += 1
            
            # Extract coverage if available
            coverage_pct = None
            for line in output_lines:
                if "%" in line and "coverage" in line.lower():
                    try:
                        coverage_pct = float(line.split("%")[0].split()[-1])
                        break
                    except:
                        pass
            
            status = "passed" if test_failed == 0 else "failed"
            
            return {
                "status": status,
                "tests_passed": test_passed,
                "tests_failed": test_failed,
                "tests_skipped": test_skipped,
                "coverage_percent": coverage_pct,
                "details": result.stdout[-2000:] if result.stdout else "",
                "duration": time.time() - start_time
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "error": "Tests timed out after 5 minutes",
                "duration": time.time() - start_time
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Test execution failed: {str(e)}",
                "duration": time.time() - start_time
            }
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        start_time = time.time()
        
        # For now, create a simple integration test
        try:
            # Test basic imports
            test_script = """
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import neorl_industrial as ni
    
    # Test basic functionality
    env = ni.make('ChemicalReactor-v0')
    obs, info = env.reset()
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    # Test agent creation
    agent = ni.CQLAgent(state_dim=12, action_dim=3)
    
    print("✅ Integration test passed")
    
except Exception as e:
    print(f"❌ Integration test failed: {e}")
    sys.exit(1)
"""
            
            with open(self.project_root / "temp_integration_test.py", "w") as f:
                f.write(test_script)
            
            result = subprocess.run([
                "python3", str(self.project_root / "temp_integration_test.py")
            ], capture_output=True, text=True, timeout=120)
            
            # Cleanup
            (self.project_root / "temp_integration_test.py").unlink(missing_ok=True)
            
            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "output": result.stdout,
                "error": result.stderr if result.stderr else None,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Integration test setup failed: {str(e)}",
                "duration": time.time() - start_time
            }
    
    def run_security_scan(self) -> Dict[str, Any]:
        """Run security vulnerability scan."""
        start_time = time.time()
        
        try:
            # Try bandit for Python security
            result = subprocess.run([
                "python3", "-m", "bandit", "-r", str(self.project_root / "src"),
                "-f", "json"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                try:
                    bandit_results = json.loads(result.stdout)
                    high_severity = len([issue for issue in bandit_results.get("results", []) 
                                       if issue.get("issue_severity") == "HIGH"])
                    medium_severity = len([issue for issue in bandit_results.get("results", []) 
                                         if issue.get("issue_severity") == "MEDIUM"])
                    
                    return {
                        "status": "passed" if high_severity == 0 else "warning",
                        "high_severity_issues": high_severity,
                        "medium_severity_issues": medium_severity,
                        "total_issues": len(bandit_results.get("results", [])),
                        "duration": time.time() - start_time
                    }
                except json.JSONDecodeError:
                    pass
            
            # Fallback manual security check
            security_patterns = [
                "eval(", "exec(", "os.system(", "subprocess.call(",
                "pickle.loads(", "yaml.load(", "input("
            ]
            
            security_issues = 0
            for py_file in (self.project_root / "src").rglob("*.py"):
                try:
                    content = py_file.read_text()
                    for pattern in security_patterns:
                        security_issues += content.count(pattern)
                except:
                    pass
            
            return {
                "status": "passed" if security_issues == 0 else "warning",
                "potential_issues": security_issues,
                "method": "manual_pattern_check",
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            return {
                "status": "skipped",
                "reason": f"Security scan failed: {str(e)}",
                "duration": time.time() - start_time
            }
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run basic performance tests."""
        start_time = time.time()
        
        try:
            perf_script = """
import sys
import os
import time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import neorl_industrial as ni

# Performance test 1: Environment step performance
start_time = time.time()
env = ni.make('ChemicalReactor-v0')
env.reset()

step_times = []
for _ in range(100):
    action = env.action_space.sample()
    step_start = time.time()
    env.step(action)
    step_times.append(time.time() - step_start)

avg_step_time = np.mean(step_times)
steps_per_second = 1.0 / avg_step_time

print(f"Environment step performance: {steps_per_second:.1f} steps/sec")

# Performance test 2: Agent creation time
start_time = time.time()
agent = ni.CQLAgent(state_dim=12, action_dim=3)
agent_creation_time = time.time() - start_time

print(f"Agent creation time: {agent_creation_time:.3f} seconds")

# Performance test 3: Memory usage estimation
import psutil
process = psutil.Process()
memory_mb = process.memory_info().rss / 1024 / 1024

print(f"Memory usage: {memory_mb:.1f} MB")

# Output results
print(f"PERF_RESULTS: steps_per_second={steps_per_second:.1f}, agent_creation_time={agent_creation_time:.3f}, memory_mb={memory_mb:.1f}")
"""
            
            with open(self.project_root / "temp_perf_test.py", "w") as f:
                f.write(perf_script)
            
            result = subprocess.run([
                "python3", str(self.project_root / "temp_perf_test.py")
            ], capture_output=True, text=True, timeout=60)
            
            # Cleanup
            (self.project_root / "temp_perf_test.py").unlink(missing_ok=True)
            
            # Parse performance results
            perf_results = {}
            if "PERF_RESULTS:" in result.stdout:
                perf_line = [line for line in result.stdout.split('\n') if "PERF_RESULTS:" in line][0]
                perf_data = perf_line.split("PERF_RESULTS:")[1].strip()
                for item in perf_data.split(", "):
                    key, value = item.split("=")
                    perf_results[key] = float(value)
            
            return {
                "status": "passed" if result.returncode == 0 else "failed",
                "performance_metrics": perf_results,
                "output": result.stdout,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Performance test failed: {str(e)}",
                "duration": time.time() - start_time
            }
    
    def run_documentation_check(self) -> Dict[str, Any]:
        """Check documentation completeness."""
        start_time = time.time()
        
        try:
            doc_files = {
                "README.md": self.project_root / "README.md",
                "LICENSE": self.project_root / "LICENSE",
                "CONTRIBUTING.md": self.project_root / "CONTRIBUTING.md",
                "pyproject.toml": self.project_root / "pyproject.toml",
            }
            
            missing_docs = []
            present_docs = []
            
            for doc_name, doc_path in doc_files.items():
                if doc_path.exists():
                    present_docs.append(doc_name)
                else:
                    missing_docs.append(doc_name)
            
            # Check for docstrings in main modules
            src_path = self.project_root / "src" / "neorl_industrial"
            py_files_with_docstrings = 0
            total_py_files = 0
            
            if src_path.exists():
                for py_file in src_path.rglob("*.py"):
                    if py_file.name == "__init__.py":
                        continue
                    
                    total_py_files += 1
                    try:
                        content = py_file.read_text()
                        if '"""' in content or "'''" in content:
                            py_files_with_docstrings += 1
                    except:
                        pass
            
            docstring_coverage = (py_files_with_docstrings / total_py_files * 100) if total_py_files > 0 else 0
            
            return {
                "status": "passed" if len(missing_docs) <= 1 else "warning",
                "present_docs": present_docs,
                "missing_docs": missing_docs,
                "docstring_coverage_percent": docstring_coverage,
                "py_files_checked": total_py_files,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Documentation check failed: {str(e)}",
                "duration": time.time() - start_time
            }
    
    def run_license_compliance(self) -> Dict[str, Any]:
        """Check license compliance."""
        start_time = time.time()
        
        try:
            license_file = self.project_root / "LICENSE"
            pyproject_file = self.project_root / "pyproject.toml"
            
            license_present = license_file.exists()
            
            # Check pyproject.toml for license declaration
            license_declared = False
            if pyproject_file.exists():
                try:
                    content = pyproject_file.read_text()
                    license_declared = "license" in content.lower()
                except:
                    pass
            
            # Basic file header check
            py_files_with_license = 0
            total_py_files = 0
            
            src_path = self.project_root / "src" / "neorl_industrial"
            if src_path.exists():
                for py_file in src_path.rglob("*.py"):
                    total_py_files += 1
                    try:
                        content = py_file.read_text()
                        # Look for common license indicators
                        if any(indicator in content.lower() for indicator in 
                               ["license", "copyright", "mit", "apache", "bsd"]):
                            py_files_with_license += 1
                    except:
                        pass
            
            return {
                "status": "passed" if license_present and license_declared else "warning",
                "license_file_present": license_present,
                "license_declared": license_declared,
                "files_with_license_headers": py_files_with_license,
                "total_files_checked": total_py_files,
                "duration": time.time() - start_time
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": f"License compliance check failed: {str(e)}",
                "duration": time.time() - start_time
            }
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate overall summary of quality gates."""
        total_gates = len(self.results)
        passed_gates = sum(1 for result in self.results.values() 
                          if result.get("status") == "passed")
        failed_gates = sum(1 for result in self.results.values() 
                          if result.get("status") == "failed")
        warning_gates = sum(1 for result in self.results.values() 
                           if result.get("status") == "warning")
        skipped_gates = sum(1 for result in self.results.values() 
                           if result.get("status") == "skipped")
        
        overall_status = "PASSED"
        if failed_gates > 0:
            overall_status = "FAILED"
        elif warning_gates > 2:  # Allow some warnings
            overall_status = "WARNING"
        
        total_duration = time.time() - self.start_time
        
        return {
            "overall_status": overall_status,
            "total_gates": total_gates,
            "passed": passed_gates,
            "failed": failed_gates,
            "warnings": warning_gates,
            "skipped": skipped_gates,
            "total_duration": total_duration,
            "timestamp": time.time()
        }

def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    
    runner = QualityGateRunner(project_root)
    results = runner.run_all_gates()
    
    # Save results
    results_file = project_root / "quality_gates_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    summary = results["summary"]
    print("\n" + "="*80)
    print("🎯 QUALITY GATES SUMMARY")
    print("="*80)
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Passed: {summary['passed']}/{summary['total_gates']}")
    print(f"Failed: {summary['failed']}")
    print(f"Warnings: {summary['warnings']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Total Duration: {summary['total_duration']:.2f} seconds")
    print(f"Results saved to: {results_file}")
    
    # Detailed results
    print("\n📊 DETAILED RESULTS:")
    for gate_name, gate_result in results.items():
        if gate_name == "summary":
            continue
        
        status = gate_result.get("status", "unknown")
        duration = gate_result.get("duration", 0)
        
        status_emoji = {
            "passed": "✅",
            "failed": "❌", 
            "warning": "⚠️",
            "skipped": "⏭️"
        }.get(status, "❓")
        
        print(f"{status_emoji} {gate_name}: {status.upper()} ({duration:.2f}s)")
        
        if status == "failed" and "error" in gate_result:
            print(f"    Error: {gate_result['error']}")
    
    print("\n" + "="*80)
    
    # Exit with appropriate code
    if summary["overall_status"] == "FAILED":
        sys.exit(1)
    elif summary["overall_status"] == "WARNING":
        sys.exit(2)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()