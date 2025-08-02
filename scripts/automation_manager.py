#!/usr/bin/env python3
"""Automation management for neoRL-industrial-gym.

This script manages various automation tasks including:
- Dependency updates
- Code quality monitoring
- Repository maintenance
- Automated reporting
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutomationManager:
    """Manage automated tasks and processes."""
    
    def __init__(self, repo_path: Path, config_path: Optional[Path] = None):
        self.repo_path = repo_path
        self.config_path = config_path or repo_path / ".github" / "project-metrics.json"
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load automation configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
            return {"automation": {"enabled_automations": []}}
    
    def _run_command(self, cmd: List[str], cwd: Path = None) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"✓ Command succeeded: {' '.join(cmd[:3])}")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Command failed: {' '.join(cmd)}")
            logger.error(f"Error output: {e.stderr}")
            raise
    
    def update_dependencies(self) -> Dict[str, Any]:
        """Update project dependencies."""
        logger.info("🔄 Starting dependency update process")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "updates_available": [],
            "updates_applied": [],
            "errors": []
        }
        
        try:
            # Check for outdated packages
            result = self._run_command(["pip", "list", "--outdated", "--format=json"])
            if result.stdout:
                outdated = json.loads(result.stdout)
                results["updates_available"] = outdated
                logger.info(f"Found {len(outdated)} outdated packages")
                
                # Update packages (be careful in production)
                for package in outdated[:5]:  # Limit to 5 updates at once
                    package_name = package["name"]
                    try:
                        self._run_command(["pip", "install", "--upgrade", package_name])
                        results["updates_applied"].append(package)
                        logger.info(f"✓ Updated {package_name}")
                    except subprocess.CalledProcessError as e:
                        error_msg = f"Failed to update {package_name}: {e}"
                        results["errors"].append(error_msg)
                        logger.error(error_msg)
            
            # Update requirements files
            self._run_command(["pip", "freeze", ">", "requirements.txt"])
            
        except Exception as e:
            error_msg = f"Dependency update failed: {e}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def check_code_quality(self) -> Dict[str, Any]:
        """Run code quality checks and report issues."""
        logger.info("🔍 Running code quality checks")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "linting": {"passed": False, "issues": []},
            "formatting": {"passed": False, "issues": []},
            "type_checking": {"passed": False, "issues": []},
            "security": {"passed": False, "issues": []}
        }
        
        # Linting with ruff
        try:
            result = self._run_command(["ruff", "check", "src/", "--format=json"])
            results["linting"]["passed"] = True
            logger.info("✓ Linting passed")
        except subprocess.CalledProcessError as e:
            if e.stdout:
                try:
                    issues = json.loads(e.stdout)
                    results["linting"]["issues"] = issues
                    logger.warning(f"Linting found {len(issues)} issues")
                except json.JSONDecodeError:
                    results["linting"]["issues"] = [{"message": e.stdout}]
        
        # Code formatting
        try:
            self._run_command(["black", "--check", "src/", "tests/"])
            results["formatting"]["passed"] = True
            logger.info("✓ Code formatting check passed")
        except subprocess.CalledProcessError as e:
            results["formatting"]["issues"] = [{"message": "Code formatting issues found"}]
            logger.warning("Code formatting issues detected")
        
        # Type checking
        try:
            result = self._run_command(["mypy", "src/"])
            results["type_checking"]["passed"] = True
            logger.info("✓ Type checking passed")
        except subprocess.CalledProcessError as e:
            results["type_checking"]["issues"] = [{"message": e.stdout}]
            logger.warning("Type checking issues found")
        
        # Security check
        try:
            result = self._run_command(["bandit", "-r", "src/", "-f", "json"])
            bandit_results = json.loads(result.stdout) if result.stdout else {"results": []}
            high_severity = [r for r in bandit_results.get("results", []) if r.get("issue_severity") == "HIGH"]
            if high_severity:
                results["security"]["issues"] = high_severity
                logger.warning(f"Found {len(high_severity)} high-severity security issues")
            else:
                results["security"]["passed"] = True
                logger.info("✓ Security check passed")
        except subprocess.CalledProcessError as e:
            results["security"]["issues"] = [{"message": "Security scan failed"}]
            logger.error("Security scan failed")
        
        return results
    
    def clean_repository(self) -> Dict[str, Any]:
        """Clean up repository artifacts and temporary files."""
        logger.info("🧹 Cleaning repository")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "cleaned_items": [],
            "space_freed_mb": 0
        }
        
        # Directories and files to clean
        cleanup_patterns = [
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "htmlcov",
            ".coverage",
            "dist",
            "build",
            "*.egg-info",
            ".DS_Store",
            "Thumbs.db"
        ]
        
        total_size_before = self._get_directory_size(self.repo_path)
        
        for pattern in cleanup_patterns:
            try:
                if pattern.startswith("*."):
                    # File pattern
                    files = list(self.repo_path.rglob(pattern))
                    for file_path in files:
                        if file_path.is_file():
                            size = file_path.stat().st_size
                            file_path.unlink()
                            results["cleaned_items"].append(str(file_path))
                            logger.debug(f"Removed file: {file_path}")
                else:
                    # Directory pattern
                    dirs = list(self.repo_path.rglob(pattern))
                    for dir_path in dirs:
                        if dir_path.is_dir():
                            import shutil
                            shutil.rmtree(dir_path)
                            results["cleaned_items"].append(str(dir_path))
                            logger.debug(f"Removed directory: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to clean {pattern}: {e}")
        
        total_size_after = self._get_directory_size(self.repo_path)
        results["space_freed_mb"] = round((total_size_before - total_size_after) / (1024 * 1024), 2)
        
        logger.info(f"✓ Cleaned {len(results['cleaned_items'])} items, freed {results['space_freed_mb']} MB")
        return results
    
    def _get_directory_size(self, path: Path) -> int:
        """Get the total size of a directory in bytes."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        pass
        except Exception:
            pass
        return total_size
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate project configuration files."""
        logger.info("✅ Validating configuration files")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "files_checked": [],
            "valid_files": [],
            "invalid_files": [],
            "warnings": []
        }
        
        config_files = [
            ("pyproject.toml", self._validate_toml),
            ("docker-compose.yml", self._validate_yaml),
            (".pre-commit-config.yaml", self._validate_yaml),
            ("package.json", self._validate_json),
            (".github/project-metrics.json", self._validate_json)
        ]
        
        for filename, validator in config_files:
            file_path = self.repo_path / filename
            results["files_checked"].append(filename)
            
            if not file_path.exists():
                results["warnings"].append(f"{filename} does not exist")
                continue
            
            try:
                is_valid, message = validator(file_path)
                if is_valid:
                    results["valid_files"].append(filename)
                    logger.debug(f"✓ {filename} is valid")
                else:
                    results["invalid_files"].append({"file": filename, "error": message})
                    logger.warning(f"✗ {filename} is invalid: {message}")
            except Exception as e:
                results["invalid_files"].append({"file": filename, "error": str(e)})
                logger.error(f"✗ Failed to validate {filename}: {e}")
        
        return results
    
    def _validate_toml(self, file_path: Path) -> tuple[bool, str]:
        """Validate TOML file."""
        try:
            import tomllib
            with open(file_path, 'rb') as f:
                tomllib.load(f)
            return True, "Valid TOML"
        except ImportError:
            # Fallback for older Python versions
            try:
                import toml
                with open(file_path, 'r') as f:
                    toml.load(f)
                return True, "Valid TOML"
            except ImportError:
                return True, "TOML validation skipped (no parser available)"
        except Exception as e:
            return False, str(e)
    
    def _validate_yaml(self, file_path: Path) -> tuple[bool, str]:
        """Validate YAML file."""
        try:
            import yaml
            with open(file_path, 'r') as f:
                yaml.safe_load(f)
            return True, "Valid YAML"
        except ImportError:
            return True, "YAML validation skipped (PyYAML not available)"
        except Exception as e:
            return False, str(e)
    
    def _validate_json(self, file_path: Path) -> tuple[bool, str]:
        """Validate JSON file."""
        try:
            with open(file_path, 'r') as f:
                json.load(f)
            return True, "Valid JSON"
        except Exception as e:
            return False, str(e)
    
    def generate_maintenance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive maintenance report."""
        logger.info("📊 Generating maintenance report")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "repository": str(self.repo_path),
            "sections": {}
        }
        
        # Repository status
        try:
            git_status = self._run_command(["git", "status", "--porcelain"])
            uncommitted_files = len(git_status.stdout.split('\n')) if git_status.stdout.strip() else 0
            report["sections"]["repository_status"] = {
                "uncommitted_files": uncommitted_files,
                "clean": uncommitted_files == 0
            }
        except Exception as e:
            report["sections"]["repository_status"] = {"error": str(e)}
        
        # Disk usage
        repo_size_mb = round(self._get_directory_size(self.repo_path) / (1024 * 1024), 2)
        report["sections"]["disk_usage"] = {
            "repository_size_mb": repo_size_mb,
            "large_files": self._find_large_files(self.repo_path, min_size_mb=10)
        }
        
        # Dependencies
        try:
            result = self._run_command(["pip", "list", "--outdated", "--format=json"])
            outdated = json.loads(result.stdout) if result.stdout else []
            report["sections"]["dependencies"] = {
                "outdated_count": len(outdated),
                "outdated_packages": [pkg["name"] for pkg in outdated[:10]]  # Top 10
            }
        except Exception as e:
            report["sections"]["dependencies"] = {"error": str(e)}
        
        # Security status
        try:
            result = self._run_command(["safety", "check", "--json"])
            safety_results = json.loads(result.stdout) if result.stdout else []
            report["sections"]["security"] = {
                "vulnerabilities_count": len(safety_results),
                "critical_vulnerabilities": [v for v in safety_results if v.get("severity") == "critical"]
            }
        except Exception:
            report["sections"]["security"] = {"status": "safety check not available"}
        
        # Test status
        test_dir = self.repo_path / "tests"
        if test_dir.exists():
            test_files = list(test_dir.rglob("test_*.py"))
            report["sections"]["testing"] = {
                "test_files": len(test_files),
                "test_directory_exists": True
            }
        else:
            report["sections"]["testing"] = {"test_directory_exists": False}
        
        return report
    
    def _find_large_files(self, path: Path, min_size_mb: int = 10) -> List[Dict[str, Any]]:
        """Find files larger than the specified size."""
        large_files = []
        min_size_bytes = min_size_mb * 1024 * 1024
        
        try:
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    try:
                        size = file_path.stat().st_size
                        if size > min_size_bytes:
                            large_files.append({
                                "path": str(file_path.relative_to(path)),
                                "size_mb": round(size / (1024 * 1024), 2)
                            })
                    except (OSError, FileNotFoundError):
                        pass
        except Exception:
            pass
        
        return sorted(large_files, key=lambda x: x["size_mb"], reverse=True)
    
    def run_automation_suite(self, tasks: List[str] = None) -> Dict[str, Any]:
        """Run a suite of automation tasks."""
        if tasks is None:
            tasks = ["quality", "config", "clean", "report"]
        
        logger.info(f"🤖 Running automation suite: {', '.join(tasks)}")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "tasks_requested": tasks,
            "task_results": {},
            "summary": {
                "total_tasks": len(tasks),
                "successful_tasks": 0,
                "failed_tasks": 0
            }
        }
        
        task_mapping = {
            "dependencies": self.update_dependencies,
            "quality": self.check_code_quality,
            "clean": self.clean_repository,
            "config": self.validate_configuration,
            "report": self.generate_maintenance_report
        }
        
        for task in tasks:
            if task in task_mapping:
                logger.info(f"▶️  Running task: {task}")
                try:
                    task_result = task_mapping[task]()
                    results["task_results"][task] = {
                        "status": "success",
                        "result": task_result
                    }
                    results["summary"]["successful_tasks"] += 1
                    logger.info(f"✅ Task completed: {task}")
                except Exception as e:
                    results["task_results"][task] = {
                        "status": "failed",
                        "error": str(e)
                    }
                    results["summary"]["failed_tasks"] += 1
                    logger.error(f"❌ Task failed: {task} - {e}")
            else:
                logger.warning(f"⚠️  Unknown task: {task}")
                results["task_results"][task] = {
                    "status": "skipped",
                    "reason": "Unknown task"
                }
        
        logger.info(f"🏁 Automation suite completed: {results['summary']['successful_tasks']}/{results['summary']['total_tasks']} tasks successful")
        return results


def main():
    """Main function for automation management."""
    parser = argparse.ArgumentParser(description="Automation manager for neoRL-industrial-gym")
    parser.add_argument("--repo-path", type=Path, default=Path.cwd(), help="Repository path")
    parser.add_argument("--config", type=Path, help="Configuration file")
    parser.add_argument("--task", choices=["dependencies", "quality", "clean", "config", "report", "suite"], 
                       help="Specific task to run")
    parser.add_argument("--tasks", nargs="+", help="Multiple tasks to run")
    parser.add_argument("--output", type=Path, help="Output file for results")
    parser.add_argument("--format", choices=["json", "text"], default="json", help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize automation manager
    manager = AutomationManager(args.repo_path, args.config)
    
    # Determine what to run
    if args.task == "suite" or (not args.task and not args.tasks):
        # Run default automation suite
        results = manager.run_automation_suite()
    elif args.task:
        # Run single task
        task_mapping = {
            "dependencies": manager.update_dependencies,
            "quality": manager.check_code_quality,
            "clean": manager.clean_repository,
            "config": manager.validate_configuration,
            "report": manager.generate_maintenance_report
        }
        results = task_mapping[args.task]()
    elif args.tasks:
        # Run specific tasks
        results = manager.run_automation_suite(args.tasks)
    
    # Output results
    if args.format == "json":
        output = json.dumps(results, indent=2)
    else:
        output = f"Automation completed at {results.get('timestamp', 'unknown time')}\n"
        output += f"Summary: {results.get('summary', {})}\n"
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        logger.info(f"Results saved to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()