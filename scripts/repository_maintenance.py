#!/usr/bin/env python3
"""Repository maintenance automation for neoRL-industrial-gym.

This script performs automated repository maintenance tasks including:
- Dependency updates and vulnerability management
- Code quality monitoring and technical debt tracking
- Performance benchmark tracking
- Security audit automation
- Documentation freshness validation
- License compliance checking
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import requests
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('repository_maintenance.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RepositoryMaintainer:
    """Automated repository maintenance system."""
    
    def __init__(self, repo_path: Path, config_path: Optional[Path] = None):
        self.repo_path = repo_path
        self.config_path = config_path or repo_path / ".github" / "project-metrics.json"
        self.config = self._load_config()
        self.maintenance_results = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load maintenance configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "automation": {
                "enabled_automations": [
                    "dependency_updates",
                    "security_scanning",
                    "code_quality_check",
                    "documentation_update"
                ]
            },
            "thresholds": {
                "critical": {"dependency_age_days": 180},
                "warning": {"dependency_age_days": 90}
            }
        }
    
    def _run_command(self, cmd: List[str], cwd: Path = None, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.repo_path,
                capture_output=capture_output,
                text=True,
                timeout=300
            )
            return result
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(cmd)}")
            raise
        except Exception as e:
            logger.error(f"Command failed: {' '.join(cmd)}, Error: {e}")
            raise
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check and update dependencies."""
        logger.info("🔍 Checking dependencies...")
        
        results = {
            "outdated_packages": [],
            "security_vulnerabilities": [],
            "license_issues": [],
            "recommendations": []
        }
        
        # Check for outdated Python packages
        try:
            result = self._run_command(["pip", "list", "--outdated", "--format=json"])
            if result.returncode == 0:
                outdated = json.loads(result.stdout)
                results["outdated_packages"] = outdated
                logger.info(f"Found {len(outdated)} outdated packages")
        except Exception as e:
            logger.error(f"Failed to check outdated packages: {e}")
        
        # Security vulnerability scan
        try:
            result = self._run_command(["safety", "check", "--json"])
            if result.returncode == 0:
                safety_results = json.loads(result.stdout)
                results["security_vulnerabilities"] = safety_results
            else:
                # Safety returns non-zero when vulnerabilities found
                if result.stdout:
                    safety_results = json.loads(result.stdout)
                    results["security_vulnerabilities"] = safety_results
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
        
        # License compliance check
        try:
            result = self._run_command(["pip-licenses", "--format=json"])
            if result.returncode == 0:
                licenses = json.loads(result.stdout)
                # Check for problematic licenses
                problematic_licenses = ["GPL", "AGPL", "LGPL"]
                for package in licenses:
                    if any(lic in package.get("License", "") for lic in problematic_licenses):
                        results["license_issues"].append(package)
        except Exception as e:
            logger.error(f"License check failed: {e}")
        
        # Generate recommendations
        if results["outdated_packages"]:
            results["recommendations"].append("Update outdated packages to latest versions")
        if results["security_vulnerabilities"]:
            results["recommendations"].append("Address security vulnerabilities immediately")
        if results["license_issues"]:
            results["recommendations"].append("Review license compatibility issues")
        
        self.maintenance_results["dependencies"] = results
        return results
    
    def analyze_code_quality(self) -> Dict[str, Any]:
        """Analyze code quality metrics."""
        logger.info("📊 Analyzing code quality...")
        
        results = {
            "complexity": {},
            "coverage": {},
            "duplication": {},
            "technical_debt": {},
            "recommendations": []
        }
        
        # Run pytest with coverage
        try:
            result = self._run_command([
                "pytest", "tests/", "--cov=neorl_industrial", 
                "--cov-report=json", "--cov-report=term-missing"
            ])
            
            # Load coverage report
            coverage_file = self.repo_path / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    results["coverage"] = {
                        "total_coverage": coverage_data.get("totals", {}).get("percent_covered", 0),
                        "file_coverage": {
                            file: data.get("summary", {}).get("percent_covered", 0)
                            for file, data in coverage_data.get("files", {}).items()
                        }
                    }
        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}")
        
        # Code complexity analysis using radon
        try:
            result = self._run_command(["radon", "cc", "src/", "--json"])
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                results["complexity"] = complexity_data
        except Exception as e:
            logger.error(f"Complexity analysis failed: {e}")
        
        # Duplication analysis
        try:
            result = self._run_command(["jscpd", "src/", "--reporters", "json", "--output", "duplication-report.json"])
            duplication_file = self.repo_path / "duplication-report.json"
            if duplication_file.exists():
                with open(duplication_file, 'r') as f:
                    duplication_data = json.load(f)
                    results["duplication"] = duplication_data
        except Exception as e:
            logger.error(f"Duplication analysis failed: {e}")
        
        # Generate quality recommendations
        coverage = results["coverage"].get("total_coverage", 0)
        threshold = self.config.get("metrics", {}).get("development", {}).get("code_quality", {}).get("coverage_threshold", 85)
        
        if coverage < threshold:
            results["recommendations"].append(f"Increase test coverage from {coverage}% to {threshold}%")
        
        self.maintenance_results["code_quality"] = results
        return results
    
    def validate_security(self) -> Dict[str, Any]:
        """Perform security validation."""
        logger.info("🔒 Performing security validation...")
        
        results = {
            "vulnerability_scan": {},
            "secret_scan": {},
            "container_security": {},
            "recommendations": []
        }
        
        # Vulnerability scanning with multiple tools
        scanners = ["bandit", "semgrep", "safety"]
        
        for scanner in scanners:
            try:
                if scanner == "bandit":
                    result = self._run_command([
                        "bandit", "-r", "src/", "-f", "json", "-o", "bandit-report.json"
                    ])
                    report_file = self.repo_path / "bandit-report.json"
                    
                elif scanner == "semgrep":
                    result = self._run_command([
                        "semgrep", "--config=auto", "src/", "--json", "--output=semgrep-report.json"
                    ])
                    report_file = self.repo_path / "semgrep-report.json"
                    
                elif scanner == "safety":
                    result = self._run_command(["safety", "check", "--json"])
                    if result.stdout:
                        results["vulnerability_scan"][scanner] = json.loads(result.stdout)
                    continue
                
                if report_file.exists():
                    with open(report_file, 'r') as f:
                        scan_results = json.load(f)
                        results["vulnerability_scan"][scanner] = scan_results
                        
            except Exception as e:
                logger.error(f"{scanner} scan failed: {e}")
        
        # Secret scanning
        try:
            result = self._run_command([
                "trufflehog", "filesystem", ".", "--json", "--no-verification"
            ])
            if result.returncode == 0 and result.stdout:
                secrets = [json.loads(line) for line in result.stdout.strip().split('\n') if line]
                results["secret_scan"] = secrets
        except Exception as e:
            logger.error(f"Secret scanning failed: {e}")
        
        # Container security (if Dockerfile exists)
        dockerfile = self.repo_path / "Dockerfile"
        if dockerfile.exists():
            try:
                result = self._run_command(["hadolint", str(dockerfile), "--format", "json"])
                if result.returncode == 0:
                    container_issues = json.loads(result.stdout)
                    results["container_security"] = container_issues
            except Exception as e:
                logger.error(f"Container security scan failed: {e}")
        
        # Generate security recommendations
        total_vulnerabilities = sum(
            len(scan_results) if isinstance(scan_results, list) else 0 
            for scan_results in results["vulnerability_scan"].values()
        )
        
        if total_vulnerabilities > 0:
            results["recommendations"].append(f"Address {total_vulnerabilities} security vulnerabilities")
        
        if results["secret_scan"]:
            results["recommendations"].append("Remove detected secrets from repository")
        
        self.maintenance_results["security"] = results
        return results
    
    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation freshness and completeness."""
        logger.info("📚 Checking documentation...")
        
        results = {
            "stale_docs": [],
            "missing_docs": [],
            "link_check": {},
            "recommendations": []
        }
        
        # Find documentation files
        doc_extensions = ['.md', '.rst', '.txt']
        doc_files = []
        for ext in doc_extensions:
            doc_files.extend(self.repo_path.glob(f"**/*{ext}"))
        
        # Check for stale documentation (not updated in 90 days)
        stale_threshold = datetime.now() - timedelta(days=90)
        
        for doc_file in doc_files:
            try:
                # Get last modified time from git
                result = self._run_command([
                    "git", "log", "-1", "--format=%ct", str(doc_file.relative_to(self.repo_path))
                ])
                if result.returncode == 0:
                    timestamp = int(result.stdout.strip())
                    last_modified = datetime.fromtimestamp(timestamp)
                    
                    if last_modified < stale_threshold:
                        results["stale_docs"].append({
                            "file": str(doc_file.relative_to(self.repo_path)),
                            "last_modified": last_modified.isoformat(),
                            "days_old": (datetime.now() - last_modified).days
                        })
            except Exception as e:
                logger.error(f"Failed to check {doc_file}: {e}")
        
        # Check for missing documentation
        required_docs = [
            "README.md", "CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md",
            "docs/ARCHITECTURE.md", "docs/DEPLOYMENT.md"
        ]
        
        for doc in required_docs:
            doc_path = self.repo_path / doc
            if not doc_path.exists():
                results["missing_docs"].append(doc)
        
        # Link validation (basic check for markdown links)
        try:
            result = self._run_command(["markdown-link-check", "README.md", "--json"])
            if result.returncode == 0:
                link_results = json.loads(result.stdout)
                results["link_check"]["README.md"] = link_results
        except Exception as e:
            logger.error(f"Link check failed: {e}")
        
        # Generate documentation recommendations
        if results["stale_docs"]:
            results["recommendations"].append(f"Update {len(results['stale_docs'])} stale documentation files")
        
        if results["missing_docs"]:
            results["recommendations"].append(f"Create missing documentation: {', '.join(results['missing_docs'])}")
        
        self.maintenance_results["documentation"] = results
        return results
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Run performance optimization checks."""
        logger.info("⚡ Running performance optimization...")
        
        results = {
            "benchmarks": {},
            "profiling": {},
            "optimization_opportunities": [],
            "recommendations": []
        }
        
        # Run performance benchmarks
        try:
            result = self._run_command([
                "python", "scripts/benchmark_suite.py", "--output", "benchmark-results.json"
            ])
            
            benchmark_file = self.repo_path / "benchmark-results.json"
            if benchmark_file.exists():
                with open(benchmark_file, 'r') as f:
                    benchmark_data = json.load(f)
                    results["benchmarks"] = benchmark_data
        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}")
        
        # Memory profiling (if py-spy is available)
        try:
            result = self._run_command(["py-spy", "record", "--output", "profile.svg", "--duration", "10", "--", "python", "-c", "import neorl_industrial; print('Profiling complete')"])
            if result.returncode == 0:
                results["profiling"]["memory"] = "profile.svg generated"
        except Exception as e:
            logger.error(f"Memory profiling failed: {e}")
        
        # Analyze dependencies for performance impact
        large_dependencies = []
        try:
            result = self._run_command(["pip", "show", "-v", "jax", "jaxlib", "mlflow"])
            # This is a simplified check - in practice, you'd parse the output
            if "Size:" in result.stdout:
                results["optimization_opportunities"].append("Consider optimizing large dependencies")
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
        
        self.maintenance_results["performance"] = results
        return results
    
    def cleanup_repository(self) -> Dict[str, Any]:
        """Perform repository cleanup tasks."""
        logger.info("🧹 Performing repository cleanup...")
        
        results = {
            "cleaned_files": [],
            "disk_space_saved": 0,
            "recommendations": []
        }
        
        # Clean up common temporary and cache files
        cleanup_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/.pytest_cache",
            "**/.mypy_cache",
            "**/.ruff_cache",
            "**/htmlcov",
            "**/coverage.xml",
            "**/coverage.json",
            "**/.coverage",
            "**/bandit-report.json",
            "**/semgrep-report.json",
            "**/duplication-report.json"
        ]
        
        total_size = 0
        for pattern in cleanup_patterns:
            paths = list(self.repo_path.glob(pattern))
            for path in paths:
                if path.exists():
                    if path.is_file():
                        total_size += path.stat().st_size
                        path.unlink()
                        results["cleaned_files"].append(str(path.relative_to(self.repo_path)))
                    elif path.is_dir():
                        import shutil
                        dir_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                        total_size += dir_size
                        shutil.rmtree(path)
                        results["cleaned_files"].append(str(path.relative_to(self.repo_path)))
        
        results["disk_space_saved"] = total_size
        
        # Git cleanup
        try:
            self._run_command(["git", "gc", "--prune=now"])
            self._run_command(["git", "remote", "prune", "origin"])
            results["recommendations"].append("Git repository cleaned and optimized")
        except Exception as e:
            logger.error(f"Git cleanup failed: {e}")
        
        self.maintenance_results["cleanup"] = results
        return results
    
    def generate_maintenance_report(self) -> str:
        """Generate a comprehensive maintenance report."""
        logger.info("📝 Generating maintenance report...")
        
        report = []
        report.append("# Repository Maintenance Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        total_issues = 0
        critical_issues = 0
        
        for section, data in self.maintenance_results.items():
            if isinstance(data, dict) and "recommendations" in data:
                issues = len(data["recommendations"])
                total_issues += issues
                # Count critical issues (security, dependencies)
                if section in ["security", "dependencies"]:
                    critical_issues += issues
        
        report.append(f"- Total Issues Found: {total_issues}")
        report.append(f"- Critical Issues: {critical_issues}")
        report.append(f"- Repository Health Score: {max(0, 100 - (total_issues * 5))}%")
        report.append("")
        
        # Detailed Sections
        for section, data in self.maintenance_results.items():
            report.append(f"## {section.title()}")
            
            if isinstance(data, dict) and "recommendations" in data:
                if data["recommendations"]:
                    report.append("### Recommendations:")
                    for rec in data["recommendations"]:
                        report.append(f"- {rec}")
                else:
                    report.append("✅ No issues found")
            
            report.append("")
        
        # Action Items
        report.append("## Action Items")
        priority_actions = []
        
        # Security actions (highest priority)
        if "security" in self.maintenance_results:
            sec_data = self.maintenance_results["security"]
            if sec_data.get("recommendations"):
                priority_actions.extend([f"🔴 SECURITY: {rec}" for rec in sec_data["recommendations"]])
        
        # Dependency actions
        if "dependencies" in self.maintenance_results:
            dep_data = self.maintenance_results["dependencies"]
            if dep_data.get("recommendations"):
                priority_actions.extend([f"🟡 DEPENDENCIES: {rec}" for rec in dep_data["recommendations"]])
        
        # Other actions
        for section, data in self.maintenance_results.items():
            if section not in ["security", "dependencies"] and isinstance(data, dict):
                if data.get("recommendations"):
                    priority_actions.extend([f"🔵 {section.upper()}: {rec}" for rec in data["recommendations"]])
        
        for action in priority_actions:
            report.append(f"- {action}")
        
        report.append("")
        report.append("---")
        report.append("*This report was generated automatically by the repository maintenance system.*")
        
        report_content = "\n".join(report)
        
        # Save report
        report_file = self.repo_path / "maintenance_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"📄 Maintenance report saved to: {report_file}")
        return report_content
    
    def run_maintenance(self, tasks: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run specified maintenance tasks."""
        if tasks is None:
            tasks = [
                "check_dependencies",
                "analyze_code_quality", 
                "validate_security",
                "check_documentation",
                "optimize_performance",
                "cleanup_repository"
            ]
        
        logger.info(f"🔧 Running maintenance tasks: {', '.join(tasks)}")
        
        task_functions = {
            "check_dependencies": self.check_dependencies,
            "analyze_code_quality": self.analyze_code_quality,
            "validate_security": self.validate_security,
            "check_documentation": self.check_documentation,
            "optimize_performance": self.optimize_performance,
            "cleanup_repository": self.cleanup_repository
        }
        
        results = {}
        for task in tasks:
            if task in task_functions:
                try:
                    logger.info(f"Executing task: {task}")
                    start_time = time.time()
                    result = task_functions[task]()
                    execution_time = time.time() - start_time
                    results[task] = {
                        "result": result,
                        "execution_time": execution_time,
                        "status": "success"
                    }
                    logger.info(f"✅ {task} completed in {execution_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"❌ {task} failed: {e}")
                    results[task] = {
                        "error": str(e),
                        "status": "failed"
                    }
            else:
                logger.warning(f"Unknown task: {task}")
        
        # Generate final report
        report = self.generate_maintenance_report()
        results["maintenance_report"] = report
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Repository maintenance automation")
    parser.add_argument("--repo-path", type=Path, default=Path.cwd(), 
                      help="Path to repository (default: current directory)")
    parser.add_argument("--config", type=Path, help="Path to configuration file")
    parser.add_argument("--tasks", nargs="+", 
                      choices=["check_dependencies", "analyze_code_quality", "validate_security", 
                              "check_documentation", "optimize_performance", "cleanup_repository"],
                      help="Specific tasks to run (default: all)")
    parser.add_argument("--output", type=Path, help="Output file for results")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    maintainer = RepositoryMaintainer(args.repo_path, args.config)
    
    if args.dry_run:
        logger.info("🔍 Dry run mode - showing planned tasks:")
        tasks = args.tasks or ["check_dependencies", "analyze_code_quality", "validate_security", 
                             "check_documentation", "optimize_performance", "cleanup_repository"]
        for task in tasks:
            logger.info(f"  - {task}")
        return
    
    try:
        results = maintainer.run_maintenance(args.tasks)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"📄 Results saved to: {args.output}")
        
        # Print summary
        successful_tasks = sum(1 for r in results.values() if isinstance(r, dict) and r.get("status") == "success")
        failed_tasks = sum(1 for r in results.values() if isinstance(r, dict) and r.get("status") == "failed")
        
        logger.info(f"🎉 Maintenance complete: {successful_tasks} successful, {failed_tasks} failed")
        
        if failed_tasks > 0:
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"💥 Maintenance failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()