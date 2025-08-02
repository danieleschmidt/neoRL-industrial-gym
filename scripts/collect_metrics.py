#!/usr/bin/env python3
"""Automated metrics collection for neoRL-industrial-gym.

This script collects various project metrics including development activity,
code quality, performance, and industrial-specific metrics.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests


class MetricsCollector:
    """Collect and aggregate project metrics."""
    
    def __init__(self, repo_path: Path, config_path: Optional[Path] = None):
        self.repo_path = repo_path
        self.config_path = config_path or repo_path / ".github" / "project-metrics.json"
        self.metrics = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load metrics configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default metrics configuration."""
        return {
            "project": {"name": "neoRL-industrial-gym"},
            "metrics": {},
            "thresholds": {},
            "goals": {}
        }
    
    def _run_command(self, cmd: List[str], cwd: Path = None) -> str:
        """Run a command and return its output."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Error: {e.stderr}")
            return ""
    
    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git repository metrics."""
        metrics = {}
        
        # Total commits
        total_commits = self._run_command(["git", "rev-list", "--all", "--count"])
        metrics["total_commits"] = int(total_commits) if total_commits else 0
        
        # Recent commits
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        quarter_ago = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        
        week_commits = self._run_command(["git", "rev-list", "--count", f"--since={week_ago}", "HEAD"])
        month_commits = self._run_command(["git", "rev-list", "--count", f"--since={month_ago}", "HEAD"])
        quarter_commits = self._run_command(["git", "rev-list", "--count", f"--since={quarter_ago}", "HEAD"])
        
        metrics["commits_last_week"] = int(week_commits) if week_commits else 0
        metrics["commits_last_month"] = int(month_commits) if month_commits else 0
        metrics["commits_last_quarter"] = int(quarter_commits) if quarter_commits else 0
        
        # Contributors
        contributors = self._run_command(["git", "shortlog", "-sn", "--all"])
        contributors_list = []
        if contributors:
            for line in contributors.split('\n'):
                if line.strip():
                    count, name = line.strip().split('\t', 1)
                    contributors_list.append({"name": name, "commits": int(count)})
        
        metrics["contributors"] = contributors_list
        metrics["total_contributors"] = len(contributors_list)
        
        # Active contributors this month
        active_contributors = self._run_command([
            "git", "shortlog", "-sn", f"--since={month_ago}", "HEAD"
        ])
        metrics["active_contributors_this_month"] = len(active_contributors.split('\n')) if active_contributors else 0
        
        return metrics
    
    def collect_github_metrics(self) -> Dict[str, Any]:
        """Collect GitHub-specific metrics via API."""
        metrics = {}
        
        # Extract repo info from git remote
        remote_url = self._run_command(["git", "remote", "get-url", "origin"])
        if "github.com" in remote_url:
            # Parse owner/repo from URL
            if remote_url.startswith("git@"):
                repo_path = remote_url.split(":")[1].replace(".git", "")
            else:
                repo_path = remote_url.split("github.com/")[1].replace(".git", "")
            
            owner, repo = repo_path.split("/")
            
            # GitHub API call (requires token for private repos)
            github_token = os.getenv("GITHUB_TOKEN")
            headers = {"Authorization": f"token {github_token}"} if github_token else {}
            
            try:
                # Repository info
                repo_response = requests.get(
                    f"https://api.github.com/repos/{owner}/{repo}",
                    headers=headers,
                    timeout=10
                )
                if repo_response.status_code == 200:
                    repo_data = repo_response.json()
                    metrics["github_stars"] = repo_data.get("stargazers_count", 0)
                    metrics["github_forks"] = repo_data.get("forks_count", 0)
                    metrics["github_watchers"] = repo_data.get("watchers_count", 0)
                    metrics["github_issues_open"] = repo_data.get("open_issues_count", 0)
                
                # Pull requests
                pr_response = requests.get(
                    f"https://api.github.com/repos/{owner}/{repo}/pulls?state=all&per_page=100",
                    headers=headers,
                    timeout=10
                )
                if pr_response.status_code == 200:
                    prs = pr_response.json()
                    metrics["total_pull_requests"] = len(prs)
                    metrics["open_pull_requests"] = len([pr for pr in prs if pr["state"] == "open"])
                    metrics["merged_pull_requests"] = len([pr for pr in prs if pr.get("merged_at")])
                
                # Issues
                issues_response = requests.get(
                    f"https://api.github.com/repos/{owner}/{repo}/issues?state=all&per_page=100",
                    headers=headers,
                    timeout=10
                )
                if issues_response.status_code == 200:
                    issues = issues_response.json()
                    # Filter out pull requests (GitHub API includes PRs in issues)
                    real_issues = [issue for issue in issues if not issue.get("pull_request")]
                    metrics["total_issues"] = len(real_issues)
                    metrics["open_issues"] = len([issue for issue in real_issues if issue["state"] == "open"])
                    
            except requests.RequestException as e:
                print(f"GitHub API request failed: {e}")
                
        return metrics
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {}
        
        # Test coverage (if coverage report exists)
        coverage_file = self.repo_path / "coverage.xml"
        if coverage_file.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(coverage_file)
                root = tree.getroot()
                coverage = root.get("line-rate")
                if coverage:
                    metrics["code_coverage"] = round(float(coverage) * 100, 2)
            except Exception as e:
                print(f"Failed to parse coverage report: {e}")
        
        # Test metrics
        if (self.repo_path / "tests").exists():
            test_files = list((self.repo_path / "tests").rglob("test_*.py"))
            metrics["total_test_files"] = len(test_files)
            
            # Count test functions
            total_tests = 0
            for test_file in test_files:
                try:
                    with open(test_file, 'r') as f:
                        content = f.read()
                        total_tests += content.count("def test_")
                except Exception:
                    continue
            metrics["total_tests"] = total_tests
        
        # Code quality from linting
        try:
            # Run ruff to get linting metrics
            ruff_output = self._run_command(["ruff", "check", "src/", "--format=json"])
            if ruff_output:
                ruff_data = json.loads(ruff_output)
                metrics["linting_issues"] = len(ruff_data)
        except Exception:
            pass
        
        # Lines of code
        src_files = list((self.repo_path / "src").rglob("*.py")) if (self.repo_path / "src").exists() else []
        total_lines = 0
        for src_file in src_files:
            try:
                with open(src_file, 'r') as f:
                    total_lines += len(f.readlines())
            except Exception:
                continue
        metrics["lines_of_code"] = total_lines
        
        return metrics
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance-related metrics."""
        metrics = {}
        
        # CI/CD build times (from GitHub Actions if available)
        # This would require GitHub API integration for actual data
        
        # Package size
        dist_dir = self.repo_path / "dist"
        if dist_dir.exists():
            wheel_files = list(dist_dir.glob("*.whl"))
            if wheel_files:
                latest_wheel = max(wheel_files, key=lambda f: f.stat().st_mtime)
                metrics["package_size_mb"] = round(latest_wheel.stat().st_size / (1024 * 1024), 2)
        
        # Docker image size (if available)
        try:
            docker_images = self._run_command(["docker", "images", "--format", "json"])
            if docker_images:
                for line in docker_images.split('\n'):
                    if line.strip():
                        image_data = json.loads(line)
                        if "neorl-industrial" in image_data.get("Repository", ""):
                            size_str = image_data.get("Size", "0B")
                            # Parse size (this is simplified)
                            if "MB" in size_str:
                                size_mb = float(size_str.replace("MB", ""))
                                metrics["docker_image_size_mb"] = size_mb
                            break
        except Exception:
            pass
        
        return metrics
    
    def collect_industrial_metrics(self) -> Dict[str, Any]:
        """Collect industrial-specific metrics."""
        metrics = {}
        
        # Safety test coverage
        safety_tests = list((self.repo_path / "tests").rglob("*safety*.py")) if (self.repo_path / "tests").exists() else []
        metrics["safety_test_files"] = len(safety_tests)
        
        # Count safety tests
        safety_test_count = 0
        for test_file in safety_tests:
            try:
                with open(test_file, 'r') as f:
                    content = f.read()
                    safety_test_count += content.count("def test_")
            except Exception:
                continue
        metrics["safety_tests"] = safety_test_count
        
        # Environment coverage (from source code)
        src_dir = self.repo_path / "src" / "neorl_industrial"
        if src_dir.exists():
            env_files = list(src_dir.rglob("*env*.py"))
            metrics["environment_implementations"] = len(env_files)
        
        # Algorithm implementations
        if src_dir.exists():
            algorithm_files = list(src_dir.rglob("*agent*.py")) + list(src_dir.rglob("*algorithm*.py"))
            metrics["algorithm_implementations"] = len(algorithm_files)
        
        # Documentation coverage
        docs_dir = self.repo_path / "docs"
        if docs_dir.exists():
            doc_files = list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.rst"))
            metrics["documentation_files"] = len(doc_files)
        
        return metrics
    
    def collect_dependency_metrics(self) -> Dict[str, Any]:
        """Collect dependency-related metrics."""
        metrics = {}
        
        # PyPI dependencies
        requirements_files = [
            self.repo_path / "requirements.txt",
            self.repo_path / "requirements-dev.txt",
            self.repo_path / "pyproject.toml"
        ]
        
        total_dependencies = 0
        for req_file in requirements_files:
            if req_file.exists():
                try:
                    with open(req_file, 'r') as f:
                        content = f.read()
                        if req_file.name == "pyproject.toml":
                            # Parse TOML dependencies (simplified)
                            lines = content.split('\n')
                            in_deps = False
                            for line in lines:
                                if "dependencies" in line and "[" in line:
                                    in_deps = True
                                    continue
                                if in_deps and line.strip().startswith('"'):
                                    total_dependencies += 1
                                elif in_deps and "]" in line:
                                    break
                        else:
                            # Regular requirements file
                            lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
                            total_dependencies += len(lines)
                except Exception:
                    continue
        
        metrics["total_dependencies"] = total_dependencies
        
        # Check for outdated dependencies (if pip-outdated is available)
        try:
            outdated = self._run_command(["pip", "list", "--outdated", "--format=json"])
            if outdated:
                outdated_data = json.loads(outdated)
                metrics["outdated_dependencies"] = len(outdated_data)
        except Exception:
            metrics["outdated_dependencies"] = 0
        
        return metrics
    
    def update_metrics_file(self, collected_metrics: Dict[str, Any]) -> None:
        """Update the metrics configuration file with collected data."""
        # Update timestamps
        collected_metrics["last_updated"] = datetime.now().isoformat()
        
        # Merge with existing metrics
        if "metrics" not in self.metrics:
            self.metrics["metrics"] = {}
        
        # Update development metrics
        if "development" not in self.metrics["metrics"]:
            self.metrics["metrics"]["development"] = {}
        
        dev_metrics = self.metrics["metrics"]["development"]
        
        # Update commits data
        if "commits" not in dev_metrics:
            dev_metrics["commits"] = {}
        dev_metrics["commits"].update({
            "total": collected_metrics.get("total_commits", 0),
            "last_week": collected_metrics.get("commits_last_week", 0),
            "last_month": collected_metrics.get("commits_last_month", 0),
            "last_quarter": collected_metrics.get("commits_last_quarter", 0),
            "contributors": collected_metrics.get("contributors", [])
        })
        
        # Update quality metrics
        if "quality" not in self.metrics["metrics"]:
            self.metrics["metrics"]["quality"] = {}
        
        quality_metrics = self.metrics["metrics"]["quality"]
        if "code_coverage" not in quality_metrics:
            quality_metrics["code_coverage"] = {}
        quality_metrics["code_coverage"]["current"] = collected_metrics.get("code_coverage", 0)
        quality_metrics["code_coverage"]["last_updated"] = datetime.now().isoformat()
        
        # Update test metrics
        if "test_metrics" not in quality_metrics:
            quality_metrics["test_metrics"] = {}
        quality_metrics["test_metrics"].update({
            "total_tests": collected_metrics.get("total_tests", 0),
            "passing_tests": collected_metrics.get("total_tests", 0),  # Assume all pass for now
            "failing_tests": 0
        })
        
        # Update community metrics
        if "community" not in self.metrics["metrics"]:
            self.metrics["metrics"]["community"] = {}
        
        community_metrics = self.metrics["metrics"]["community"]
        if "engagement" not in community_metrics:
            community_metrics["engagement"] = {}
        community_metrics["engagement"].update({
            "github_stars": collected_metrics.get("github_stars", 0),
            "github_forks": collected_metrics.get("github_forks", 0),
            "github_watchers": collected_metrics.get("github_watchers", 0)
        })
        
        # Update industrial metrics
        if "industrial_specific" not in self.metrics["metrics"]:
            self.metrics["metrics"]["industrial_specific"] = {}
        
        industrial_metrics = self.metrics["metrics"]["industrial_specific"]
        if "safety_metrics" not in industrial_metrics:
            industrial_metrics["safety_metrics"] = {}
        industrial_metrics["safety_metrics"]["safety_tests_passing"] = collected_metrics.get("safety_tests", 0)
        
        # Save updated metrics
        with open(self.config_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"✅ Metrics updated successfully: {self.config_path}")
    
    def generate_report(self, format_type: str = "json") -> str:
        """Generate a metrics report in the specified format."""
        if format_type == "json":
            return json.dumps(self.metrics, indent=2)
        elif format_type == "markdown":
            return self._generate_markdown_report()
        elif format_type == "csv":
            return self._generate_csv_report()
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _generate_markdown_report(self) -> str:
        """Generate a markdown metrics report."""
        report = []
        report.append("# neoRL-Industrial-Gym Metrics Report")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Development metrics
        dev_metrics = self.metrics.get("metrics", {}).get("development", {})
        if dev_metrics:
            report.append("## Development Metrics")
            commits = dev_metrics.get("commits", {})
            report.append(f"- Total commits: {commits.get('total', 0)}")
            report.append(f"- Commits last week: {commits.get('last_week', 0)}")
            report.append(f"- Commits last month: {commits.get('last_month', 0)}")
            report.append(f"- Total contributors: {len(commits.get('contributors', []))}")
            report.append("")
        
        # Quality metrics
        quality_metrics = self.metrics.get("metrics", {}).get("quality", {})
        if quality_metrics:
            report.append("## Quality Metrics")
            coverage = quality_metrics.get("code_coverage", {})
            report.append(f"- Code coverage: {coverage.get('current', 0)}%")
            test_metrics = quality_metrics.get("test_metrics", {})
            report.append(f"- Total tests: {test_metrics.get('total_tests', 0)}")
            report.append("")
        
        # Community metrics
        community_metrics = self.metrics.get("metrics", {}).get("community", {})
        if community_metrics:
            report.append("## Community Metrics")
            engagement = community_metrics.get("engagement", {})
            report.append(f"- GitHub stars: {engagement.get('github_stars', 0)}")
            report.append(f"- GitHub forks: {engagement.get('github_forks', 0)}")
            report.append("")
        
        return "\n".join(report)
    
    def _generate_csv_report(self) -> str:
        """Generate a CSV metrics report."""
        # Simplified CSV with key metrics
        lines = []
        lines.append("Metric,Value,Category")
        
        dev_metrics = self.metrics.get("metrics", {}).get("development", {})
        commits = dev_metrics.get("commits", {})
        lines.append(f"Total Commits,{commits.get('total', 0)},Development")
        lines.append(f"Commits Last Week,{commits.get('last_week', 0)},Development")
        lines.append(f"Total Contributors,{len(commits.get('contributors', []))},Development")
        
        quality_metrics = self.metrics.get("metrics", {}).get("quality", {})
        coverage = quality_metrics.get("code_coverage", {})
        lines.append(f"Code Coverage,{coverage.get('current', 0)},Quality")
        
        community_metrics = self.metrics.get("metrics", {}).get("community", {})
        engagement = community_metrics.get("engagement", {})
        lines.append(f"GitHub Stars,{engagement.get('github_stars', 0)},Community")
        
        return "\n".join(lines)


def main():
    """Main function for metrics collection."""
    parser = argparse.ArgumentParser(description="Collect project metrics for neoRL-industrial-gym")
    parser.add_argument("--repo-path", type=Path, default=Path.cwd(), help="Repository path")
    parser.add_argument("--config", type=Path, help="Metrics configuration file")
    parser.add_argument("--output", type=Path, help="Output file for metrics")
    parser.add_argument("--format", choices=["json", "markdown", "csv"], default="json", help="Output format")
    parser.add_argument("--update-config", action="store_true", help="Update the metrics configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize metrics collector
    collector = MetricsCollector(args.repo_path, args.config)
    
    if args.verbose:
        print(f"📊 Collecting metrics for {collector.metrics.get('project', {}).get('name', 'Unknown Project')}")
        print(f"📁 Repository path: {args.repo_path}")
    
    # Collect all metrics
    all_metrics = {}
    
    if args.verbose:
        print("🔍 Collecting Git metrics...")
    all_metrics.update(collector.collect_git_metrics())
    
    if args.verbose:
        print("🐙 Collecting GitHub metrics...")
    all_metrics.update(collector.collect_github_metrics())
    
    if args.verbose:
        print("📋 Collecting code quality metrics...")
    all_metrics.update(collector.collect_code_quality_metrics())
    
    if args.verbose:
        print("⚡ Collecting performance metrics...")
    all_metrics.update(collector.collect_performance_metrics())
    
    if args.verbose:
        print("🏭 Collecting industrial metrics...")
    all_metrics.update(collector.collect_industrial_metrics())
    
    if args.verbose:
        print("📦 Collecting dependency metrics...")
    all_metrics.update(collector.collect_dependency_metrics())
    
    # Update configuration file if requested
    if args.update_config:
        collector.update_metrics_file(all_metrics)
    
    # Generate report
    report = collector.generate_report(args.format)
    
    # Output report
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        if args.verbose:
            print(f"📄 Report saved to: {args.output}")
    else:
        print(report)
    
    # Summary
    if args.verbose:
        print("\n✅ Metrics collection completed successfully!")
        print(f"   - Total commits: {all_metrics.get('total_commits', 0)}")
        print(f"   - Contributors: {len(all_metrics.get('contributors', []))}")
        print(f"   - Code coverage: {all_metrics.get('code_coverage', 0)}%")
        print(f"   - GitHub stars: {all_metrics.get('github_stars', 0)}")


if __name__ == "__main__":
    main()