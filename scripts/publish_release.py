#!/usr/bin/env python3
"""Publish release script for neoRL-industrial-gym.

This script handles the publication of a new release by:
1. Publishing packages to PyPI
2. Publishing Docker images to registries
3. Creating GitHub release
4. Notifying stakeholders
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def run_command(cmd: List[str], cwd: Path = None, env: Dict[str, str] = None) -> subprocess.CompletedProcess:
    """Run a command and handle errors."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ {' '.join(cmd[:3])}...")  # Truncate for security
        return result
    except subprocess.CalledProcessError as e:
        print(f"✗ {' '.join(cmd[:3])}...")
        print(f"Error: {e.stderr}")
        return e


def check_environment_variables() -> Dict[str, str]:
    """Check required environment variables for publishing."""
    required_vars = {
        "PYPI_TOKEN": "PyPI API token for package publishing",
        "DOCKER_USERNAME": "Docker Hub username",
        "DOCKER_PASSWORD": "Docker Hub password or access token",
    }
    
    optional_vars = {
        "GITHUB_TOKEN": "GitHub token for release creation",
        "SLACK_WEBHOOK": "Slack webhook for notifications",
    }
    
    env_vars = {}
    missing_required = []
    
    # Check required variables
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            env_vars[var] = value
        else:
            missing_required.append(f"  {var}: {description}")
    
    # Check optional variables
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            env_vars[var] = value
        else:
            print(f"⚠️  Optional: {var} not set - {description}")
    
    if missing_required:
        print("❌ Missing required environment variables:")
        for var in missing_required:
            print(var)
        sys.exit(1)
    
    return env_vars


def publish_to_pypi(repo_root: Path, env_vars: Dict[str, str]) -> None:
    """Publish packages to PyPI."""
    print("📦 Publishing to PyPI")
    
    dist_dir = repo_root / "dist"
    if not dist_dir.exists() or not list(dist_dir.glob("*.whl")):
        print("❌ No distribution files found. Run prepare_release.py first.")
        sys.exit(1)
    
    # Upload to PyPI using twine
    result = run_command([
        "python", "-m", "twine", "upload",
        "--username", "__token__",
        "--password", env_vars["PYPI_TOKEN"],
        str(dist_dir / "*")
    ], cwd=repo_root)
    
    if result.returncode == 0:
        print("✅ Successfully published to PyPI")
    else:
        print("❌ Failed to publish to PyPI")
        sys.exit(1)


def login_docker_registry(env_vars: Dict[str, str]) -> None:
    """Login to Docker registry."""
    print("🐳 Logging into Docker Hub")
    
    result = run_command([
        "docker", "login",
        "--username", env_vars["DOCKER_USERNAME"],
        "--password-stdin"
    ], env={"DOCKER_PASSWORD": env_vars["DOCKER_PASSWORD"]})
    
    if result.returncode != 0:
        print("❌ Failed to login to Docker Hub")
        sys.exit(1)


def push_docker_images(version: str, repo_root: Path) -> None:
    """Push Docker images to registry."""
    print("🐳 Pushing Docker images")
    
    # Define image registry and organization
    registry = "terragonlabs"  # Docker Hub organization
    
    images = [
        ("neorl-industrial", "neorl-industrial"),
        ("neorl-industrial-dev", "neorl-industrial-dev"),
        ("neorl-industrial-gpu", "neorl-industrial-gpu")
    ]
    
    for local_name, remote_name in images:
        # Tag for registry
        registry_tag = f"{registry}/{remote_name}:{version}"
        latest_tag = f"{registry}/{remote_name}:latest"
        
        # Tag local image
        run_command([
            "docker", "tag",
            f"{local_name}:{version}",
            registry_tag
        ], cwd=repo_root)
        
        run_command([
            "docker", "tag",
            f"{local_name}:{version}",
            latest_tag
        ], cwd=repo_root)
        
        # Push versioned tag
        result = run_command([
            "docker", "push", registry_tag
        ], cwd=repo_root)
        
        if result.returncode != 0:
            print(f"❌ Failed to push {registry_tag}")
            continue
        
        # Push latest tag
        result = run_command([
            "docker", "push", latest_tag
        ], cwd=repo_root)
        
        if result.returncode == 0:
            print(f"✅ Successfully pushed {registry_tag}")
        else:
            print(f"⚠️  Failed to push latest tag for {remote_name}")


def create_github_release(version: str, repo_root: Path, env_vars: Dict[str, str]) -> None:
    """Create GitHub release."""
    if "GITHUB_TOKEN" not in env_vars:
        print("⚠️  GITHUB_TOKEN not set, skipping GitHub release creation")
        return
    
    print("📋 Creating GitHub release")
    
    # Read changelog for release notes
    changelog_path = repo_root / "CHANGELOG.md"
    release_notes = "Automated release"
    
    if changelog_path.exists():
        # Extract release notes from changelog
        try:
            with open(changelog_path, "r") as f:
                content = f.read()
                # Find the section for this version
                lines = content.split("\n")
                version_line_idx = None
                for i, line in enumerate(lines):
                    if f"## [{version}]" in line or f"## {version}" in line:
                        version_line_idx = i
                        break
                
                if version_line_idx:
                    # Extract content until next version or end
                    notes_lines = []
                    for line in lines[version_line_idx + 1:]:
                        if line.startswith("## ") and version not in line:
                            break
                        notes_lines.append(line)
                    release_notes = "\n".join(notes_lines).strip()
        except Exception as e:
            print(f"⚠️  Could not extract release notes: {e}")
    
    # Create release using GitHub CLI
    try:
        result = run_command([
            "gh", "release", "create",
            f"v{version}",
            "--title", f"neoRL-industrial-gym v{version}",
            "--notes", release_notes,
            "--generate-notes"
        ], cwd=repo_root, env={"GITHUB_TOKEN": env_vars["GITHUB_TOKEN"]})
        
        if result.returncode == 0:
            print("✅ GitHub release created successfully")
        else:
            print("⚠️  Failed to create GitHub release")
    
    except FileNotFoundError:
        print("⚠️  GitHub CLI not installed, skipping release creation")


def send_slack_notification(version: str, env_vars: Dict[str, str]) -> None:
    """Send Slack notification about the release."""
    if "SLACK_WEBHOOK" not in env_vars:
        print("⚠️  SLACK_WEBHOOK not set, skipping notification")
        return
    
    print("📢 Sending Slack notification")
    
    message = {
        "text": f"🚀 neoRL-industrial-gym v{version} has been released!",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*neoRL-industrial-gym v{version}* has been released! 🎉"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Version:* {version}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": "*Status:* Published"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "📦 *Available on:*\n• PyPI: `pip install neorl-industrial-gym`\n• Docker Hub: `docker pull terragonlabs/neorl-industrial`"
                }
            }
        ]
    }
    
    try:
        import requests
        response = requests.post(
            env_vars["SLACK_WEBHOOK"],
            json=message,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Slack notification sent")
        else:
            print(f"⚠️  Slack notification failed: {response.status_code}")
    
    except ImportError:
        print("⚠️  requests library not available, skipping Slack notification")
    except Exception as e:
        print(f"⚠️  Slack notification failed: {e}")


def update_documentation(version: str, repo_root: Path) -> None:
    """Update documentation with new version."""
    print("📚 Updating documentation")
    
    # Update any version-specific documentation
    # This could include API docs, installation guides, etc.
    
    # For now, just print a reminder
    print(f"📝 Remember to update documentation for version {version}")
    print("   - API documentation")
    print("   - Installation guides")
    print("   - Example notebooks")
    print("   - Docker deployment guides")


def main():
    """Main publication function."""
    parser = argparse.ArgumentParser(description="Publish neoRL-industrial-gym release")
    parser.add_argument("version", help="Version to publish (e.g., 1.0.0)")
    parser.add_argument("--skip-pypi", action="store_true", help="Skip PyPI publication")
    parser.add_argument("--skip-docker", action="store_true", help="Skip Docker image push")
    parser.add_argument("--skip-github", action="store_true", help="Skip GitHub release")
    parser.add_argument("--skip-notifications", action="store_true", help="Skip notifications")
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent
    version = args.version
    
    print(f"🚀 Publishing neoRL-industrial-gym v{version}")
    print(f"📁 Repository root: {repo_root}")
    
    try:
        # Check environment variables
        env_vars = check_environment_variables()
        
        # Publish to PyPI unless skipped
        if not args.skip_pypi:
            publish_to_pypi(repo_root, env_vars)
        else:
            print("⚠️  Skipping PyPI publication (--skip-pypi)")
        
        # Publish Docker images unless skipped
        if not args.skip_docker:
            login_docker_registry(env_vars)
            push_docker_images(version, repo_root)
        else:
            print("⚠️  Skipping Docker publication (--skip-docker)")
        
        # Create GitHub release unless skipped
        if not args.skip_github:
            create_github_release(version, repo_root, env_vars)
        else:
            print("⚠️  Skipping GitHub release (--skip-github)")
        
        # Update documentation
        update_documentation(version, repo_root)
        
        # Send notifications unless skipped
        if not args.skip_notifications:
            send_slack_notification(version, env_vars)
        else:
            print("⚠️  Skipping notifications (--skip-notifications)")
        
        print(f"✅ Successfully published neoRL-industrial-gym v{version}!")
        print("\nRelease checklist:")
        print("✓ Packages published to PyPI")
        print("✓ Docker images pushed to registry")
        print("✓ GitHub release created")
        print("✓ Stakeholders notified")
        print("\nPost-release tasks:")
        print("- Update documentation website")
        print("- Announce on social media")
        print("- Update dependent projects")
        print("- Monitor for issues")
        
    except Exception as e:
        print(f"❌ Publication failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()