#!/usr/bin/env python3
"""Prepare release script for neoRL-industrial-gym.

This script prepares the repository for a new release by:
1. Updating version numbers
2. Building distribution packages
3. Creating Docker images
4. Running pre-release validation
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def run_command(cmd: List[str], cwd: Path = None) -> subprocess.CompletedProcess:
    """Run a command and handle errors."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ {' '.join(cmd)}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"✗ {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        sys.exit(1)


def update_version_files(version: str, repo_root: Path) -> None:
    """Update version in relevant files."""
    print(f"📝 Updating version to {version}")
    
    # Update pyproject.toml version if using dynamic versioning
    pyproject_path = repo_root / "pyproject.toml"
    if pyproject_path.exists():
        print(f"✓ Version managed by setuptools_scm in {pyproject_path}")
    
    # Version is automatically handled by setuptools_scm
    # but we could add manual version file updates here if needed


def run_tests(repo_root: Path) -> None:
    """Run test suite before release."""
    print("🧪 Running test suite")
    
    # Run tests with coverage
    run_command([
        "python", "-m", "pytest",
        "tests/",
        "--cov=neorl_industrial",
        "--cov-report=term-missing",
        "--cov-fail-under=80"
    ], cwd=repo_root)
    
    # Run safety tests specifically
    run_command([
        "python", "-m", "pytest",
        "-m", "safety",
        "-v"
    ], cwd=repo_root)


def run_linting(repo_root: Path) -> None:
    """Run code quality checks."""
    print("🔍 Running code quality checks")
    
    # Run type checking
    run_command(["python", "-m", "mypy", "src/neorl_industrial"], cwd=repo_root)
    
    # Run linting
    run_command(["python", "-m", "ruff", "check", "src/", "tests/"], cwd=repo_root)
    
    # Check formatting
    run_command(["python", "-m", "black", "--check", "src/", "tests/"], cwd=repo_root)


def build_packages(repo_root: Path) -> None:
    """Build distribution packages."""
    print("📦 Building distribution packages")
    
    # Clean previous builds
    dist_dir = repo_root / "dist"
    if dist_dir.exists():
        run_command(["rm", "-rf", "dist"], cwd=repo_root)
    
    # Build source distribution and wheel
    run_command(["python", "-m", "build"], cwd=repo_root)
    
    print("✓ Distribution packages built successfully")


def build_docker_images(version: str, repo_root: Path) -> None:
    """Build Docker images for the release."""
    print("🐳 Building Docker images")
    
    # Create docker-images directory
    docker_images_dir = repo_root / "docker-images"
    docker_images_dir.mkdir(exist_ok=True)
    
    images = [
        ("production", "neorl-industrial"),
        ("development", "neorl-industrial-dev"),
        ("gpu", "neorl-industrial-gpu")
    ]
    
    for target, image_name in images:
        print(f"Building {image_name}:{version} ({target})")
        
        # Build image
        run_command([
            "docker", "build",
            "--target", target,
            "-t", f"{image_name}:{version}",
            "-t", f"{image_name}:latest",
            "."
        ], cwd=repo_root)
        
        # Save image to file
        image_file = docker_images_dir / f"{image_name}-{version}.tar"
        run_command([
            "docker", "save",
            "-o", str(image_file),
            f"{image_name}:{version}"
        ], cwd=repo_root)
        
        print(f"✓ Saved {image_name}:{version} to {image_file}")


def validate_docker_images(version: str, repo_root: Path) -> None:
    """Validate Docker images work correctly."""
    print("🔬 Validating Docker images")
    
    # Test production image
    print("Testing production image...")
    run_command([
        "docker", "run", "--rm",
        f"neorl-industrial:{version}",
        "python", "-c", "import neorl_industrial; print('Production image OK')"
    ], cwd=repo_root)
    
    # Test development image
    print("Testing development image...")
    run_command([
        "docker", "run", "--rm",
        f"neorl-industrial-dev:{version}",
        "python", "-c", "import neorl_industrial; print('Development image OK')"
    ], cwd=repo_root)
    
    print("✓ Docker images validated successfully")


def run_security_scan(repo_root: Path) -> None:
    """Run security scans on the codebase."""
    print("🔒 Running security scans")
    
    # Run safety check on dependencies
    try:
        run_command(["python", "-m", "safety", "check"], cwd=repo_root)
    except FileNotFoundError:
        print("⚠️  Safety not installed, skipping dependency vulnerability scan")
    
    # Run bandit security scan
    try:
        run_command([
            "python", "-m", "bandit",
            "-r", "src/",
            "-f", "text",
            "-ll"  # Low severity and low confidence
        ], cwd=repo_root)
    except FileNotFoundError:
        print("⚠️  Bandit not installed, skipping security scan")


def validate_package_integrity(repo_root: Path) -> None:
    """Validate the built packages."""
    print("📋 Validating package integrity")
    
    dist_dir = repo_root / "dist"
    if not dist_dir.exists() or not list(dist_dir.glob("*.whl")):
        print("✗ No wheel files found in dist/")
        sys.exit(1)
    
    # Check wheel contents
    wheel_file = next(dist_dir.glob("*.whl"))
    run_command(["python", "-m", "wheel", "unpack", str(wheel_file), "/tmp/wheel_check"], cwd=repo_root)
    
    print("✓ Package integrity validated")


def main():
    """Main release preparation function."""
    parser = argparse.ArgumentParser(description="Prepare neoRL-industrial-gym release")
    parser.add_argument("version", help="Version to release (e.g., 1.0.0)")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-docker", action="store_true", help="Skip Docker image building")
    parser.add_argument("--skip-security", action="store_true", help="Skip security scans")
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent
    version = args.version
    
    print(f"🚀 Preparing release {version} for neoRL-industrial-gym")
    print(f"📁 Repository root: {repo_root}")
    
    try:
        # Update version files
        update_version_files(version, repo_root)
        
        # Run code quality checks
        run_linting(repo_root)
        
        # Run tests unless skipped
        if not args.skip_tests:
            run_tests(repo_root)
        else:
            print("⚠️  Skipping tests (--skip-tests)")
        
        # Run security scans unless skipped
        if not args.skip_security:
            run_security_scan(repo_root)
        else:
            print("⚠️  Skipping security scans (--skip-security)")
        
        # Build distribution packages
        build_packages(repo_root)
        
        # Validate packages
        validate_package_integrity(repo_root)
        
        # Build Docker images unless skipped
        if not args.skip_docker:
            build_docker_images(version, repo_root)
            validate_docker_images(version, repo_root)
        else:
            print("⚠️  Skipping Docker builds (--skip-docker)")
        
        print(f"✅ Release {version} prepared successfully!")
        print("\nNext steps:")
        print("1. Review the generated CHANGELOG.md")
        print("2. Verify all tests pass")
        print("3. Check Docker images work correctly")
        print("4. Commit and push changes")
        print("5. Create release tag")
        
    except Exception as e:
        print(f"❌ Release preparation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()