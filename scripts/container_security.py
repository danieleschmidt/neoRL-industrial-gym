#!/usr/bin/env python3
"""
Container security scanning script for neoRL-industrial-gym.

This script performs comprehensive container security analysis including:
- Dockerfile linting with Hadolint
- Image vulnerability scanning with Trivy
- Container configuration security checks
- Runtime security recommendations
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def run_command(cmd: List[str], capture_output: bool = True) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            check=False
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def check_docker_available() -> bool:
    """Check if Docker is available."""
    exit_code, _, _ = run_command(["docker", "--version"])
    return exit_code == 0


def lint_dockerfile(dockerfile_path: str = "Dockerfile") -> Dict:
    """Lint Dockerfile with Hadolint."""
    print(f"🔍 Linting {dockerfile_path} with Hadolint...")
    
    if not Path(dockerfile_path).exists():
        return {"status": "error", "error": f"{dockerfile_path} not found"}
    
    # Try to run hadolint directly, fall back to Docker if not available
    exit_code, stdout, stderr = run_command([
        "hadolint", "--format", "json", dockerfile_path
    ])
    
    if exit_code == 127:  # Command not found
        print("📦 Hadolint not found locally, using Docker image...")
        exit_code, stdout, stderr = run_command([
            "docker", "run", "--rm", "-i", "hadolint/hadolint", "hadolint", 
            "--format", "json", "-"
        ], capture_output=True)
        
        if exit_code != 0:
            # Read dockerfile and pipe to docker command
            with open(dockerfile_path, 'r') as f:
                dockerfile_content = f.read()
            
            process = subprocess.run([
                "docker", "run", "--rm", "-i", "hadolint/hadolint", 
                "hadolint", "--format", "json", "-"
            ], input=dockerfile_content, text=True, capture_output=True)
            exit_code, stdout, stderr = process.returncode, process.stdout, process.stderr
    
    try:
        if stdout:
            issues = json.loads(stdout) if stdout.strip() else []
            print(f"🔍 Found {len(issues)} Dockerfile issues")
            return {"status": "completed", "issues": issues}
        else:
            print("✅ No Dockerfile issues found")
            return {"status": "clean", "issues": []}
    except json.JSONDecodeError:
        if exit_code == 0:
            print("✅ No Dockerfile issues found")
            return {"status": "clean", "issues": []}
        else:
            print(f"❌ Hadolint failed: {stderr}")
            return {"status": "error", "error": stderr}


def build_image(image_name: str = "neorl-industrial:security-scan") -> Dict:
    """Build Docker image for scanning."""
    print(f"🏗️  Building image {image_name}...")
    
    exit_code, stdout, stderr = run_command([
        "docker", "build", "-t", image_name, "."
    ])
    
    if exit_code == 0:
        print("✅ Image built successfully")
        return {"status": "success", "image": image_name}
    else:
        print(f"❌ Image build failed: {stderr}")
        return {"status": "error", "error": stderr}


def scan_with_trivy(image_name: str) -> Dict:
    """Scan image with Trivy vulnerability scanner."""
    print(f"🔍 Scanning {image_name} with Trivy...")
    
    # Try to use trivy directly, fall back to Docker if not available
    exit_code, stdout, stderr = run_command([
        "trivy", "image", "--format", "json", "--quiet", image_name
    ])
    
    if exit_code == 127:  # Command not found
        print("📦 Trivy not found locally, using Docker image...")
        exit_code, stdout, stderr = run_command([
            "docker", "run", "--rm", "-v", "/var/run/docker.sock:/var/run/docker.sock",
            "aquasec/trivy:latest", "image", "--format", "json", "--quiet", image_name
        ])
    
    try:
        if stdout:
            scan_results = json.loads(stdout)
            
            # Count vulnerabilities by severity
            vuln_count = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
            
            for result in scan_results.get("Results", []):
                for vuln in result.get("Vulnerabilities", []):
                    severity = vuln.get("Severity", "UNKNOWN")
                    if severity in vuln_count:
                        vuln_count[severity] += 1
            
            total_vulns = sum(vuln_count.values())
            print(f"🔍 Found {total_vulns} vulnerabilities: {vuln_count}")
            
            return {
                "status": "completed",
                "vulnerability_count": vuln_count,
                "total_vulnerabilities": total_vulns,
                "scan_results": scan_results
            }
        else:
            print("✅ No vulnerabilities found")
            return {"status": "clean", "vulnerability_count": {}}
    except json.JSONDecodeError:
        print(f"❌ Trivy scan failed: {stderr}")
        return {"status": "error", "error": stderr}


def analyze_docker_compose(compose_file: str = "docker-compose.yml") -> Dict:
    """Analyze Docker Compose configuration for security issues."""
    print(f"🔍 Analyzing {compose_file} for security issues...")
    
    if not Path(compose_file).exists():
        return {"status": "skipped", "reason": f"{compose_file} not found"}
    
    try:
        import yaml
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml"], 
                      capture_output=True)
        import yaml
    
    try:
        with open(compose_file, 'r') as f:
            compose_data = yaml.safe_load(f)
        
        security_issues = []
        
        services = compose_data.get("services", {})
        for service_name, service_config in services.items():
            # Check for security-related configurations
            
            # Check for privileged mode
            if service_config.get("privileged", False):
                security_issues.append({
                    "service": service_name,
                    "issue": "privileged_mode",
                    "severity": "HIGH",
                    "description": "Service running in privileged mode"
                })
            
            # Check for host network mode
            if service_config.get("network_mode") == "host":
                security_issues.append({
                    "service": service_name,
                    "issue": "host_network",
                    "severity": "MEDIUM",
                    "description": "Service using host network mode"
                })
            
            # Check for bind mounts of sensitive directories
            volumes = service_config.get("volumes", [])
            for volume in volumes:
                if isinstance(volume, str):
                    if volume.startswith("/var/run/docker.sock"):
                        security_issues.append({
                            "service": service_name,
                            "issue": "docker_socket_mount",
                            "severity": "HIGH",
                            "description": "Docker socket mounted (allows container escape)"
                        })
                    elif volume.startswith("/"):
                        host_path = volume.split(":")[0]
                        if host_path in ["/", "/etc", "/usr", "/var"]:
                            security_issues.append({
                                "service": service_name,
                                "issue": "sensitive_bind_mount",
                                "severity": "MEDIUM",
                                "description": f"Sensitive directory {host_path} mounted"
                            })
            
            # Check for missing user specification
            if "user" not in service_config:
                security_issues.append({
                    "service": service_name,
                    "issue": "no_user_specified",
                    "severity": "LOW",
                    "description": "Container may run as root"
                })
        
        print(f"🔍 Found {len(security_issues)} Docker Compose security issues")
        return {
            "status": "completed",
            "issues": security_issues,
            "total_issues": len(security_issues)
        }
        
    except Exception as e:
        print(f"❌ Docker Compose analysis failed: {str(e)}")
        return {"status": "error", "error": str(e)}


def generate_security_recommendations() -> Dict:
    """Generate container security recommendations."""
    recommendations = [
        {
            "category": "Image Security",
            "recommendations": [
                "Use official base images or trusted registries",
                "Keep base images updated regularly",
                "Use specific image tags instead of 'latest'",
                "Minimize the number of layers and installed packages",
                "Use multi-stage builds to reduce attack surface"
            ]
        },
        {
            "category": "Runtime Security",
            "recommendations": [
                "Run containers as non-root user",
                "Use security profiles (AppArmor, SELinux)",
                "Limit container capabilities",
                "Use read-only root filesystem where possible",
                "Implement proper secrets management"
            ]
        },
        {
            "category": "Network Security",
            "recommendations": [
                "Use custom networks instead of default bridge",
                "Implement network segmentation",
                "Limit exposed ports to minimum required",
                "Use TLS for inter-service communication"
            ]
        },
        {
            "category": "Industrial Safety",
            "recommendations": [
                "Implement emergency shutdown mechanisms",
                "Monitor container resource usage",
                "Set up proper logging and monitoring",
                "Test failure scenarios regularly",
                "Implement circuit breakers for external services"
            ]
        }
    ]
    
    return {"status": "generated", "recommendations": recommendations}


def main():
    """Main container security scanning function."""
    parser = argparse.ArgumentParser(description="Run container security scans")
    parser.add_argument(
        "--output", 
        help="Output file for consolidated report", 
        default="container-security-report.json"
    )
    parser.add_argument(
        "--image-name", 
        help="Docker image name for scanning", 
        default="neorl-industrial:security-scan"
    )
    parser.add_argument(
        "--skip-build", 
        action="store_true",
        help="Skip building image (use existing)"
    )
    parser.add_argument(
        "--dockerfile", 
        help="Path to Dockerfile", 
        default="Dockerfile"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting container security analysis...\n")
    
    # Check if Docker is available
    if not check_docker_available():
        print("❌ Docker is not available. Please install Docker to run container security scans.")
        sys.exit(1)
    
    # Run security analysis
    results = {
        "timestamp": subprocess.check_output(["date", "-u"]).decode().strip(),
        "dockerfile_lint": lint_dockerfile(args.dockerfile),
        "docker_compose_analysis": analyze_docker_compose(),
        "security_recommendations": generate_security_recommendations()
    }
    
    # Build image if requested
    if not args.skip_build:
        build_result = build_image(args.image_name)
        results["image_build"] = build_result
        
        # Only scan if build was successful
        if build_result["status"] == "success":
            results["vulnerability_scan"] = scan_with_trivy(args.image_name)
    else:
        print(f"⏭️  Skipping build, scanning existing image {args.image_name}")
        results["vulnerability_scan"] = scan_with_trivy(args.image_name)
    
    # Save consolidated report
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Container security report saved to {args.output}")
    
    # Summary
    dockerfile_issues = len(results.get("dockerfile_lint", {}).get("issues", []))
    compose_issues = results.get("docker_compose_analysis", {}).get("total_issues", 0)
    
    vuln_scan = results.get("vulnerability_scan", {})
    if vuln_scan.get("status") == "completed":
        critical_vulns = vuln_scan.get("vulnerability_count", {}).get("CRITICAL", 0)
        high_vulns = vuln_scan.get("vulnerability_count", {}).get("HIGH", 0)
        total_vulns = vuln_scan.get("total_vulnerabilities", 0)
        print(f"\n📊 Security Summary:")
        print(f"   • Dockerfile issues: {dockerfile_issues}")
        print(f"   • Docker Compose issues: {compose_issues}")
        print(f"   • Container vulnerabilities: {total_vulns} (Critical: {critical_vulns}, High: {high_vulns})")
    
    print("\n🔒 Container security analysis completed")


if __name__ == "__main__":
    main()