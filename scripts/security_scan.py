#!/usr/bin/env python3
"""
Security scanning script for neoRL-industrial-gym.

This script performs comprehensive security scanning including:
- Dependency vulnerability scanning with Safety
- Source code security analysis with Bandit
- License compliance checking
- SBOM generation for supply chain transparency
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


def check_safety() -> Dict:
    """Run Safety dependency vulnerability scan."""
    print("🔍 Running Safety dependency scan...")
    
    exit_code, stdout, stderr = run_command([
        "safety", "check", "--json", "--full-report"
    ])
    
    if exit_code == 0:
        print("✅ No known vulnerabilities found")
        return {"status": "clean", "vulnerabilities": []}
    
    try:
        safety_data = json.loads(stdout) if stdout else []
        print(f"⚠️  Found {len(safety_data)} potential vulnerabilities")
        return {"status": "vulnerabilities_found", "vulnerabilities": safety_data}
    except json.JSONDecodeError:
        print(f"❌ Safety scan failed: {stderr}")
        return {"status": "error", "error": stderr}


def check_bandit() -> Dict:
    """Run Bandit security analysis on source code."""
    print("🔍 Running Bandit security analysis...")
    
    exit_code, stdout, stderr = run_command([
        "bandit", "-r", "src/", "-f", "json", "--severity-level", "medium"
    ])
    
    try:
        if stdout:
            bandit_data = json.loads(stdout)
            issues = bandit_data.get("results", [])
            print(f"🔍 Bandit found {len(issues)} security issues")
            return {
                "status": "completed",
                "issues": issues,
                "summary": bandit_data.get("metrics", {})
            }
        else:
            print("✅ No security issues found by Bandit")
            return {"status": "clean", "issues": []}
    except json.JSONDecodeError:
        print(f"❌ Bandit scan failed: {stderr}")
        return {"status": "error", "error": stderr}


def check_licenses() -> Dict:
    """Check license compliance of dependencies."""
    print("🔍 Checking license compliance...")
    
    # First, try to install pip-licenses if not available
    subprocess.run([sys.executable, "-m", "pip", "install", "pip-licenses"], 
                  capture_output=True)
    
    exit_code, stdout, stderr = run_command([
        "pip-licenses", "--format=json", "--with-license-file", "--no-license-path"
    ])
    
    if exit_code != 0:
        print(f"❌ License check failed: {stderr}")
        return {"status": "error", "error": stderr}
    
    try:
        licenses = json.loads(stdout)
        
        # Check for potentially problematic licenses
        problematic_licenses = [
            "GPL", "AGPL", "LGPL", "Copyleft", "Commercial"
        ]
        
        flagged = []
        for pkg in licenses:
            license_name = pkg.get("License", "Unknown")
            if any(prob in license_name.upper() for prob in problematic_licenses):
                flagged.append(pkg)
        
        if flagged:
            print(f"⚠️  Found {len(flagged)} packages with potentially problematic licenses")
        else:
            print("✅ All licenses appear compatible")
        
        return {
            "status": "completed",
            "total_packages": len(licenses),
            "flagged_packages": flagged,
            "all_licenses": licenses
        }
    except json.JSONDecodeError:
        print(f"❌ License parsing failed: {stderr}")
        return {"status": "error", "error": stderr}


def generate_sbom() -> Dict:
    """Generate Software Bill of Materials (SBOM)."""
    print("🔍 Generating SBOM...")
    
    # Try to install cyclonedx-bom if not available
    subprocess.run([sys.executable, "-m", "pip", "install", "cyclonedx-bom"], 
                  capture_output=True)
    
    # Generate JSON SBOM
    exit_code, stdout, stderr = run_command([
        "cyclonedx-py", "--output-format", "json", "--output-file", "sbom.json"
    ])
    
    if exit_code == 0:
        print("✅ SBOM generated successfully")
        return {"status": "success", "files": ["sbom.json"]}
    else:
        print(f"❌ SBOM generation failed: {stderr}")
        return {"status": "error", "error": stderr}


def main():
    """Main security scanning function."""
    parser = argparse.ArgumentParser(description="Run security scans")
    parser.add_argument(
        "--output", 
        help="Output file for consolidated report", 
        default="security-report.json"
    )
    parser.add_argument(
        "--fail-on-vuln", 
        action="store_true",
        help="Exit with error code if vulnerabilities found"
    )
    parser.add_argument(
        "--skip-sbom", 
        action="store_true",
        help="Skip SBOM generation"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting comprehensive security scan...\n")
    
    # Run all security checks
    results = {
        "timestamp": subprocess.check_output(["date", "-u"]).decode().strip(),
        "safety": check_safety(),
        "bandit": check_bandit(),
        "licenses": check_licenses(),
    }
    
    if not args.skip_sbom:
        results["sbom"] = generate_sbom()
    
    # Save consolidated report
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Security report saved to {args.output}")
    
    # Determine exit code
    has_vulnerabilities = (
        results["safety"]["status"] == "vulnerabilities_found" or
        (results["bandit"]["status"] == "completed" and 
         len(results["bandit"]["issues"]) > 0)
    )
    
    if has_vulnerabilities:
        print("\n⚠️  Security issues detected!")
        if args.fail_on_vuln:
            print("🚨 Exiting with error due to --fail-on-vuln flag")
            sys.exit(1)
    else:
        print("\n✅ No critical security issues detected")
    
    print("\n🔒 Security scan completed")


if __name__ == "__main__":
    main()