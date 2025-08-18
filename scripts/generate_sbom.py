#!/usr/bin/env python3
"""Generate Software Bill of Materials (SBOM) for neoRL-industrial-gym.

This script creates comprehensive SBOMs in multiple formats (SPDX, CycloneDX)
for compliance with security and regulatory requirements.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import uuid

import pkg_resources


class SBOMGenerator:
    """Generate SBOM in various formats for industrial compliance."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.timestamp = datetime.now(timezone.utc).isoformat()
        
    def get_python_dependencies(self) -> List[Dict[str, Any]]:
        """Extract Python package dependencies."""
        dependencies = []
        
        try:
            # Get installed packages
            for package in pkg_resources.working_set:
                dependencies.append({
                    "name": package.project_name,
                    "version": package.version,
                    "type": "python-package",
                    "location": str(package.location) if package.location else "",
                    "requires": [str(req) for req in package.requires()],
                })
        except Exception as e:
            print(f"Warning: Could not extract all Python dependencies: {e}")
        
        return dependencies
    
    def get_system_dependencies(self) -> List[Dict[str, Any]]:
        """Extract system-level dependencies (from Dockerfile)."""
        dependencies = []
        dockerfile_path = self.project_root / "Dockerfile"
        
        if dockerfile_path.exists():
            try:
                with open(dockerfile_path, 'r') as f:
                    content = f.read()
                
                # Parse APT packages from Dockerfile
                for line in content.split('\n'):
                    if 'apt-get install' in line and '-y' in line:
                        # Extract package names (simple parsing)
                        packages = line.split('install')[1].split('\\')[0].strip()
                        for pkg in packages.split():
                            if pkg.startswith('-') or pkg in ['&&', '||']:
                                continue
                            dependencies.append({
                                "name": pkg,
                                "version": "system-managed",
                                "type": "system-package",
                                "location": "/usr/local",
                                "requires": [],
                            })
            except Exception as e:
                print(f"Warning: Could not parse Dockerfile dependencies: {e}")
        
        return dependencies
    
    def generate_spdx_sbom(self, output_path: Path) -> None:
        """Generate SPDX format SBOM."""
        python_deps = self.get_python_dependencies()
        system_deps = self.get_system_dependencies()
        all_deps = python_deps + system_deps
        
        spdx_document = {
            "SPDXID": "SPDXRef-DOCUMENT",
            "spdxVersion": "SPDX-2.3",
            "creationInfo": {
                "created": self.timestamp,
                "creators": [
                    "Tool: neoRL-industrial-gym-sbom-generator",
                    "Organization: Terragon Labs"
                ],
                "licenseListVersion": "3.19"
            },
            "name": "neoRL-industrial-gym-SBOM",
            "dataLicense": "CC0-1.0",
            "documentNamespace": f"https://terragon.ai/spdx/neorl-industrial-gym-{uuid.uuid4()}",
            "documentDescribes": ["SPDXRef-Package-neorl-industrial-gym"],
            "packages": []
        }
        
        # Main package
        main_package = {
            "SPDXID": "SPDXRef-Package-neorl-industrial-gym",
            "name": "neorl-industrial-gym",
            "downloadLocation": "https://github.com/terragon-labs/neoRL-industrial-gym",
            "filesAnalyzed": False,
            "licenseConcluded": "MIT",
            "licenseDeclared": "MIT",
            "copyrightText": "Copyright (c) 2025 Terragon Labs",
            "supplier": "Organization: Terragon Labs"
        }
        spdx_document["packages"].append(main_package)
        
        # Dependencies
        for i, dep in enumerate(all_deps):
            package_id = f"SPDXRef-Package-{dep['name']}-{i}"
            package = {
                "SPDXID": package_id,
                "name": dep["name"],
                "versionInfo": dep["version"],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
                "copyrightText": "NOASSERTION",
                "externalRefs": []
            }
            
            if dep["type"] == "python-package":
                package["externalRefs"].append({
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:pypi/{dep['name']}@{dep['version']}"
                })
            
            spdx_document["packages"].append(package)
        
        with open(output_path, 'w') as f:
            json.dump(spdx_document, f, indent=2, sort_keys=True)
        
        print(f"SPDX SBOM generated: {output_path}")
    
    def generate_cyclonedx_sbom(self, output_path: Path) -> None:
        """Generate CycloneDX format SBOM."""
        python_deps = self.get_python_dependencies()
        system_deps = self.get_system_dependencies()
        all_deps = python_deps + system_deps
        
        cyclonedx_document = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "serialNumber": f"urn:uuid:{uuid.uuid4()}",
            "version": 1,
            "metadata": {
                "timestamp": self.timestamp,
                "tools": [
                    {
                        "vendor": "Terragon Labs",
                        "name": "neoRL-industrial-gym-sbom-generator",
                        "version": "1.0.0"
                    }
                ],
                "component": {
                    "type": "application",
                    "bom-ref": "neorl-industrial-gym",
                    "name": "neorl-industrial-gym",
                    "version": "0.1.0",
                    "description": "Industrial-grade Offline RL benchmark & library",
                    "licenses": [{"license": {"id": "MIT"}}],
                    "purl": "pkg:generic/neorl-industrial-gym@0.1.0"
                }
            },
            "components": []
        }
        
        for dep in all_deps:
            component = {
                "type": "library",
                "bom-ref": f"{dep['name']}@{dep['version']}",
                "name": dep["name"],
                "version": dep["version"],
                "scope": "required"
            }
            
            if dep["type"] == "python-package":
                component["purl"] = f"pkg:pypi/{dep['name']}@{dep['version']}"
            elif dep["type"] == "system-package":
                component["purl"] = f"pkg:deb/ubuntu/{dep['name']}@{dep['version']}"
            
            cyclonedx_document["components"].append(component)
        
        with open(output_path, 'w') as f:
            json.dump(cyclonedx_document, f, indent=2, sort_keys=True)
        
        print(f"CycloneDX SBOM generated: {output_path}")
    
    def generate_container_sbom(self, image_name: str, output_path: Path) -> None:
        """Generate SBOM for container image using syft."""
        try:
            cmd = [
                "syft", "packages", image_name,
                "-o", f"spdx-json={output_path}"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Container SBOM generated: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating container SBOM: {e}")
            print(f"stderr: {e.stderr}")
            # Fallback to manual generation
            self.generate_spdx_sbom(output_path)
        except FileNotFoundError:
            print("Warning: syft not found, falling back to manual SBOM generation")
            self.generate_spdx_sbom(output_path)
    
    def generate_vulnerability_report(self, output_path: Path) -> None:
        """Generate vulnerability report for dependencies."""
        try:
            # Use safety to check for known vulnerabilities
            cmd = ["safety", "check", "--json", "--output", str(output_path)]
            subprocess.run(cmd, check=True)
            print(f"Vulnerability report generated: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not generate vulnerability report: {e}")
            # Generate empty report
            with open(output_path, 'w') as f:
                json.dump({
                    "report_meta": {
                        "timestamp": self.timestamp,
                        "tool": "manual",
                        "status": "no_scanner_available"
                    },
                    "vulnerabilities": []
                }, f, indent=2)
        except FileNotFoundError:
            print("Warning: safety tool not found")
    
    def generate_license_report(self, output_path: Path) -> None:
        """Generate license report for all dependencies."""
        python_deps = self.get_python_dependencies()
        
        license_report = {
            "report_meta": {
                "timestamp": self.timestamp,
                "generator": "neoRL-industrial-gym-sbom-generator"
            },
            "project": {
                "name": "neorl-industrial-gym",
                "license": "MIT"
            },
            "dependencies": []
        }
        
        for dep in python_deps:
            try:
                # Try to get license info from package metadata
                dist = pkg_resources.get_distribution(dep["name"])
                license_info = "Unknown"
                
                if hasattr(dist, 'get_metadata'):
                    metadata = dist.get_metadata('METADATA')
                    for line in metadata.split('\n'):
                        if line.startswith('License:'):
                            license_info = line.split(':', 1)[1].strip()
                            break
                
                license_report["dependencies"].append({
                    "name": dep["name"],
                    "version": dep["version"],
                    "license": license_info
                })
            except Exception:
                license_report["dependencies"].append({
                    "name": dep["name"],
                    "version": dep["version"],
                    "license": "Unknown"
                })
        
        with open(output_path, 'w') as f:
            json.dump(license_report, f, indent=2, sort_keys=True)
        
        print(f"License report generated: {output_path}")


def main():
    """Main entry point for SBOM generation."""
    parser = argparse.ArgumentParser(
        description="Generate Software Bill of Materials for neoRL-industrial-gym"
    )
    parser.add_argument(
        "--format", 
        choices=["spdx", "cyclonedx", "both"],
        default="both",
        help="SBOM format to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sbom"),
        help="Output directory for SBOM files"
    )
    parser.add_argument(
        "--container-image",
        help="Generate SBOM for container image"
    )
    parser.add_argument(
        "--include-vulnerabilities",
        action="store_true",
        help="Include vulnerability report"
    )
    parser.add_argument(
        "--include-licenses",
        action="store_true", 
        help="Include license report"
    )
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = SBOMGenerator(project_root)
    
    # Generate SBOMs
    if args.format in ["spdx", "both"]:
        spdx_path = args.output_dir / "neorl-industrial-gym.spdx.json"
        generator.generate_spdx_sbom(spdx_path)
    
    if args.format in ["cyclonedx", "both"]:
        cyclonedx_path = args.output_dir / "neorl-industrial-gym.cyclonedx.json"
        generator.generate_cyclonedx_sbom(cyclonedx_path)
    
    # Generate container SBOM if requested
    if args.container_image:
        container_sbom_path = args.output_dir / f"container-{args.container_image.replace(':', '-')}.spdx.json"
        generator.generate_container_sbom(args.container_image, container_sbom_path)
    
    # Generate additional reports if requested
    if args.include_vulnerabilities:
        vuln_path = args.output_dir / "vulnerabilities.json"
        generator.generate_vulnerability_report(vuln_path)
    
    if args.include_licenses:
        license_path = args.output_dir / "licenses.json"
        generator.generate_license_report(license_path)
    
    print(f"\nSBOM generation complete. Files saved to: {args.output_dir}")
    print("\nGenerated files:")
    for file_path in args.output_dir.glob("*"):
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()