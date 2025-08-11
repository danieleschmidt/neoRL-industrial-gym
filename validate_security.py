#!/usr/bin/env python3
"""Comprehensive security validation for neoRL-industrial-gym."""

import os
import sys
import ast
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import hashlib

def check_file_security(filepath: Path) -> Dict[str, Any]:
    """Check a Python file for common security issues."""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for common security anti-patterns
        security_patterns = [
            (r'eval\s*\(', 'Dangerous use of eval()', 'HIGH'),
            (r'exec\s*\(', 'Dangerous use of exec()', 'HIGH'),
            (r'subprocess\.call.*shell\s*=\s*True', 'Shell injection risk', 'HIGH'),
            (r'os\.system\s*\(', 'Command injection risk', 'HIGH'),
            (r'pickle\.loads?\s*\(', 'Pickle deserialization risk', 'MEDIUM'),
            (r'yaml\.load\s*\([^,]*\)', 'Unsafe YAML loading', 'MEDIUM'),
            (r'input\s*\(.*\)', 'User input without validation', 'LOW'),
            (r'open\s*\([^,]*,.*["\']w', 'File write without validation', 'LOW'),
            (r'random\.seed\s*\(\d+\)', 'Hardcoded random seed', 'LOW'),
            (
                r'password\s*=\s*["\'][^"\']+["\']', 
                'Hardcoded password', 'HIGH'
            ),
            (
                r'api[_-]?key\s*=\s*["\'][^"\']+["\']', 
                'Hardcoded API key', 'HIGH'
            ),
            (
                r'secret\s*=\s*["\'][^"\']+["\']', 
                'Hardcoded secret', 'HIGH'
            ),
        ]
        
        for pattern, message, severity in security_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                issues.append({
                    'type': 'security_pattern',
                    'severity': severity,
                    'message': message,
                    'line': line_num,
                    'match': match.group(0),
                })
        
        # Check for SQL injection risks
        sql_patterns = [
            r'execute\s*\([^,]*%[^,]*\)',
            r'execute\s*\([^,]*\.format\s*\(',
            r'execute\s*\([^,]*\+',
        ]
        
        for pattern in sql_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                issues.append({
                    'type': 'sql_injection',
                    'severity': 'HIGH',
                    'message': 'Potential SQL injection vulnerability',
                    'line': line_num,
                    'match': match.group(0),
                })
        
        # Parse AST for more sophisticated checks
        try:
            tree = ast.parse(content)
            
            class SecurityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.issues = []
                
                def visit_Call(self, node):
                    # Check for dangerous function calls
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', '__import__']:
                            self.issues.append({
                                'type': 'dangerous_call',
                                'severity': 'HIGH',
                                'message': f'Dangerous function call: {node.func.id}',
                                'line': node.lineno,
                            })
                    
                    elif isinstance(node.func, ast.Attribute):
                        # Check for subprocess with shell=True
                        if (
                            hasattr(node.func, 'attr') and 
                            node.func.attr in ['call', 'run', 'Popen'] and
                            any(
                                isinstance(kw.value, ast.Constant) and 
                                kw.value.value is True and 
                                kw.arg == 'shell' 
                                for kw in node.keywords
                            )
                        ):
                            self.issues.append({
                                'type': 'shell_injection',
                                'severity': 'HIGH', 
                                'message': 'subprocess call with shell=True',
                                'line': node.lineno,
                            })
                    
                    self.generic_visit(node)
                
                def visit_Assign(self, node):
                    # Check for hardcoded secrets in assignments
                    if (
                        isinstance(node.value, ast.Constant) and 
                        isinstance(node.value.value, str)
                    ):
                        
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                var_name = target.id.lower()
                                if any(
                                    secret in var_name 
                                    for secret in ['password', 'secret', 'key', 'token']
                                ):
                                    self.issues.append({
                                        'type': 'hardcoded_secret',
                                        'severity': 'HIGH',
                                        'message': f'Potential hardcoded secret: {target.id}',
                                        'line': node.lineno,
                                    })
                    
                    self.generic_visit(node)
            
            visitor = SecurityVisitor()
            visitor.visit(tree)
            issues.extend(visitor.issues)
            
        except SyntaxError:
            # Skip files with syntax errors
            pass
    
    except Exception as e:
        issues.append({
            'type': 'analysis_error',
            'severity': 'LOW',
            'message': f'Failed to analyze file: {e}',
            'line': 0,
        })
    
    return {
        'file': str(filepath),
        'issues': issues,
        'issue_count': len(issues),
    }

def check_file_permissions() -> List[Dict[str, Any]]:
    """Check for insecure file permissions."""
    issues = []
    
    # Check Python files for world-writable permissions
    for py_file in Path('src').rglob('*.py'):
        try:
            stat = py_file.stat()
            mode = oct(stat.st_mode)
            
            # Check if world-writable (002 or 022)
            if stat.st_mode & 0o002:
                issues.append({
                    'type': 'file_permissions',
                    'severity': 'MEDIUM',
                    'file': str(py_file),
                    'message': f'World-writable file: {mode}',
                })
        except Exception:
            continue
    
    return issues

def check_configuration_security() -> List[Dict[str, Any]]:
    """Check configuration files for security issues."""
    issues = []
    
    config_files = [
        'config/**/*.yaml',
        'config/**/*.yml', 
        'config/**/*.json',
        '*.env*',
        'docker-compose.yml',
        'Dockerfile',
    ]
    
    for pattern in config_files:
        for config_file in Path('.').glob(pattern):
            if not config_file.is_file():
                continue
                
            try:
                content = config_file.read_text()
                
                # Check for secrets in config
                secret_patterns = [
                    r'password\s*[:=]\s*["\'][^"\']{3,}["\']',
                    r'secret\s*[:=]\s*["\'][^"\']{3,}["\']',
                    r'api[_-]?key\s*[:=]\s*["\'][^"\']{3,}["\']',
                    r'token\s*[:=]\s*["\'][^"\']{8,}["\']',
                ]
                
                for pattern in secret_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        issues.append({
                            'type': 'config_secret',
                            'severity': 'HIGH',
                            'file': str(config_file),
                            'message': 'Potential secret in configuration file',
                            'line': line_num,
                        })
                
                # Check for debug settings
                if 'debug\s*[:=]\s*true' in content.lower():
                    issues.append({
                        'type': 'debug_enabled',
                        'severity': 'MEDIUM',
                        'file': str(config_file),
                        'message': 'Debug mode enabled in configuration',
                    })
            
            except Exception:
                continue
    
    return issues

def check_dependency_security() -> Dict[str, Any]:
    """Check dependencies for known security issues."""
    
    # Read requirements files
    requirements_files = ['requirements.txt', 'requirements-dev.txt']
    dependencies = set()
    
    for req_file in requirements_files:
        req_path = Path(req_file)
        if req_path.exists():
            try:
                content = req_path.read_text()
                for line in content.splitlines():
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name
                        pkg_name = re.split(r'[>=<!=]', line)[0].strip()
                        if pkg_name:
                            dependencies.add(pkg_name)
            except Exception:
                continue
    
    # Known vulnerable packages (simplified list)
    known_vulnerabilities = {
        'pyyaml': 'Versions < 5.4 have deserialization vulnerability',
        'pillow': 'Various versions have image processing vulnerabilities',
        'requests': 'Versions < 2.20.0 have various vulnerabilities',
        'urllib3': 'Various versions have certificate verification issues',
        'jinja2': 'Versions < 2.11.3 have XSS vulnerabilities',
    }
    
    issues = []
    for dep in dependencies:
        if dep.lower() in known_vulnerabilities:
            issues.append({
                'type': 'vulnerable_dependency',
                'severity': 'MEDIUM',
                'package': dep,
                'message': known_vulnerabilities[dep.lower()],
            })
    
    return {
        'total_dependencies': len(dependencies),
        'dependencies': sorted(list(dependencies)),
        'vulnerable_count': len(issues),
        'vulnerabilities': issues,
    }

def generate_security_report() -> Dict[str, Any]:
    """Generate comprehensive security report."""
    
    print("🛡️  Starting comprehensive security validation...")
    
    # Initialize report
    report = {
        'timestamp': '2025-08-10T02:31:31Z',
        'scan_type': 'comprehensive_security_validation',
        'summary': {
            'total_files_scanned': 0,
            'total_issues': 0,
            'high_severity': 0,
            'medium_severity': 0,
            'low_severity': 0,
        },
        'file_analysis': [],
        'configuration_security': [],
        'file_permissions': [],
        'dependency_security': {},
    }
    
    # 1. Analyze Python source files
    print("🔍 Analyzing Python source files...")
    python_files = list(Path('src').rglob('*.py'))
    
    for py_file in python_files:
        file_report = check_file_security(py_file)
        report['file_analysis'].append(file_report)
        report['summary']['total_files_scanned'] += 1
        
        for issue in file_report['issues']:
            report['summary']['total_issues'] += 1
            severity = issue['severity'].lower()
            if severity == 'high':
                report['summary']['high_severity'] += 1
            elif severity == 'medium':
                report['summary']['medium_severity'] += 1
            else:
                report['summary']['low_severity'] += 1
    
    print(f"   Analyzed {len(python_files)} Python files")
    
    # 2. Check file permissions
    print("🔍 Checking file permissions...")
    perm_issues = check_file_permissions()
    report['file_permissions'] = perm_issues
    
    for issue in perm_issues:
        report['summary']['total_issues'] += 1
        report['summary']['medium_severity'] += 1
    
    # 3. Check configuration security
    print("🔍 Analyzing configuration files...")
    config_issues = check_configuration_security()
    report['configuration_security'] = config_issues
    
    for issue in config_issues:
        report['summary']['total_issues'] += 1
        severity = issue['severity'].lower()
        if severity == 'high':
            report['summary']['high_severity'] += 1
        elif severity == 'medium':
            report['summary']['medium_severity'] += 1
        else:
            report['summary']['low_severity'] += 1
    
    # 4. Check dependency security
    print("🔍 Checking dependency security...")
    dep_report = check_dependency_security()
    report['dependency_security'] = dep_report
    
    for vuln in dep_report.get('vulnerabilities', []):
        report['summary']['total_issues'] += 1
        report['summary']['medium_severity'] += 1
    
    # 5. Generate overall security score
    total_issues = report['summary']['total_issues']
    high_issues = report['summary']['high_severity']
    medium_issues = report['summary']['medium_severity']
    
    # Security score calculation (100 - weighted penalty)
    penalty = (
        (high_issues * 10) + 
        (medium_issues * 3) + 
        (report['summary']['low_severity'] * 1)
    )
    security_score = max(0, 100 - penalty)
    
    report['security_score'] = security_score
    report['risk_level'] = (
        'LOW' if security_score >= 90 else
        'MEDIUM' if security_score >= 70 else
        'HIGH' if security_score >= 50 else
        'CRITICAL'
    )
    
    return report

def main():
    """Main security validation function."""
    
    # Generate report
    report = generate_security_report()
    
    # Save detailed report
    with open('security_validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🛡️  SECURITY VALIDATION SUMMARY")
    print("=" * 60)
    
    summary = report['summary']
    print(f"📁 Files Scanned: {summary['total_files_scanned']}")
    print(f"🔍 Total Issues Found: {summary['total_issues']}")
    print(f"🚨 High Severity: {summary['high_severity']}")
    print(f"⚠️  Medium Severity: {summary['medium_severity']}")
    print(f"ℹ️  Low Severity: {summary['low_severity']}")
    
    print(f"\n📊 Security Score: {report['security_score']}/100")
    print(f"⚖️  Risk Level: {report['risk_level']}")
    
    # Show top issues
    if summary['high_severity'] > 0:
        print("\n🚨 HIGH SEVERITY ISSUES:")
        count = 0
        for file_report in report['file_analysis']:
            for issue in file_report['issues']:
                if issue['severity'] == 'HIGH' and count < 5:
                    print(
                        f"   📄 {file_report['file']}:{issue.get('line', '?')} - {issue['message']}"
                    )
                    count += 1
        
        for issue in report['configuration_security']:
            if issue['severity'] == 'HIGH' and count < 5:
                print(f"   ⚙️  {issue['file']} - {issue['message']}")
                count += 1
    
    # Dependencies summary
    dep_info = report['dependency_security']
    print(
        f"\n📦 Dependencies: {dep_info['total_dependencies']} total, "
        f"{dep_info['vulnerable_count']} potentially vulnerable"
    )
    
    print(f"\n📋 Detailed report saved to: security_validation_report.json")
    
    # Final verdict
    if report['risk_level'] in ['LOW', 'MEDIUM']:
        print("\n✅ SECURITY VALIDATION PASSED")
        print("System meets security requirements for production deployment.")
        return 0
    else:
        print("\n❌ SECURITY VALIDATION FAILED")
        print("Critical security issues must be resolved before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
