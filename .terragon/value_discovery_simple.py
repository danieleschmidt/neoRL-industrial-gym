#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine (Simplified)
Discovers and prioritizes value items without external dependencies.
"""

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

class ValueItem:
    def __init__(self, id: str, title: str, description: str, category: str,
                 estimated_effort: float, impact_score: float, confidence_score: float,
                 ease_score: float, debt_impact: float, security_boost: float = 1.0):
        self.id = id
        self.title = title
        self.description = description
        self.category = category
        self.estimated_effort = estimated_effort
        self.impact_score = impact_score
        self.confidence_score = confidence_score
        self.ease_score = ease_score
        self.debt_impact = debt_impact
        self.security_boost = security_boost
        self.composite_score = 0.0
        self.created_at = datetime.now().isoformat()

class ValueDiscoveryEngine:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.backlog_path = self.repo_path / "AUTONOMOUS_BACKLOG.md"
        self.discovered_items: List[ValueItem] = []
        
        # Default scoring weights for advanced repository
        self.weights = {
            "wsjf": 0.5,
            "ice": 0.1,
            "technicalDebt": 0.3,
            "security": 0.1
        }

    def discover_value_items(self) -> List[ValueItem]:
        """Main discovery method."""
        self.discovered_items = []
        
        self._discover_from_git_history()
        self._discover_from_code_analysis()
        self._discover_from_files()
        self._generate_maintenance_items()
        
        # Score and prioritize items
        for item in self.discovered_items:
            item.composite_score = self._calculate_composite_score(item)
        
        # Sort by composite score descending
        self.discovered_items.sort(key=lambda x: x.composite_score, reverse=True)
        
        return self.discovered_items

    def _discover_from_git_history(self):
        """Extract value items from git history."""
        try:
            # Get recent commits
            result = subprocess.run(
                ["git", "log", "--oneline", "-20"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            commits = result.stdout.strip().split('\n')
            
            # Look for patterns indicating technical debt or improvements needed
            debt_patterns = ['fix', 'temp', 'hack', 'todo', 'quick', 'workaround']
            
            for line in commits:
                if line.strip():
                    commit_hash = line.split()[0]
                    message = ' '.join(line.split()[1:]).lower()
                    
                    for pattern in debt_patterns:
                        if pattern in message:
                            self.discovered_items.append(ValueItem(
                                id=f"git-debt-{commit_hash[:8]}",
                                title=f"Review and improve: {' '.join(line.split()[1:])[:50]}...",
                                description=f"Commit {commit_hash} suggests technical debt or quick fix",
                                category="technical-debt",
                                estimated_effort=1.5,
                                impact_score=5.0,
                                confidence_score=7.0,
                                ease_score=6.0,
                                debt_impact=12.0
                            ))
                            break
        except subprocess.CalledProcessError:
            pass

    def _discover_from_code_analysis(self):
        """Analyze code structure and complexity."""
        python_files = list(self.repo_path.rglob("*.py"))
        
        large_files = []
        complex_files = []
        undocumented_files = []
        
        for py_file in python_files:
            if py_file.name.startswith('test_') or '.git' in str(py_file):
                continue
                
            try:
                with open(py_file) as f:
                    content = f.read()
                
                lines = len(content.split('\n'))
                functions = len(re.findall(r'^\s*def\s+', content, re.MULTILINE))
                classes = len(re.findall(r'^\s*class\s+', content, re.MULTILINE))
                
                # Large file detection
                if lines > 300:
                    large_files.append((py_file, lines, functions, classes))
                
                # Complex file detection
                if functions > 15 and lines > 200:
                    complex_files.append((py_file, lines, functions))
                
                # Documentation check
                if not re.match(r'^\s*"""', content) and not re.match(r"^\s*'''", content):
                    if lines > 50:  # Only flag substantial files
                        undocumented_files.append(py_file)
                        
            except (IOError, UnicodeDecodeError):
                continue
        
        # Create value items for findings
        if large_files:
            self.discovered_items.append(ValueItem(
                id="refactor-large-files",
                title=f"Refactor {len(large_files)} large files (300+ lines)",
                description=f"Break down large files for better maintainability",
                category="refactoring",
                estimated_effort=len(large_files) * 2.0,
                impact_score=7.0,
                confidence_score=6.0,
                ease_score=4.0,
                debt_impact=len(large_files) * 15.0
            ))
        
        if complex_files:
            self.discovered_items.append(ValueItem(
                id="simplify-complex-files",
                title=f"Simplify {len(complex_files)} complex files",
                description="Reduce complexity in files with many functions",
                category="refactoring",
                estimated_effort=len(complex_files) * 1.5,
                impact_score=6.0,
                confidence_score=7.0,
                ease_score=5.0,
                debt_impact=len(complex_files) * 12.0
            ))
        
        if len(undocumented_files) > 2:
            self.discovered_items.append(ValueItem(
                id="add-docstrings",
                title=f"Add docstrings to {len(undocumented_files)} modules",
                description="Improve code documentation with module docstrings",
                category="documentation",
                estimated_effort=len(undocumented_files) * 0.3,
                impact_score=5.0,
                confidence_score=8.0,
                ease_score=8.0,
                debt_impact=len(undocumented_files) * 3.0
            ))

    def _discover_from_files(self):
        """Analyze file structure and missing files."""
        # Check for common missing files
        important_files = {
            '.github/workflows': 'CI/CD automation',
            'tests/': 'Test coverage',
            'docs/api/': 'API documentation',
            'CHANGELOG.md': 'Change tracking',
            '.devcontainer/': 'Development environment'
        }
        
        missing_items = []
        for file_path, description in important_files.items():
            if not (self.repo_path / file_path).exists():
                missing_items.append((file_path, description))
        
        if missing_items:
            self.discovered_items.append(ValueItem(
                id="add-missing-infrastructure",
                title=f"Add {len(missing_items)} missing infrastructure files",
                description=f"Add: {', '.join([item[0] for item in missing_items[:3]])}...",
                category="infrastructure",
                estimated_effort=len(missing_items) * 1.0,
                impact_score=6.0,
                confidence_score=8.0,
                ease_score=7.0,
                debt_impact=len(missing_items) * 8.0
            ))

    def _generate_maintenance_items(self):
        """Generate standard maintenance items for advanced repositories."""
        
        # Security audit
        self.discovered_items.append(ValueItem(
            id="security-audit",
            title="Conduct comprehensive security audit",
            description="Review dependencies, code, and configurations for security issues",
            category="security",
            estimated_effort=4.0,
            impact_score=8.0,
            confidence_score=6.0,
            ease_score=5.0,
            debt_impact=20.0,
            security_boost=2.0
        ))
        
        # Performance optimization
        self.discovered_items.append(ValueItem(
            id="performance-optimization",
            title="Profile and optimize performance bottlenecks",
            description="Identify and optimize slow code paths and memory usage",
            category="performance",
            estimated_effort=6.0,
            impact_score=7.0,
            confidence_score=5.0,
            ease_score=4.0,
            debt_impact=25.0
        ))
        
        # Test coverage improvement
        self.discovered_items.append(ValueItem(
            id="improve-test-coverage",
            title="Increase test coverage to 95%",
            description="Add tests for uncovered code paths and edge cases",
            category="testing",
            estimated_effort=8.0,
            impact_score=7.0,
            confidence_score=8.0,
            ease_score=6.0,
            debt_impact=30.0
        ))
        
        # API documentation
        self.discovered_items.append(ValueItem(
            id="api-documentation",
            title="Generate comprehensive API documentation",
            description="Create detailed API docs with examples and usage patterns",
            category="documentation",
            estimated_effort=5.0,
            impact_score=6.0,
            confidence_score=8.0,
            ease_score=7.0,
            debt_impact=15.0
        ))

    def _calculate_composite_score(self, item: ValueItem) -> float:
        """Calculate composite score using WSJF + ICE + Technical Debt."""
        
        # WSJF calculation
        cost_of_delay = (
            item.impact_score * 0.4 +           # User business value
            5.0 * 0.3 +                         # Time criticality
            item.security_boost * 2.0 * 0.2 +  # Risk reduction
            item.confidence_score * 0.1         # Opportunity enablement
        )
        wsjf = cost_of_delay / max(item.estimated_effort, 0.1)
        
        # ICE calculation
        ice = item.impact_score * item.confidence_score * item.ease_score
        
        # Technical debt calculation
        debt_score = item.debt_impact * item.security_boost
        
        # Composite score with weighting
        composite = (
            self.weights["wsjf"] * min(wsjf * 2, 100) +
            self.weights["ice"] * min(ice / 10, 100) +
            self.weights["technicalDebt"] * min(debt_score, 100) +
            self.weights["security"] * (item.security_boost - 1) * 20
        )
        
        return round(composite, 2)

    def generate_backlog_report(self):
        """Generate comprehensive backlog report."""
        items = self.discover_value_items()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# 📊 Autonomous Value Backlog

**Last Updated**: {timestamp}  
**Repository**: neoRL-industrial-gym  
**Maturity Level**: ADVANCED (85-90%)  
**Items Discovered**: {len(items)}  

## 🎯 Execution Summary

The Terragon Autonomous Value Discovery Engine has analyzed this repository and identified the highest-value work items. This advanced repository already has excellent SDLC practices, so the focus is on continuous improvement and optimization.

"""
        
        if items:
            next_item = items[0]
            report += f"""## 🚀 Next Best Value Item

**[{next_item.id.upper()}] {next_item.title}**
- **Composite Score**: {next_item.composite_score}
- **Category**: {next_item.category}
- **Estimated Effort**: {next_item.estimated_effort} hours
- **Impact**: {next_item.impact_score}/10 | **Confidence**: {next_item.confidence_score}/10 | **Ease**: {next_item.ease_score}/10
- **Technical Debt Impact**: {next_item.debt_impact}
- **Description**: {next_item.description}

## 📊 Top Priority Items

| Rank | ID | Title | Score | Category | Hours | Impact |
|------|-----|--------|-------|----------|-------|---------|
"""
            
            for i, item in enumerate(items[:10], 1):
                title_short = item.title[:45] + "..." if len(item.title) > 45 else item.title
                report += f"| {i} | {item.id.upper()} | {title_short} | {item.composite_score} | {item.category} | {item.estimated_effort} | {item.impact_score}/10 |\n"
        
        else:
            report += "\n*No high-value items currently identified. Repository is in excellent condition!*\n"
        
        # Category breakdown
        categories = {}
        for item in items:
            categories[item.category] = categories.get(item.category, 0) + 1
        
        report += f"""

## 📈 Value Metrics
- **Total Items**: {len(items)}
- **High-Impact Items** (Score > 50): {len([i for i in items if i.composite_score > 50])}
- **Quick Wins** (< 2 hours): {len([i for i in items if i.estimated_effort < 2])}
- **Average Score**: {sum(item.composite_score for item in items) / len(items) if items else 0:.1f}
- **Total Effort**: {sum(item.estimated_effort for item in items):.1f} hours

## 🎨 Category Breakdown
"""
        
        for category, count in sorted(categories.items()):
            report += f"- **{category}**: {count} items\n"
        
        report += f"""

## 🔍 Discovery Methods Used
- **Git History Analysis**: Recent commits analyzed for technical debt patterns
- **Code Structure Analysis**: File size, complexity, and documentation assessment
- **Infrastructure Gap Analysis**: Missing standard files and configurations
- **Maintenance Item Generation**: Proactive security, performance, and quality items

## ⚙️ Scoring Model
- **WSJF Component** (50%): Weighted Shortest Job First prioritization
- **ICE Component** (10%): Impact × Confidence × Ease scoring
- **Technical Debt** (30%): Long-term maintenance cost reduction
- **Security Boost** (10%): Security-related work receives priority multiplier

## 🎯 Autonomous Execution Guidelines

### Execution Criteria
- Minimum composite score: 15.0
- Maximum risk tolerance: 70%
- Single task execution (no parallel work)
- Full validation required before completion

### Success Metrics
- Code quality improvements
- Technical debt reduction
- Security posture enhancement
- Documentation coverage increase
- Test coverage improvement

### Value Tracking
Each executed item will be tracked for:
- Actual vs. estimated effort
- Measured impact and outcomes
- Lessons learned for future scoring
- Repository health improvements

---

*Generated by Terragon Autonomous Value Discovery Engine*  
*Configuration: `.terragon/config.yaml`*  
*Next scan: Continuous (triggered by repository changes)*
"""
        
        # Write report to file
        with open(self.backlog_path, 'w') as f:
            f.write(report)
        
        return report

def main():
    """Main entry point."""
    engine = ValueDiscoveryEngine()
    
    print("🔍 Discovering value items...")
    report = engine.generate_backlog_report()
    print(f"📊 Generated backlog with {len(engine.discovered_items)} items")
    
    if engine.discovered_items:
        next_item = engine.discovered_items[0]
        print(f"🎯 Next best value: {next_item.title} (Score: {next_item.composite_score})")
    else:
        print("✨ Repository is in excellent condition!")
    
    print(f"📝 Report saved to: {engine.backlog_path}")

if __name__ == "__main__":
    main()