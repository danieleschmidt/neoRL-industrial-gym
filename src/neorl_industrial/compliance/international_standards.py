"""International standards compliance for global industrial deployment."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.types import Array


class InternationalStandard(Enum):
    """Supported international standards."""
    ISO_27001 = "iso_27001"  # Information Security Management
    ISO_9001 = "iso_9001"   # Quality Management
    IEC_61508 = "iec_61508"  # Functional Safety
    ISO_14001 = "iso_14001"  # Environmental Management
    GDPR = "gdpr"           # General Data Protection Regulation
    CCPA = "ccpa"           # California Consumer Privacy Act
    PDPA = "pdpa"           # Personal Data Protection Act (Singapore)
    SOC2 = "soc2"           # Service Organization Control 2
    NIST_CSF = "nist_csf"   # NIST Cybersecurity Framework
    IEC_62443 = "iec_62443"  # Industrial Communication Networks Security


class ComplianceLevel(Enum):
    """Compliance implementation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    ENTERPRISE = "enterprise"


@dataclass
class ComplianceRequirement:
    """Single compliance requirement definition."""
    
    id: str
    standard: InternationalStandard
    title: str
    description: str
    mandatory: bool
    implementation_level: ComplianceLevel
    verification_method: str
    documentation_required: bool = True


@dataclass
class ComplianceResult:
    """Result of compliance check."""
    
    requirement_id: str
    compliant: bool
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    recommendations: List[str]
    evidence: List[str]
    timestamp: datetime


class ComplianceChecker(ABC):
    """Abstract base class for compliance checkers."""
    
    def __init__(self, standard: InternationalStandard):
        self.standard = standard
        self.logger = logging.getLogger(f"ComplianceChecker.{standard.value}")
        
    @abstractmethod
    def get_requirements(self) -> List[ComplianceRequirement]:
        """Get all requirements for this standard."""
        pass
        
    @abstractmethod
    def check_compliance(self, system_data: Dict[str, Any]) -> List[ComplianceResult]:
        """Check system compliance against requirements."""
        pass


class ISO27001Checker(ComplianceChecker):
    """ISO 27001 Information Security Management compliance checker."""
    
    def __init__(self):
        super().__init__(InternationalStandard.ISO_27001)
        
    def get_requirements(self) -> List[ComplianceRequirement]:
        """Get ISO 27001 requirements."""
        return [
            ComplianceRequirement(
                id="A.9.1.1",
                standard=self.standard,
                title="Access Control Policy",
                description="Access control policy shall be established, documented and reviewed",
                mandatory=True,
                implementation_level=ComplianceLevel.BASIC,
                verification_method="documentation_review"
            ),
            ComplianceRequirement(
                id="A.9.2.1",
                standard=self.standard,
                title="User Registration and De-registration",
                description="Formal user registration and de-registration process",
                mandatory=True,
                implementation_level=ComplianceLevel.STANDARD,
                verification_method="process_audit"
            ),
            ComplianceRequirement(
                id="A.12.1.2",
                standard=self.standard,
                title="Change Management",
                description="Changes to systems shall be controlled",
                mandatory=True,
                implementation_level=ComplianceLevel.STANDARD,
                verification_method="change_log_review"
            ),
            ComplianceRequirement(
                id="A.12.6.1",
                standard=self.standard,
                title="Management of Technical Vulnerabilities",
                description="Information about technical vulnerabilities shall be obtained",
                mandatory=True,
                implementation_level=ComplianceLevel.ADVANCED,
                verification_method="vulnerability_assessment"
            ),
            ComplianceRequirement(
                id="A.18.1.4",
                standard=self.standard,
                title="Privacy and Protection of PII",
                description="Privacy and protection requirements for PII shall be identified",
                mandatory=True,
                implementation_level=ComplianceLevel.ENTERPRISE,
                verification_method="privacy_impact_assessment"
            )
        ]
        
    def check_compliance(self, system_data: Dict[str, Any]) -> List[ComplianceResult]:
        """Check ISO 27001 compliance."""
        results = []
        
        # A.9.1.1 - Access Control Policy
        access_policy = system_data.get("access_control_policy", {})
        access_compliance = self._check_access_control_policy(access_policy)
        results.append(access_compliance)
        
        # A.9.2.1 - User Registration
        user_management = system_data.get("user_management", {})
        user_compliance = self._check_user_registration(user_management)
        results.append(user_compliance)
        
        # A.12.1.2 - Change Management
        change_mgmt = system_data.get("change_management", {})
        change_compliance = self._check_change_management(change_mgmt)
        results.append(change_compliance)
        
        # A.12.6.1 - Vulnerability Management
        vuln_mgmt = system_data.get("vulnerability_management", {})
        vuln_compliance = self._check_vulnerability_management(vuln_mgmt)
        results.append(vuln_compliance)
        
        # A.18.1.4 - Privacy Protection
        privacy_controls = system_data.get("privacy_controls", {})
        privacy_compliance = self._check_privacy_protection(privacy_controls)
        results.append(privacy_compliance)
        
        return results
        
    def _check_access_control_policy(self, policy_data: Dict[str, Any]) -> ComplianceResult:
        """Check access control policy compliance."""
        score = 0.0
        details = {}
        recommendations = []
        evidence = []
        
        # Check if policy exists
        if policy_data.get("documented", False):
            score += 0.3
            evidence.append("Access control policy is documented")
        else:
            recommendations.append("Document access control policy")
            
        # Check if policy is current
        if policy_data.get("last_reviewed"):
            last_review = datetime.fromisoformat(policy_data["last_reviewed"])
            days_since_review = (datetime.now() - last_review).days
            if days_since_review <= 365:
                score += 0.3
                evidence.append(f"Policy reviewed {days_since_review} days ago")
            else:
                recommendations.append("Review access control policy (overdue)")
                
        # Check role-based access
        if policy_data.get("role_based_access", False):
            score += 0.2
            evidence.append("Role-based access control implemented")
        else:
            recommendations.append("Implement role-based access control")
            
        # Check principle of least privilege
        if policy_data.get("least_privilege", False):
            score += 0.2
            evidence.append("Principle of least privilege applied")
        else:
            recommendations.append("Apply principle of least privilege")
            
        details["policy_score"] = score
        details["components_checked"] = ["documentation", "review_date", "rbac", "least_privilege"]
        
        return ComplianceResult(
            requirement_id="A.9.1.1",
            compliant=score >= 0.8,
            score=score,
            details=details,
            recommendations=recommendations,
            evidence=evidence,
            timestamp=datetime.now()
        )
        
    def _check_user_registration(self, user_data: Dict[str, Any]) -> ComplianceResult:
        """Check user registration process compliance."""
        score = 0.0
        details = {}
        recommendations = []
        evidence = []
        
        # Check formal registration process
        if user_data.get("formal_process", False):
            score += 0.4
            evidence.append("Formal user registration process exists")
        else:
            recommendations.append("Establish formal user registration process")
            
        # Check approval workflow
        if user_data.get("approval_required", False):
            score += 0.3
            evidence.append("User registration requires approval")
        else:
            recommendations.append("Implement approval workflow for user registration")
            
        # Check de-registration process
        if user_data.get("deregistration_process", False):
            score += 0.3
            evidence.append("User de-registration process exists")
        else:
            recommendations.append("Establish user de-registration process")
            
        details["registration_score"] = score
        
        return ComplianceResult(
            requirement_id="A.9.2.1",
            compliant=score >= 0.8,
            score=score,
            details=details,
            recommendations=recommendations,
            evidence=evidence,
            timestamp=datetime.now()
        )
        
    def _check_change_management(self, change_data: Dict[str, Any]) -> ComplianceResult:
        """Check change management compliance."""
        score = 0.0
        details = {}
        recommendations = []
        evidence = []
        
        # Check change control process
        if change_data.get("change_control_process", False):
            score += 0.3
            evidence.append("Change control process documented")
        else:
            recommendations.append("Document change control process")
            
        # Check change approval
        if change_data.get("approval_required", False):
            score += 0.3
            evidence.append("Changes require approval")
        else:
            recommendations.append("Require approval for system changes")
            
        # Check change testing
        if change_data.get("testing_required", False):
            score += 0.2
            evidence.append("Changes are tested before implementation")
        else:
            recommendations.append("Test changes before implementation")
            
        # Check rollback procedures
        if change_data.get("rollback_procedures", False):
            score += 0.2
            evidence.append("Rollback procedures exist")
        else:
            recommendations.append("Establish rollback procedures")
            
        details["change_score"] = score
        
        return ComplianceResult(
            requirement_id="A.12.1.2",
            compliant=score >= 0.8,
            score=score,
            details=details,
            recommendations=recommendations,
            evidence=evidence,
            timestamp=datetime.now()
        )
        
    def _check_vulnerability_management(self, vuln_data: Dict[str, Any]) -> ComplianceResult:
        """Check vulnerability management compliance."""
        score = 0.0
        details = {}
        recommendations = []
        evidence = []
        
        # Check vulnerability scanning
        if vuln_data.get("regular_scanning", False):
            score += 0.4
            evidence.append("Regular vulnerability scanning performed")
        else:
            recommendations.append("Implement regular vulnerability scanning")
            
        # Check patch management
        if vuln_data.get("patch_management", False):
            score += 0.3
            evidence.append("Patch management process exists")
        else:
            recommendations.append("Establish patch management process")
            
        # Check vulnerability assessment
        if vuln_data.get("risk_assessment", False):
            score += 0.3
            evidence.append("Vulnerability risk assessment performed")
        else:
            recommendations.append("Perform vulnerability risk assessments")
            
        details["vulnerability_score"] = score
        
        return ComplianceResult(
            requirement_id="A.12.6.1",
            compliant=score >= 0.8,
            score=score,
            details=details,
            recommendations=recommendations,
            evidence=evidence,
            timestamp=datetime.now()
        )
        
    def _check_privacy_protection(self, privacy_data: Dict[str, Any]) -> ComplianceResult:
        """Check privacy protection compliance."""
        score = 0.0
        details = {}
        recommendations = []
        evidence = []
        
        # Check privacy policy
        if privacy_data.get("privacy_policy", False):
            score += 0.3
            evidence.append("Privacy policy exists")
        else:
            recommendations.append("Establish privacy policy")
            
        # Check data classification
        if privacy_data.get("data_classification", False):
            score += 0.3
            evidence.append("Data classification implemented")
        else:
            recommendations.append("Implement data classification")
            
        # Check consent management
        if privacy_data.get("consent_management", False):
            score += 0.2
            evidence.append("Consent management process exists")
        else:
            recommendations.append("Implement consent management")
            
        # Check data subject rights
        if privacy_data.get("data_subject_rights", False):
            score += 0.2
            evidence.append("Data subject rights procedures exist")
        else:
            recommendations.append("Establish data subject rights procedures")
            
        details["privacy_score"] = score
        
        return ComplianceResult(
            requirement_id="A.18.1.4",
            compliant=score >= 0.8,
            score=score,
            details=details,
            recommendations=recommendations,
            evidence=evidence,
            timestamp=datetime.now()
        )


class GDPRChecker(ComplianceChecker):
    """GDPR compliance checker for EU data protection."""
    
    def __init__(self):
        super().__init__(InternationalStandard.GDPR)
        
    def get_requirements(self) -> List[ComplianceRequirement]:
        """Get GDPR requirements."""
        return [
            ComplianceRequirement(
                id="Art.5",
                standard=self.standard,
                title="Principles of Processing",
                description="Personal data shall be processed lawfully, fairly and transparently",
                mandatory=True,
                implementation_level=ComplianceLevel.BASIC,
                verification_method="process_review"
            ),
            ComplianceRequirement(
                id="Art.6",
                standard=self.standard,
                title="Lawfulness of Processing",
                description="Processing shall be lawful only if one of the legal bases applies",
                mandatory=True,
                implementation_level=ComplianceLevel.STANDARD,
                verification_method="legal_basis_assessment"
            ),
            ComplianceRequirement(
                id="Art.25",
                standard=self.standard,
                title="Data Protection by Design and by Default",
                description="Data protection shall be integrated into processing activities",
                mandatory=True,
                implementation_level=ComplianceLevel.ADVANCED,
                verification_method="design_review"
            ),
            ComplianceRequirement(
                id="Art.32",
                standard=self.standard,
                title="Security of Processing",
                description="Appropriate technical and organizational measures shall be implemented",
                mandatory=True,
                implementation_level=ComplianceLevel.ENTERPRISE,
                verification_method="security_assessment"
            )
        ]
        
    def check_compliance(self, system_data: Dict[str, Any]) -> List[ComplianceResult]:
        """Check GDPR compliance."""
        results = []
        
        # Article 5 - Principles of Processing
        processing_data = system_data.get("data_processing", {})
        principles_compliance = self._check_processing_principles(processing_data)
        results.append(principles_compliance)
        
        # Article 6 - Lawfulness
        legal_basis_data = system_data.get("legal_basis", {})
        lawfulness_compliance = self._check_lawfulness(legal_basis_data)
        results.append(lawfulness_compliance)
        
        # Article 25 - Privacy by Design
        design_data = system_data.get("privacy_by_design", {})
        design_compliance = self._check_privacy_by_design(design_data)
        results.append(design_compliance)
        
        # Article 32 - Security
        security_data = system_data.get("security_measures", {})
        security_compliance = self._check_security_of_processing(security_data)
        results.append(security_compliance)
        
        return results
        
    def _check_processing_principles(self, processing_data: Dict[str, Any]) -> ComplianceResult:
        """Check data processing principles compliance."""
        score = 0.0
        details = {}
        recommendations = []
        evidence = []
        
        # Check lawfulness, fairness, transparency
        if processing_data.get("lawful_basis_documented", False):
            score += 0.2
            evidence.append("Lawful basis for processing documented")
        else:
            recommendations.append("Document lawful basis for processing")
            
        # Check purpose limitation
        if processing_data.get("purpose_specified", False):
            score += 0.2
            evidence.append("Processing purposes specified")
        else:
            recommendations.append("Specify purposes for data processing")
            
        # Check data minimization
        if processing_data.get("data_minimization", False):
            score += 0.2
            evidence.append("Data minimization principles applied")
        else:
            recommendations.append("Apply data minimization principles")
            
        # Check accuracy
        if processing_data.get("data_accuracy_measures", False):
            score += 0.2
            evidence.append("Data accuracy measures implemented")
        else:
            recommendations.append("Implement data accuracy measures")
            
        # Check storage limitation
        if processing_data.get("retention_policy", False):
            score += 0.2
            evidence.append("Data retention policy exists")
        else:
            recommendations.append("Establish data retention policy")
            
        details["principles_score"] = score
        
        return ComplianceResult(
            requirement_id="Art.5",
            compliant=score >= 0.8,
            score=score,
            details=details,
            recommendations=recommendations,
            evidence=evidence,
            timestamp=datetime.now()
        )
        
    def _check_lawfulness(self, legal_data: Dict[str, Any]) -> ComplianceResult:
        """Check lawfulness of processing."""
        score = 0.0
        details = {}
        recommendations = []
        evidence = []
        
        # Check legal basis identification
        legal_bases = legal_data.get("legal_bases", [])
        if legal_bases:
            score += 0.5
            evidence.append(f"Legal bases identified: {', '.join(legal_bases)}")
        else:
            recommendations.append("Identify legal basis for processing")
            
        # Check consent management (if applicable)
        if "consent" in legal_bases:
            if legal_data.get("consent_management", False):
                score += 0.3
                evidence.append("Consent management implemented")
            else:
                recommendations.append("Implement consent management")
                
        # Check legitimate interest assessment (if applicable)  
        if "legitimate_interest" in legal_bases:
            if legal_data.get("legitimate_interest_assessment", False):
                score += 0.2
                evidence.append("Legitimate interest assessment performed")
            else:
                recommendations.append("Perform legitimate interest assessment")
                
        details["lawfulness_score"] = score
        
        return ComplianceResult(
            requirement_id="Art.6",
            compliant=score >= 0.8,
            score=score,
            details=details,
            recommendations=recommendations,
            evidence=evidence,
            timestamp=datetime.now()
        )
        
    def _check_privacy_by_design(self, design_data: Dict[str, Any]) -> ComplianceResult:
        """Check privacy by design implementation."""
        score = 0.0
        details = {}
        recommendations = []
        evidence = []
        
        # Check DPIA conducted
        if design_data.get("dpia_conducted", False):
            score += 0.4
            evidence.append("Data Protection Impact Assessment conducted")
        else:
            recommendations.append("Conduct Data Protection Impact Assessment")
            
        # Check privacy controls
        if design_data.get("privacy_controls", False):
            score += 0.3
            evidence.append("Privacy controls integrated into design")
        else:
            recommendations.append("Integrate privacy controls into system design")
            
        # Check default settings
        if design_data.get("privacy_by_default", False):
            score += 0.3
            evidence.append("Privacy-friendly default settings")
        else:
            recommendations.append("Implement privacy-friendly default settings")
            
        details["design_score"] = score
        
        return ComplianceResult(
            requirement_id="Art.25",
            compliant=score >= 0.8,
            score=score,
            details=details,
            recommendations=recommendations,
            evidence=evidence,
            timestamp=datetime.now()
        )
        
    def _check_security_of_processing(self, security_data: Dict[str, Any]) -> ComplianceResult:
        """Check security of processing measures."""
        score = 0.0
        details = {}
        recommendations = []
        evidence = []
        
        # Check encryption
        if security_data.get("encryption_at_rest", False):
            score += 0.2
            evidence.append("Encryption at rest implemented")
        else:
            recommendations.append("Implement encryption at rest")
            
        if security_data.get("encryption_in_transit", False):
            score += 0.2
            evidence.append("Encryption in transit implemented")
        else:
            recommendations.append("Implement encryption in transit")
            
        # Check access controls
        if security_data.get("access_controls", False):
            score += 0.2
            evidence.append("Access controls implemented")
        else:
            recommendations.append("Implement access controls")
            
        # Check monitoring
        if security_data.get("security_monitoring", False):
            score += 0.2
            evidence.append("Security monitoring active")
        else:
            recommendations.append("Implement security monitoring")
            
        # Check incident response
        if security_data.get("incident_response", False):
            score += 0.2
            evidence.append("Incident response procedures exist")
        else:
            recommendations.append("Establish incident response procedures")
            
        details["security_score"] = score
        
        return ComplianceResult(
            requirement_id="Art.32",
            compliant=score >= 0.8,
            score=score,
            details=details,
            recommendations=recommendations,
            evidence=evidence,
            timestamp=datetime.now()
        )


class InternationalComplianceManager:
    """Manager for international standards compliance across multiple jurisdictions."""
    
    def __init__(self, required_standards: Optional[List[InternationalStandard]] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize checkers for required standards
        self.checkers = {}
        
        if required_standards is None:
            required_standards = [InternationalStandard.ISO_27001, InternationalStandard.GDPR]
            
        for standard in required_standards:
            if standard == InternationalStandard.ISO_27001:
                self.checkers[standard] = ISO27001Checker()
            elif standard == InternationalStandard.GDPR:
                self.checkers[standard] = GDPRChecker()
            # Add other standard checkers as needed
            
        self.logger.info(f"Compliance manager initialized for standards: {[s.value for s in required_standards]}")
        
    def assess_compliance(self, system_data: Dict[str, Any]) -> Dict[InternationalStandard, List[ComplianceResult]]:
        """Assess compliance across all configured standards."""
        
        results = {}
        
        for standard, checker in self.checkers.items():
            try:
                self.logger.info(f"Assessing compliance for {standard.value}")
                standard_results = checker.check_compliance(system_data)
                results[standard] = standard_results
                
                # Log compliance summary
                compliant_count = sum(1 for r in standard_results if r.compliant)
                total_count = len(standard_results)
                compliance_rate = compliant_count / total_count if total_count > 0 else 0
                
                self.logger.info(
                    f"{standard.value}: {compliant_count}/{total_count} requirements "
                    f"compliant ({compliance_rate:.1%})"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to assess compliance for {standard.value}: {e}")
                results[standard] = []
                
        return results
        
    def generate_compliance_report(
        self, 
        assessment_results: Dict[InternationalStandard, List[ComplianceResult]]
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        report = {
            "assessment_date": datetime.now().isoformat(),
            "standards_assessed": [s.value for s in assessment_results.keys()],
            "overall_summary": {},
            "detailed_results": {},
            "action_items": [],
            "certification_readiness": {}
        }
        
        # Calculate overall statistics
        all_results = []
        for results in assessment_results.values():
            all_results.extend(results)
            
        if all_results:
            total_compliant = sum(1 for r in all_results if r.compliant)
            overall_compliance_rate = total_compliant / len(all_results)
            average_score = sum(r.score for r in all_results) / len(all_results)
            
            report["overall_summary"] = {
                "total_requirements": len(all_results),
                "compliant_requirements": total_compliant,
                "compliance_rate": overall_compliance_rate,
                "average_score": average_score,
                "overall_status": "COMPLIANT" if overall_compliance_rate >= 0.8 else "NON_COMPLIANT"
            }
            
        # Detailed results by standard
        for standard, results in assessment_results.items():
            if results:
                compliant_count = sum(1 for r in results if r.compliant)
                compliance_rate = compliant_count / len(results)
                avg_score = sum(r.score for r in results) / len(results)
                
                report["detailed_results"][standard.value] = {
                    "requirements_checked": len(results),
                    "compliant_requirements": compliant_count,
                    "compliance_rate": compliance_rate,
                    "average_score": avg_score,
                    "status": "COMPLIANT" if compliance_rate >= 0.8 else "NON_COMPLIANT",
                    "results": [
                        {
                            "requirement_id": r.requirement_id,
                            "compliant": r.compliant,
                            "score": r.score,
                            "recommendations": r.recommendations
                        }
                        for r in results
                    ]
                }
                
                # Collect action items
                for result in results:
                    if not result.compliant:
                        for recommendation in result.recommendations:
                            report["action_items"].append({
                                "standard": standard.value,
                                "requirement": result.requirement_id,
                                "action": recommendation,
                                "priority": "HIGH" if result.score < 0.5 else "MEDIUM"
                            })
                            
                # Certification readiness
                report["certification_readiness"][standard.value] = {
                    "ready": compliance_rate >= 0.95,
                    "compliance_gap": max(0, 0.95 - compliance_rate),
                    "estimated_effort": self._estimate_certification_effort(compliance_rate)
                }
                
        return report
        
    def _estimate_certification_effort(self, compliance_rate: float) -> str:
        """Estimate effort required for certification."""
        if compliance_rate >= 0.95:
            return "LOW - Ready for certification"
        elif compliance_rate >= 0.85:
            return "MEDIUM - Minor gaps to address"
        elif compliance_rate >= 0.70:
            return "HIGH - Significant improvements needed"
        else:
            return "VERY HIGH - Major compliance work required"
            
    def get_supported_standards(self) -> List[InternationalStandard]:
        """Get list of currently supported standards."""
        return list(self.checkers.keys())
        
    def add_standard(self, standard: InternationalStandard) -> bool:
        """Add support for additional standard."""
        try:
            if standard == InternationalStandard.ISO_27001:
                self.checkers[standard] = ISO27001Checker()
            elif standard == InternationalStandard.GDPR:
                self.checkers[standard] = GDPRChecker()
            else:
                self.logger.warning(f"Standard {standard.value} not yet implemented")
                return False
                
            self.logger.info(f"Added support for {standard.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add standard {standard.value}: {e}")
            return False