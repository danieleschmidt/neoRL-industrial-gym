# Security Policy

## Industrial Safety Context

neoRL-industrial-gym is designed for industrial control applications where security vulnerabilities can have serious safety implications. We take security issues very seriously and follow responsible disclosure practices.

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          | Security Support Until |
| ------- | ------------------ | --------------------- |
| 0.1.x   | :white_check_mark: | TBD                   |
| < 0.1   | :x:                | N/A                   |

## Reporting a Vulnerability

**For critical security issues that could impact industrial safety, please email security@terragon.ai directly rather than filing a public issue.**

For non-critical security issues, you may file a security issue using our [vulnerability report template](https://github.com/terragon-labs/neoRL-industrial-gym/issues/new?template=security_vulnerability.yml).

### Severity Classification

We classify security vulnerabilities using the following scale:

#### Critical (CVSS 9.0-10.0)
- Remote code execution with system privileges
- Complete system compromise
- Industrial safety system bypass
- **Response time: 24 hours, Fix target: 7 days**

#### High (CVSS 7.0-8.9)
- Remote code execution with limited privileges
- Significant data exposure
- Safety constraint bypass
- **Response time: 48 hours, Fix target: 14 days**

#### Medium (CVSS 4.0-6.9)
- Local privilege escalation
- Significant denial of service
- Limited information disclosure
- **Response time: 5 days, Fix target: 30 days**

#### Low (CVSS 0.1-3.9)
- Minor information disclosure
- Low-impact denial of service
- **Response time: 10 days, Fix target: 90 days**

### What to include in your report:

1. **Description**: Clear description of the vulnerability
2. **Impact**: Potential safety and security implications
3. **Reproduction**: Steps to reproduce the issue
4. **Environment**: Version, platform, and configuration details
5. **Suggested fix**: If you have ideas for mitigation
6. **Proof of Concept**: Code demonstrating the issue (if safe to share)

### Our commitment:

- **Acknowledgment**: We will acknowledge receipt within 24-48 hours
- **Initial assessment**: We will provide an initial assessment within 5 business days
- **Regular updates**: We will provide updates every 10 business days until resolved
- **Fix timeline**: Based on severity classification above
- **Credit**: We will credit security researchers in our security advisory (if desired)
- **CVE coordination**: We will coordinate CVE assignment for significant vulnerabilities

## Security Best Practices

### Deployment Security

When deploying neoRL-industrial-gym in industrial environments:

1. **Network Segmentation**
   - Deploy in isolated industrial networks (DMZ)
   - Use firewalls to restrict access
   - Implement network monitoring

2. **Access Control**
   - Implement multi-factor authentication
   - Use role-based access control (RBAC)
   - Regular access reviews and deprovisioning

3. **Container Security**
   - Run containers as non-root users
   - Use security scanning on container images
   - Keep base images updated
   - Implement runtime security monitoring

4. **Data Security**
   - Encrypt data at rest and in transit
   - Sanitize industrial data before training
   - Implement secure backup procedures
   - Regular data retention policy review

5. **Monitoring and Logging**
   - Comprehensive audit logging
   - Security event monitoring
   - Anomaly detection
   - Incident response procedures

### Development Security

For developers contributing to neoRL-industrial-gym:

1. **Secure Coding Practices**
   - Input validation and sanitization
   - Output encoding
   - Secure error handling
   - Regular security code reviews

2. **Dependency Management**
   - Regular dependency updates
   - Vulnerability scanning
   - License compliance checks
   - Minimal dependency principle

3. **Testing Security**
   - Security-focused test cases
   - Penetration testing
   - Fuzzing for input validation
   - Safety constraint testing

## Industrial Security Standards Compliance

We align with relevant industrial security standards including:

- **IEC 62443** (Industrial communication networks – IT security)
  - Zone and conduit model implementation
  - Security levels 1-4 support
  - Risk assessment methodology

- **NIST Cybersecurity Framework**
  - Identify, Protect, Detect, Respond, Recover
  - Continuous monitoring
  - Risk management

- **ISO 27001** (Information security management)
  - Security management system
  - Risk assessment and treatment
  - Incident management

- **NERC CIP** (North American Electric Reliability Corporation Critical Infrastructure Protection)
  - Critical asset identification
  - Security controls implementation
  - Personnel and training requirements

## Security Features

### Built-in Security Controls

- **Input Validation**: Comprehensive validation of all external inputs
- **Safety Constraints**: Enforced safety boundaries with emergency shutdown
- **Audit Logging**: Complete audit trail of all system interactions
- **Secure Containers**: Security-hardened Docker configurations
- **Dependency Scanning**: Automated vulnerability scanning of dependencies
- **SBOM Generation**: Software Bill of Materials for supply chain security

### Safety-Security Integration

- **Defense in Depth**: Multiple layers of security controls
- **Fail-Safe Design**: System fails to safe state on security violations
- **Real-time Monitoring**: Continuous monitoring of safety and security metrics
- **Emergency Response**: Automated response to security incidents affecting safety

## Vulnerability Disclosure Timeline

1. **Day 0**: Vulnerability reported
2. **Day 1-2**: Acknowledgment and initial triage
3. **Day 3-7**: Detailed analysis and impact assessment
4. **Day 8-14**: Develop and test fix (for critical/high severity)
5. **Day 15-21**: Security advisory preparation
6. **Day 22-30**: Coordinated disclosure and patch release
7. **Day 30+**: Public disclosure with full details

## Contact Information

- **Security Team**: security@terragon.ai
- **Emergency Contact**: +1-555-SECURITY (for critical industrial safety issues)
- **GPG Key**: [Download public key](https://terragon.ai/security/pgp-key.asc)

## Security Hall of Fame

We recognize security researchers who help improve the security of neoRL-industrial-gym:

*No submissions yet - be the first to help us improve industrial AI security!*

## Industrial Safety Warning

⚠️ **CRITICAL**: This software is intended for research and simulation only. Always validate policies extensively before deploying to real industrial systems. The maintainers are not responsible for any damage or safety incidents resulting from deployment to real systems.

Before deploying trained policies to real industrial systems:

1. Conduct extensive safety validation
2. Implement proper monitoring and fail-safes
3. Follow your organization's safety protocols
4. Consider hiring industrial safety experts
5. Validate against OSHA and relevant regulations
6. Perform security assessments
7. Implement proper access controls and monitoring

---

**Last Updated**: August 18, 2025  
**Next Review**: November 18, 2025