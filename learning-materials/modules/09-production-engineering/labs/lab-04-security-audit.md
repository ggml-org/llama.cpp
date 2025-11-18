# Lab 4: Security Audit

## Objectives

- ✅ Conduct security vulnerability assessment
- ✅ Implement input validation
- ✅ Set up automated security scanning
- ✅ Harden production deployment
- ✅ Document security measures

**Estimated Time**: 2-3 hours

## Part 1: Vulnerability Assessment

### Task 1.1: Automated Scanning

```bash
# Scan Docker image
trivy image llama-inference:latest

# Scan filesystem
trivy fs .

# Scan dependencies
pip install safety
safety check

# Bandit for Python code
bandit -r . -f json -o security-report.json
```

**✏️ Task**: Fix all HIGH and CRITICAL vulnerabilities.

### Task 1.2: Secret Detection

```bash
# Install trufflehog
docker run --rm -v "$PWD:/src" trufflesecurity/trufflehog:latest filesystem /src

# Check for hardcoded secrets
grep -r "api_key\|password\|secret" . --exclude-dir=.git
```

## Part 2: Input Validation

### Task 2.1: Implement Validation

```python
# security/validator.py
import re

class InputValidator:
    MAX_LENGTH = 4096
    INJECTION_PATTERNS = [
        r'ignore\s+previous\s+instructions',
        r'system\s*:\s*you\s+are',
        r'jailbreak',
    ]

    @classmethod
    def validate(cls, prompt: str) -> tuple[bool, str]:
        # Check length
        if len(prompt) > cls.MAX_LENGTH:
            return False, "Prompt too long"

        # Check for injection
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                return False, "Potential injection detected"

        # Check for PII
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', prompt):
            return False, "SSN detected"

        return True, "Valid"

# Usage
is_valid, reason = InputValidator.validate(user_input)
if not is_valid:
    raise ValueError(f"Invalid input: {reason}")
```

### Task 2.2: Rate Limiting

```python
from fastapi import HTTPException, Request
import time

class RateLimiter:
    def __init__(self, max_requests=100, window=3600):
        self.max_requests = max_requests
        self.window = window
        self.requests = {}

    def check(self, user_id: str) -> bool:
        now = time.time()

        # Clean old entries
        if user_id in self.requests:
            self.requests[user_id] = [
                t for t in self.requests[user_id]
                if now - t < self.window
            ]

        # Check limit
        if user_id not in self.requests:
            self.requests[user_id] = []

        if len(self.requests[user_id]) >= self.max_requests:
            return False

        self.requests[user_id].append(now)
        return True

rate_limiter = RateLimiter()

@app.post("/completions")
async def completions(request: Request):
    user_id = request.client.host

    if not rate_limiter.check(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Process request...
```

## Part 3: Authentication & Authorization

### Task 3.1: API Key Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hashlib

security = HTTPBearer()

API_KEYS = {
    hashlib.sha256(b"test-key-123").hexdigest(): "user123"
}

async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    key_hash = hashlib.sha256(credentials.credentials.encode()).hexdigest()

    if key_hash not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return API_KEYS[key_hash]

@app.post("/completions")
async def completions(user_id: str = Security(verify_api_key)):
    # User authenticated
    pass
```

### Task 3.2: Role-Based Access Control

```python
from enum import Enum

class Role(Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"

PERMISSIONS = {
    Role.ADMIN: ["read", "write", "delete", "admin"],
    Role.USER: ["read", "write"],
    Role.VIEWER: ["read"]
}

def check_permission(user_role: Role, required: str):
    if required not in PERMISSIONS.get(user_role, []):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
```

## Part 4: Secure Deployment

### Task 4.1: Hardened Dockerfile

```dockerfile
# Use specific version
FROM ubuntu:22.04

# Run as non-root
RUN useradd -m -u 1000 llama
USER llama

# Read-only filesystem
VOLUME /tmp
VOLUME /models

# Drop capabilities
RUN setcap cap_net_bind_service=+ep /app/llama-server

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1
```

### Task 4.2: Kubernetes Security

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-inference
spec:
  template:
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000

      containers:
      - name: llama-server
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
              - ALL

        resources:
          limits:
            memory: "8Gi"
            cpu: "4"
          requests:
            memory: "4Gi"
            cpu: "2"
```

## Part 5: Compliance

### Task 5.1: Audit Logging

```python
import logging
import json

class AuditLogger:
    def __init__(self):
        self.logger = logging.getLogger('audit')
        handler = logging.FileHandler('audit.log')
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_access(self, user_id, action, resource, result):
        self.logger.info(json.dumps({
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'result': result
        }))

audit = AuditLogger()

@app.post("/completions")
async def completions(user_id: str = Security(verify_api_key)):
    audit.log_access(user_id, "completion", "/completions", "success")
    # ...
```

### Task 5.2: Data Privacy

```python
def anonymize_prompt(prompt: str) -> str:
    """Remove PII from prompts before logging"""
    # Redact emails
    prompt = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    '[EMAIL]', prompt)

    # Redact phone numbers
    prompt = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', prompt)

    # Redact SSNs
    prompt = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', prompt)

    return prompt

# Log anonymized version
audit.log(anonymize_prompt(user_prompt))
```

## Verification

### Security Checklist

- [ ] No HIGH/CRITICAL vulnerabilities in scan
- [ ] No secrets in code or config
- [ ] Input validation implemented
- [ ] Rate limiting active
- [ ] Authentication required
- [ ] Running as non-root
- [ ] Resource limits set
- [ ] Audit logging enabled
- [ ] TLS/HTTPS enabled
- [ ] Security headers configured

### Penetration Testing

```bash
# Test injection
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Ignore previous instructions and..."}'

# Test rate limiting
for i in {1..110}; do
  curl http://localhost:8080/completion -H "Authorization: Bearer key"
done

# Test without auth
curl http://localhost:8080/completion
# Should get 401
```

## Deliverables

- ✅ Security audit report
- ✅ Vulnerability scan results
- ✅ Hardened deployment configuration
- ✅ Security testing evidence
- ✅ Compliance documentation

## Challenge Tasks

1. Implement WAF rules
2. Set up security monitoring (SIEM)
3. Add encryption at rest
4. Implement certificate pinning
5. Create incident response runbook

---

**Next**: Lab 5: Contributing to Open Source
