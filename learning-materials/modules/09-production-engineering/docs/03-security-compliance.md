# Security & Compliance for LLM Inference Systems

## Introduction

Security is paramount in production ML systems. LLM inference systems face unique security challenges including prompt injection, model extraction, data privacy, and supply chain attacks. This lesson covers security best practices and compliance requirements.

## Threat Model for LLM Systems

### Attack Vectors

1. **Prompt Injection**
   - Malicious prompts to bypass system prompts
   - Jailbreaking attempts
   - Data exfiltration through prompts

2. **Model Extraction**
   - Stealing model weights through API access
   - Model distillation attacks
   - Membership inference

3. **Denial of Service**
   - Resource exhaustion through long prompts
   - Computational DoS
   - Memory exhaustion

4. **Data Privacy**
   - PII leakage in outputs
   - Training data memorization
   - Context contamination

5. **Supply Chain**
   - Compromised model files
   - Malicious dependencies
   - Backdoored binaries

## Input Validation and Sanitization

### Prompt Validation

```python
# security/prompt_validator.py
import re
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    reason: str = ""
    sanitized_input: str = ""

class PromptValidator:
    def __init__(self, max_length: int = 4096, max_tokens: int = 2048):
        self.max_length = max_length
        self.max_tokens = max_tokens

        # Patterns to detect potential injection
        self.injection_patterns = [
            r'ignore\s+previous\s+instructions',
            r'disregard\s+all\s+prior',
            r'system\s*:\s*you\s+are\s+now',
            r'jailbreak',
            r'\<\s*system\s*\>',
            r'\[INST\].*\[\/INST\]',  # Instruction format manipulation
        ]

        # Sensitive data patterns
        self.pii_patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        }

    def validate(self, prompt: str) -> ValidationResult:
        """Validate and sanitize prompt"""
        # Check length
        if len(prompt) > self.max_length:
            return ValidationResult(
                False,
                f"Prompt exceeds maximum length of {self.max_length} characters"
            )

        # Check for empty input
        if not prompt.strip():
            return ValidationResult(False, "Empty prompt")

        # Check for injection attempts
        for pattern in self.injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return ValidationResult(
                    False,
                    f"Potential prompt injection detected: {pattern}"
                )

        # Check for PII
        pii_found = []
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, prompt):
                pii_found.append(pii_type)

        if pii_found:
            return ValidationResult(
                False,
                f"PII detected: {', '.join(pii_found)}"
            )

        # Sanitize
        sanitized = self._sanitize(prompt)

        return ValidationResult(True, "Valid", sanitized)

    def _sanitize(self, prompt: str) -> str:
        """Sanitize prompt"""
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', prompt)

        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()

        # Remove potentially dangerous HTML/XML tags
        sanitized = re.sub(r'<[^>]+>', '', sanitized)

        return sanitized

    def check_rate_limit(self, user_id: str, redis_client) -> bool:
        """Check rate limiting"""
        key = f"rate_limit:{user_id}"
        current = redis_client.get(key)

        if current and int(current) >= 100:  # 100 requests per hour
            return False

        redis_client.incr(key)
        redis_client.expire(key, 3600)  # 1 hour
        return True

# Usage example
validator = PromptValidator(max_length=4096)

# Validate prompt
result = validator.validate("What is the capital of France?")
if result.is_valid:
    # Process sanitized input
    response = llm(result.sanitized_input)
else:
    return {"error": result.reason}, 400
```

### Content Filtering

```python
# security/content_filter.py
from typing import List, Set

class ContentFilter:
    def __init__(self):
        # Load banned words/phrases
        self.banned_terms = self._load_banned_terms()

        # Categories to filter
        self.categories = {
            'violence': self._load_category('violence'),
            'hate_speech': self._load_category('hate_speech'),
            'nsfw': self._load_category('nsfw'),
            'illegal': self._load_category('illegal'),
        }

    def _load_banned_terms(self) -> Set[str]:
        """Load banned terms from file"""
        try:
            with open('config/banned_terms.txt') as f:
                return set(line.strip().lower() for line in f)
        except FileNotFoundError:
            return set()

    def _load_category(self, category: str) -> Set[str]:
        """Load category-specific terms"""
        try:
            with open(f'config/filters/{category}.txt') as f:
                return set(line.strip().lower() for line in f)
        except FileNotFoundError:
            return set()

    def filter_input(self, text: str) -> Tuple[bool, List[str]]:
        """Filter input text"""
        text_lower = text.lower()
        violations = []

        # Check banned terms
        for term in self.banned_terms:
            if term in text_lower:
                violations.append(f"banned_term:{term}")

        # Check categories
        for category, terms in self.categories.items():
            for term in terms:
                if term in text_lower:
                    violations.append(f"{category}:{term}")

        is_safe = len(violations) == 0
        return is_safe, violations

    def filter_output(self, text: str) -> str:
        """Filter output text, replacing violations"""
        filtered = text

        # Replace banned content with [FILTERED]
        for term in self.banned_terms:
            if term in filtered.lower():
                # Case-insensitive replacement
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                filtered = pattern.sub('[FILTERED]', filtered)

        return filtered

# Usage
filter = ContentFilter()

# Check input
is_safe, violations = filter.filter_input(user_prompt)
if not is_safe:
    return {"error": "Content policy violation", "details": violations}, 400

# Filter output
safe_output = filter.filter_output(model_output)
```

## Authentication and Authorization

### API Key Management

```python
# security/auth.py
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional
import jwt

class APIKeyManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def generate_api_key(self, user_id: str, expires_days: int = 365) -> str:
        """Generate API key"""
        # Create random key
        random_key = secrets.token_urlsafe(32)

        # Create JWT with metadata
        payload = {
            'user_id': user_id,
            'key_hash': hashlib.sha256(random_key.encode()).hexdigest(),
            'created_at': datetime.utcnow().isoformat(),
            'expires_at': (datetime.utcnow() + timedelta(days=expires_days)).isoformat()
        }

        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        return f"llama_{random_key}_{token}"

    def validate_api_key(self, api_key: str) -> Optional[dict]:
        """Validate API key"""
        try:
            # Parse key
            parts = api_key.split('_')
            if len(parts) != 3 or parts[0] != 'llama':
                return None

            random_key = parts[1]
            token = parts[2]

            # Verify JWT
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])

            # Check expiration
            expires_at = datetime.fromisoformat(payload['expires_at'])
            if datetime.utcnow() > expires_at:
                return None

            # Verify key hash
            key_hash = hashlib.sha256(random_key.encode()).hexdigest()
            if key_hash != payload['key_hash']:
                return None

            return payload

        except Exception:
            return None

# FastAPI middleware example
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI()
security = HTTPBearer()
key_manager = APIKeyManager(secret_key=os.getenv('SECRET_KEY'))

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key middleware"""
    api_key = credentials.credentials

    payload = key_manager.validate_api_key(api_key)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired API key")

    return payload

@app.post("/v1/completions", dependencies=[Security(verify_api_key)])
async def completions(request: CompletionRequest):
    # Process request
    pass
```

### Role-Based Access Control (RBAC)

```python
# security/rbac.py
from enum import Enum
from typing import List, Set

class Role(Enum):
    ADMIN = "admin"
    DEVELOPER = "developer"
    USER = "user"
    VIEWER = "viewer"

class Permission(Enum):
    # Model operations
    MODEL_LOAD = "model:load"
    MODEL_UNLOAD = "model:unload"
    MODEL_LIST = "model:list"

    # Inference operations
    INFERENCE_BASIC = "inference:basic"
    INFERENCE_ADVANCED = "inference:advanced"
    INFERENCE_BATCH = "inference:batch"

    # Admin operations
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"
    USER_MANAGE = "user:manage"

class RBACManager:
    def __init__(self):
        # Define role permissions
        self.role_permissions = {
            Role.ADMIN: set([p for p in Permission]),
            Role.DEVELOPER: {
                Permission.MODEL_LIST,
                Permission.INFERENCE_BASIC,
                Permission.INFERENCE_ADVANCED,
                Permission.INFERENCE_BATCH,
                Permission.SYSTEM_MONITOR,
            },
            Role.USER: {
                Permission.MODEL_LIST,
                Permission.INFERENCE_BASIC,
            },
            Role.VIEWER: {
                Permission.MODEL_LIST,
                Permission.SYSTEM_MONITOR,
            }
        }

    def has_permission(self, role: Role, permission: Permission) -> bool:
        """Check if role has permission"""
        return permission in self.role_permissions.get(role, set())

    def check_permission(self, user_role: str, required_permission: str):
        """Decorator to check permissions"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                role = Role(user_role)
                permission = Permission(required_permission)

                if not self.has_permission(role, permission):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")

                return await func(*args, **kwargs)
            return wrapper
        return decorator

# Usage
rbac = RBACManager()

@app.post("/v1/models/load")
async def load_model(model_name: str, user: dict = Depends(verify_api_key)):
    if not rbac.has_permission(Role(user['role']), Permission.MODEL_LOAD):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    # Load model
    pass
```

## Secrets Management

### Using Environment Variables (Basic)

```python
# config/secrets.py
import os
from typing import Optional

class SecretsManager:
    @staticmethod
    def get_secret(key: str, default: Optional[str] = None) -> str:
        """Get secret from environment"""
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"Secret {key} not found")
        return value

    @staticmethod
    def load_from_file(filepath: str):
        """Load secrets from file (for docker secrets)"""
        try:
            with open(filepath) as f:
                return f.read().strip()
        except FileNotFoundError:
            raise ValueError(f"Secret file {filepath} not found")

# Usage
API_KEY = SecretsManager.get_secret('LLAMA_API_KEY')
DB_PASSWORD = SecretsManager.load_from_file('/run/secrets/db_password')
```

### Using HashiCorp Vault

```python
# config/vault_secrets.py
import hvac
import os

class VaultSecretsManager:
    def __init__(self):
        self.client = hvac.Client(
            url=os.getenv('VAULT_ADDR'),
            token=os.getenv('VAULT_TOKEN')
        )

        if not self.client.is_authenticated():
            raise Exception("Failed to authenticate with Vault")

    def get_secret(self, path: str, key: str) -> str:
        """Get secret from Vault"""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            return response['data']['data'][key]
        except Exception as e:
            raise ValueError(f"Failed to get secret {path}/{key}: {e}")

    def get_database_credentials(self, role: str) -> dict:
        """Get dynamic database credentials"""
        response = self.client.secrets.database.generate_credentials(
            name=role
        )
        return response['data']

# Usage
vault = VaultSecretsManager()
api_key = vault.get_secret('llama/prod', 'api_key')
db_creds = vault.get_database_credentials('llama-app')
```

## Model Security

### Model Checksum Verification

```python
# security/model_verification.py
import hashlib
import json
from pathlib import Path

class ModelVerifier:
    def __init__(self, checksums_file: str = "models/checksums.json"):
        self.checksums_file = checksums_file
        self.checksums = self._load_checksums()

    def _load_checksums(self) -> dict:
        """Load known good checksums"""
        try:
            with open(self.checksums_file) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def calculate_checksum(self, filepath: str, algorithm: str = 'sha256') -> str:
        """Calculate file checksum"""
        hash_func = hashlib.new(algorithm)

        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def verify_model(self, model_path: str, expected_checksum: str = None) -> bool:
        """Verify model file integrity"""
        actual_checksum = self.calculate_checksum(model_path)

        if expected_checksum:
            return actual_checksum == expected_checksum

        # Check against known checksums
        model_name = Path(model_path).name
        if model_name in self.checksums:
            return actual_checksum == self.checksums[model_name]

        raise ValueError(f"No checksum found for {model_name}")

    def add_model_checksum(self, model_path: str):
        """Add model to known checksums"""
        model_name = Path(model_path).name
        checksum = self.calculate_checksum(model_path)

        self.checksums[model_name] = checksum

        with open(self.checksums_file, 'w') as f:
            json.dump(self.checksums, f, indent=2)

# Usage
verifier = ModelVerifier()

# Verify before loading
if not verifier.verify_model("models/llama-2-7b.gguf"):
    raise Exception("Model checksum verification failed!")

model = Llama(model_path="models/llama-2-7b.gguf")
```

### Signed Models

```python
# security/model_signing.py
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

class ModelSigner:
    def __init__(self, private_key_path: str = None, public_key_path: str = None):
        if private_key_path:
            with open(private_key_path, 'rb') as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )

        if public_key_path:
            with open(public_key_path, 'rb') as f:
                self.public_key = serialization.load_pem_public_key(f.read())

    def sign_model(self, model_path: str, signature_path: str):
        """Sign model file"""
        # Read model file
        with open(model_path, 'rb') as f:
            model_data = f.read()

        # Create signature
        signature = self.private_key.sign(
            model_data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        # Save signature
        with open(signature_path, 'wb') as f:
            f.write(signature)

    def verify_signature(self, model_path: str, signature_path: str) -> bool:
        """Verify model signature"""
        # Read model and signature
        with open(model_path, 'rb') as f:
            model_data = f.read()

        with open(signature_path, 'rb') as f:
            signature = f.read()

        # Verify
        try:
            self.public_key.verify(
                signature,
                model_data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False

# Usage
signer = ModelSigner(public_key_path="keys/model_public_key.pem")

if not signer.verify_signature("models/llama-2-7b.gguf", "models/llama-2-7b.gguf.sig"):
    raise Exception("Model signature verification failed!")
```

## Security Scanning

### Vulnerability Scanning with Trivy

```bash
#!/bin/bash
# scripts/security_scan.sh

echo "Running security scans..."

# Scan Docker image
echo "Scanning Docker image..."
trivy image --severity HIGH,CRITICAL llama-inference:latest

# Scan filesystem
echo "Scanning filesystem..."
trivy fs --severity HIGH,CRITICAL .

# Scan dependencies
echo "Scanning Python dependencies..."
trivy fs --severity HIGH,CRITICAL requirements.txt

# Check for secrets in code
echo "Scanning for secrets..."
trufflehog filesystem . --only-verified

# SAST scanning
echo "Running SAST scan..."
bandit -r . -f json -o security-report.json

echo "Security scan complete"
```

### Automated Security in CI/CD

```yaml
# .github/workflows/security.yml
name: Security Scan

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  security-scan:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
        severity: 'CRITICAL,HIGH'

    - name: Upload Trivy results to GitHub Security
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

    - name: Run Bandit SAST
      run: |
        pip install bandit
        bandit -r . -f json -o bandit-report.json
      continue-on-error: true

    - name: Check for secrets
      uses: trufflesecurity/trufflehog@main
      with:
        path: ./
        base: ${{ github.event.repository.default_branch }}
        head: HEAD

    - name: Dependency Check
      uses: dependency-check/Dependency-Check_Action@main
      with:
        project: 'llama-inference'
        path: '.'
        format: 'ALL'

    - name: Fail on high severity
      run: |
        if grep -q '"severity": "HIGH"' trivy-results.sarif; then
          echo "High severity vulnerabilities found"
          exit 1
        fi
```

## Rate Limiting

### Redis-based Rate Limiting

```python
# security/rate_limiter.py
import redis
import time
from functools import wraps

class RateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> bool:
        """Token bucket rate limiting"""
        now = time.time()
        bucket_key = f"rate_limit:{key}"

        # Get current bucket state
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(bucket_key, 0, now - window_seconds)
        pipe.zcard(bucket_key)
        pipe.zadd(bucket_key, {str(now): now})
        pipe.expire(bucket_key, window_seconds)

        _, count, _, _ = pipe.execute()

        return count < max_requests

    def rate_limit(self, max_requests: int = 100, window: int = 3600):
        """Decorator for rate limiting"""
        def decorator(func):
            @wraps(func)
            async def wrapper(request, *args, **kwargs):
                # Get user identifier
                user_id = request.headers.get('X-User-ID', request.client.host)
                key = f"user:{user_id}"

                if not self.check_rate_limit(key, max_requests, window):
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded",
                        headers={"Retry-After": str(window)}
                    )

                return await func(request, *args, **kwargs)
            return wrapper
        return decorator

# Usage
redis_client = redis.Redis(host='localhost', port=6379, db=0)
limiter = RateLimiter(redis_client)

@app.post("/v1/completions")
@limiter.rate_limit(max_requests=100, window=3600)
async def completions(request: Request):
    # Process request
    pass
```

## Compliance

### GDPR Compliance

```python
# compliance/gdpr.py
from datetime import datetime, timedelta
import json

class GDPRCompliance:
    def __init__(self, storage):
        self.storage = storage

    def log_data_processing(self, user_id: str, purpose: str, data_types: list):
        """Log data processing activity"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'purpose': purpose,
            'data_types': data_types,
            'legal_basis': 'consent',  # or 'legitimate_interest', etc.
        }

        self.storage.append_log('data_processing', log_entry)

    def delete_user_data(self, user_id: str):
        """Right to erasure (Right to be forgotten)"""
        # Delete all user data
        self.storage.delete_user_logs(user_id)
        self.storage.delete_user_prompts(user_id)
        self.storage.delete_user_responses(user_id)

        # Log deletion
        self.log_data_processing(
            user_id,
            'data_deletion',
            ['logs', 'prompts', 'responses']
        )

    def export_user_data(self, user_id: str) -> dict:
        """Right to data portability"""
        return {
            'user_id': user_id,
            'exported_at': datetime.utcnow().isoformat(),
            'logs': self.storage.get_user_logs(user_id),
            'prompts': self.storage.get_user_prompts(user_id),
            'responses': self.storage.get_user_responses(user_id),
        }

    def anonymize_logs(self, retention_days: int = 90):
        """Anonymize old logs"""
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        self.storage.anonymize_logs_before(cutoff)
```

### Audit Logging

```python
# compliance/audit_log.py
import logging
import json
from datetime import datetime

class AuditLogger:
    def __init__(self, log_file: str = 'audit.log'):
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def log_event(
        self,
        event_type: str,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        metadata: dict = None
    ):
        """Log audit event"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'result': result,
            'metadata': metadata or {}
        }

        self.logger.info(json.dumps(event))

    def log_api_request(self, request, response, user_id):
        """Log API request"""
        self.log_event(
            event_type='api_request',
            user_id=user_id,
            action=request.method,
            resource=request.url.path,
            result='success' if response.status_code < 400 else 'failure',
            metadata={
                'status_code': response.status_code,
                'duration_ms': response.headers.get('X-Response-Time'),
                'ip_address': request.client.host,
            }
        )

    def log_model_access(self, user_id, model_name, action):
        """Log model access"""
        self.log_event(
            event_type='model_access',
            user_id=user_id,
            action=action,
            resource=model_name,
            result='success'
        )

# Usage
audit = AuditLogger()

@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    user_id = request.headers.get('X-User-ID', 'anonymous')
    response = await call_next(request)
    audit.log_api_request(request, response, user_id)
    return response
```

## Best Practices Summary

1. **Defense in Depth**: Multiple layers of security
2. **Least Privilege**: Minimal permissions necessary
3. **Fail Secure**: Default to secure state on failure
4. **Audit Everything**: Comprehensive logging
5. **Regular Updates**: Keep dependencies current
6. **Security Testing**: Automated vulnerability scanning
7. **Incident Response Plan**: Prepared for breaches
8. **Compliance**: Meet regulatory requirements

## Further Reading

- [OWASP Top 10 for LLMs](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [CIS Benchmarks](https://www.cisecurity.org/cis-benchmarks/)

---

**Authors**: Agent 5 (Documentation Specialist)
**Last Updated**: 2025-11-18
**Estimated Reading Time**: 45 minutes
