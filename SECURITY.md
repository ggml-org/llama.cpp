# Security Policy

 - [**Using llama.cpp securely**](#using-llamacpp-securely)
   - [Untrusted models](#untrusted-models)
   - [Untrusted inputs](#untrusted-inputs)
   - [Data privacy](#data-privacy)
   - [Untrusted environments or networks](#untrusted-environments-or-networks)
   - [Multi-Tenant environments](#multi-tenant-environments)
 - [**Reporting a vulnerability**](#reporting-a-vulnerability)

## Using llama.cpp securely

### Untrusted models
Be careful when running untrusted models. This classification includes models created by unknown developers or utilizing data obtained from unknown sources.

*Always execute untrusted models within a secure, isolated environment such as a sandbox* (e.g., containers, virtual machines). This helps protect your system from potentially malicious code.

> [!NOTE]
> The trustworthiness of a model is not binary. You must always determine the proper level of caution depending on the specific model and how it matches your use case and risk tolerance.

### Untrusted inputs

Some models accept various input formats (text, images, audio, etc.). The libraries converting these inputs have varying security levels, so it's crucial to isolate the model and carefully pre-process inputs to mitigate script injection risks.

For maximum security when handling untrusted inputs, you may need to employ the following:

* Sandboxing: Isolate the environment where the inference happens.
* Pre-analysis: Check how the model performs by default when exposed to prompt injection (e.g. using [fuzzing for prompt injection](https://github.com/FonduAI/awesome-prompt-injection?tab=readme-ov-file#tools)). This will give you leads on how hard you will have to work on the next topics.
* Updates: Keep both LLaMA C++ and your libraries updated with the latest security patches.
* Input Sanitation: Before feeding data to the model, sanitize inputs rigorously. This involves techniques such as:
    * Validation: Enforce strict rules on allowed characters and data types.
    * Filtering: Remove potentially malicious scripts or code fragments.
    * Encoding: Convert special characters into safe representations.
    * Verification: Run tooling that identifies potential script injections (e.g. [models that detect prompt injection attempts](https://python.langchain.com/docs/guides/safety/hugging_face_prompt_injection)).

### Data privacy

To protect sensitive data from potential leaks or unauthorized access, it is crucial to sandbox the model execution. This means running the model in a secure, isolated environment, which helps mitigate many attack vectors.

### Untrusted environments or networks

If you can't run your models in a secure and isolated environment or if it must be exposed to an untrusted network, make sure to take the following security precautions:
* Confirm the hash of any downloaded artifact (e.g. pre-trained model weights) matches a known-good value
* Encrypt your data if sending it over the network.

### Multi-Tenant environments

If you intend to run multiple models in parallel with shared memory, it is your responsibility to ensure the models do not interact or access each other's data. The primary areas of concern are tenant isolation, resource allocation, model sharing and hardware attacks.

1. Tenant Isolation: Models should run separately with strong isolation methods to prevent unwanted data access. Separating networks is crucial for isolation, as it prevents unauthorized access to data or models and malicious users from sending graphs to execute under another tenant's identity.

2. Resource Allocation: A denial of service caused by one model can impact the overall system health. Implement safeguards like rate limits, access controls, and health monitoring.

3. Model Sharing: In a multitenant model sharing design, tenants and users must understand the security risks of running code provided by others. Since there are no reliable methods to detect malicious models, sandboxing the model execution is the recommended approach to mitigate the risk.

4. Hardware Attacks: GPUs or TPUs can also be attacked. [Researches](https://scholar.google.com/scholar?q=gpu+side+channel) has shown that side channel attacks on GPUs are possible, which can make data leak from other models or processes running on the same system at the same time.

## Reporting a vulnerability

Beware that none of the topics under [Using llama.cpp securely](#using-llamacpp-securely) are considered vulnerabilities of LLaMA C++.

<!-- normal version -->
However, If you have discovered a security vulnerability in this project, please report it privately. **Do not disclose it as a public issue.** This gives us time to work with you to fix the issue before public exposure, reducing the chance that the exploit will be used before a patch is released.

Please disclose it as a private [security advisory](https://github.com/ggml-org/llama.cpp/security/advisories/new).

A team of volunteers on a reasonable-effort basis maintains this project. As such, please give us at least 90 days to work on a fix before public exposure.
