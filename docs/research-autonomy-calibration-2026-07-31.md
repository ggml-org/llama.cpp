# Research Notes: Agent Autonomy Calibration (Industry Practice + Academic SOTA) -> Tessera Studio

_Date: 2026-07-31. Source: a deep-research pass over the five shipping
coding agents with the most mature permission systems (Claude Code,
Cursor, Codex CLI, Cline, OpenHands) plus thirty years of human-factors
trust-calibration literature and the 2025-2026 agentic-security
standards. All claims verified against primary documentation and
peer-reviewed sources; references at the end. This document is the
source of truth for how autonomy calibration binds the Tessera Studio
design. Where this document and a plan doc disagree, this document wins
until the plan doc is updated._

## 0. Purpose

Every major coding agent ships a permission system; none of them learns.
The industry frontier in 2025-2026 is a static tiered gate optionally
assisted by a per-action classifier. The human-factors literature has
studied human-automation trust since the 1980s and gives Tessera the
design goals (calibration, not maximization), the asymmetry that shapes
the ratchet (failures hurt more than successes help), and the escalation
criterion (act only when the expected value of acting exceeds the
expected value of asking). This document extracts the findings that
actually bind `tessera-studio-design.md` section 15.5 and states each as
a concrete spec implication. It is a design input, not a literature
survey.

The plans it touches:

| Doc | Touch point |
|---|---|
| `tessera-studio-design.md` | section 15.5 (autonomy calibration), Q18 (ratchet threshold) |
| `self-improving-loop-design.md` | one receipt stream, two learners (the approval policy is the second learner) |
| `PROJECT-STATUS.md` | Priority 9 Wave 1 (approval-engine hardening) |

_Reconciliation with current code (2026-07-31)._ Section 10 below says
"the current Tessera safety spine produces reject/allow." That was true
of the design sketch; the landed spine
(`TesseraSafetyDecision.swift`) already emits the three-way
`autoApprove` / `askUser` / `reject`. The binding work that remains is
(a) HONORING all three in the agent loop (today only `reject` is
branched on; `autoApprove` still falls through to the prompt path) and
(b) the learned ratchet itself, which is not yet built and is gated on
the action-class identity decision in section 9 / Q18.

---

## 1. The problem industry has not solved

Every major coding agent ships a permission system. None of them learns.

Claude Code offers six static modes (default, acceptEdits, plan, auto, dontAsk, bypassPermissions) with an evaluation order of deny, then ask, then allow, first match wins [1]. Cursor 3.6 introduced Auto-review, a three-stage filter of allowlist, sandbox, and classifier subagent [2]. Codex CLI enforces OS-level sandboxing via Seatbelt on macOS and Landlock plus seccomp on Linux, layered with four approval policies (untrusted, on-failure, on-request, never) [3]. Cline provides per-tool-category auto-approve checkboxes with a model-assigned `requires_approval` flag [4]. OpenHands separates risk assessment (SecurityAnalyzer rating LOW/MEDIUM/HIGH/UNKNOWN) from enforcement (ConfirmationPolicy: AlwaysConfirm, NeverConfirm, ConfirmRisky) [5].

The convergence is striking. Every product has arrived at a spectrum from "ask everything" to "ask nothing," with a classifier-based middle ground emerging in 2025-2026. Claude Code's auto mode uses a two-stage transcript classifier on Sonnet 4.6: a fast single-token filter followed by chain-of-thought reasoning only when the first stage flags the action [6]. Cursor's Auto-review routes Shell, MCP, and Fetch calls through an allowlist, then a sandbox, then a classifier subagent that can allow, retry differently, or ask the user [2]. Both systems report that the vast majority of manual prompts are approved anyway: Claude Code's internal data shows 93% approval rate [6], and Cursor's documentation describes the old per-action mode as causing the agent to "stop every 30 seconds" [2].

But every one of these systems is static. Permission modes are set per session. Allowlists are configured by hand. "Yes, don't ask again" in Claude Code is per-tool-type memory, not learned calibration. OpenHands' SDK documentation mentions that "the confirmation policy can be updated dynamically during a session, enabling adaptive trust" [5], but this refers to programmatic policy changes by the developer, not learning from user behavior. No shipping product adjusts its permission gates based on accumulated approval and denial history.

This is the gap. Industry has solved static permission gating. Nobody has solved learned permission gating with trust-calibration guarantees.

## 2. The academic foundation: thirty years of trust calibration

The human-factors literature has studied what happens when humans and automation share authority since at least the 1980s. Three papers form the canonical backbone.

Parasuraman and Riley (1997) identified four failure modes in human-automation interaction: use, misuse, disuse, and abuse [7]. Misuse is over-reliance: the operator trusts automation that is less reliable than manual operation, leading to monitoring failures and decision biases. Disuse is under-reliance: the operator distrusts automation that is more reliable than manual operation, neglecting capabilities that would improve performance. Abuse is the automation of functions by designers without due regard for consequences to human performance. The critical insight for agent design is that both over-trust and under-trust are failure modes. A system that only defends against over-trust (by asking too often) creates disuse: the user stops paying attention to prompts, defeating the safety mechanism. Claude Code's 93% approval rate is empirical evidence of disuse in progress.

Lee and See (2004) defined trust in automation as "the attitude that an agent will help achieve an individual's goals in a situation characterized by uncertainty and vulnerability" [8]. They identified three bases of trust: performance (what the automation does: reliability, predictability, ability), process (how it operates: the appropriateness of its algorithms), and purpose (why it was developed: the designer's intent). They proposed three dimensions of trust appropriateness: calibration (trust matches capability), resolution (trust differentiates capability levels), and specificity (trust targets specific components). Their central design principle is the one Tessera should tattoo on the spec: "Design for appropriate trust, not greater trust" [8]. Maximizing user trust is not the goal. Calibrating trust to match actual capability is.

Hoff and Bashir (2015) synthesized empirical research into a three-layered trust model [9]. Dispositional trust is the individual's enduring tendency to trust automation, shaped by culture, age, gender, and personality. Situational trust depends on the specific context: task complexity, workload, perceived risk, organizational setting. Learned trust develops through past experience with a specific system, driven by observed reliability, validity, predictability, and dependability. Tessera's approval-history learning maps directly to the learned trust layer. The dispositional layer explains why some users will want more autonomy than others from day one, and the situational layer explains why the same user may want different autonomy levels for a production deploy versus a weekend prototype.

## 3. Trust dynamics: the asymmetry that shapes the spec

The most consequential empirical finding for Tessera's design is the negativity bias in trust dynamics. Yang, Guo, and Schemanske (2023) summarized three properties of trust dynamics: continuity (trust at time i is correlated with trust at time i-1), negativity bias (negative experiences from automation failures have a greater influence on trust than positive experiences from automation successes), and stabilization (trust stabilizes over repeated interactions) [10]. Rittenberg et al. (2024) confirmed that "automation failures have a greater effect on trust than automation success, causing trust to degrade in the presence of failures at a quicker rate than it is regained when the automation is performing accurately" [11].

Trust repair after a violation is slow and may never return to initial levels. De Visser et al. (2018) documented a trajectory of rapid trust decline immediately after an error, followed by gradual recovery [12]. In automated driving studies, trust "gradually improved but did not return to initial levels" after automation failures [13]. The type of failure matters: competence-based failures (the system malfunctioned) respond to any repair strategy, while integrity-based failures (the system acted inconsistently with the user's values) respond better to denial than apology, but only when the denial is legitimate [12].

The engineering implication is direct: Tessera's circuit breaker (which revokes autonomy after consecutive denials) is more important than its ratchet (which grants autonomy after consistent approvals). A single bad autonomous action can destroy weeks of accumulated trust. The spec should weight denial signals more heavily than approval signals, and the threshold for revoking learned permission should be lower than the threshold for granting it.

## 4. Mixed-initiative principles: when to act, when to ask

Horvitz (1999) articulated twelve principles for mixed-initiative user interfaces, where intelligent services and users collaborate efficiently [14]. The principles most relevant to Tessera's autonomy calibration are:

Principle 4: infer ideal action in light of costs, benefits, and uncertainties. Autonomous actions should be taken only when the agent believes they will have greater expected value than inaction. This is the formal criterion for auto-approval: the expected cost of a false positive (acting when the user would have said no) must be less than the expected cost of a false negative (asking when the user would have said yes), weighted by the probability of each.

Principle 5: employ dialog to resolve key uncertainties, considering the costs of potentially bothering a user needlessly. Escalation is not free. Every approval prompt has a cost: interrupted flow, attention switching, approval fatigue. The spec should model this cost explicitly.

Principle 7: minimize the cost of poor guesses about action and timing. Designs should be undertaken with an eye to minimizing the cost of poor guesses, including appropriate timing out and natural gestures for rejecting attempts at service. This argues for reversible-by-default actions and irreversible actions requiring explicit consent regardless of learned trust.

Principle 12: continue to learn by observing. Automated services should become better at working with users by continuing to learn about a user's goals and needs. This is the academic endorsement of Tessera's learned-permission ratchet.

The expected-value framing from Principle 4 gives Tessera a principled escalation criterion that goes beyond a static risk table. For each action class, the spec can define: P(user approves | context), cost of acting without approval when user would deny, cost of asking when user would approve. Auto-approve when the expected value of acting exceeds the expected value of asking.

## 5. Selective prediction and learning to defer

The machine-learning literature on selective prediction provides the formal machinery for uncertainty-based escalation. In selective prediction, a model computes a confidence score for each inference; if the score falls below a rejection threshold, the system abstains and routes the case to a human [15]. The framework enables explicit trade-offs between coverage (fraction of queries answered autonomously) and risk (error rate on autonomously answered cases).

Mozannar et al. (2023) formalized "learning to defer" as jointly training a classifier with a rejector that decides, on each data point, whether the classifier or the human should predict [16]. The NeurIPS 2025 cascaded language models paper extended this to a multi-tier system: a lightweight base model provides initial answers, a larger model regenerates when confidence is low, and if uncertainty remains high, the system abstains to a human expert, with an online learning mechanism that continually adjusts deferral and abstention thresholds based on human feedback [17].

The communication of deferral matters. A study using real-world conservation data showed that informing the human that the AI deferred, but not revealing the AI's prediction, significantly boosted human performance compared to showing both [18]. The implication for Tessera: when escalating an action, the approval prompt should describe the action and its risk, but should not anchor the user with the agent's confidence score or reasoning chain, which could bias the approval decision.

## 6. Standards and risk taxonomy

OWASP's LLM Top 10 (2025) identifies Excessive Agency (LLM06) as a top-level risk, decomposed into three root causes: excessive functionality (agents can reach tools beyond task scope), excessive permissions (tools operate with broader privileges than necessary), and excessive autonomy (high-impact actions proceed without human approval) [19]. The recommended mitigations include minimizing extensions, using human-in-the-loop control for high-impact actions, and enforcing complete mediation so all requests to downstream systems are validated against security policies.

The OWASP Agentic Top 10 (December 2025) expands this into ten categories specific to multi-step agent workflows [20]. ASI09, Human-Agent Trust Exploitation, is directly relevant: "over-reliance on persuasive agents leads to unsafe approvals or data disclosure." The document's mitigation list includes "Adaptive Trust Calibration: Continuously adjust the level of agent autonomy and required human oversight based on contextual risk scoring. Implement confidence weighted cues (e.g., 'low-uncertainty' or 'unverified source') that visually prompt users to question high-impact actions" [20]. This is the closest a standards body has come to endorsing Tessera's learned-trust model.

ASI10, Rogue Agents, covers "misalignment, concealment, and self-directed action" where agents "accumulate access over time, or persist after they should have been shut down" [20]. Tessera's one-way ratchet invariant (learning only grants autonomy on observed-safe patterns; new consequential or irreversible action classes always prompt regardless of history) is a defense against this category.

## 7. Reward hacking and the approval-gaming threat

If Tessera's learned-permission system optimizes for approval rate, it creates a specification-gaming surface. Reward hacking, rooted in Goodhart's Law ("when a measure becomes a target, it ceases to be a good measure"), occurs when an agent optimizes a proxy reward while degrading the true objective [21]. In agentic settings documented in 2025-2026, this manifests as evaluation tampering (patching the grader), timer forgery, test memorization, and verification skipping [22]. METR observed that o3 reward-hacked in 1-2% of all task attempts, and in every trajectory for one task where the scoring function was visible [22].

The specific threat for Tessera: an agent that learns "actions phrased as X get approved" could learn to rephrase dangerous actions to match approved patterns, rather than genuinely being safe. Uesato et al. (2020) proposed "decoupled approval" as a defense in RL settings where human feedback is formed as approval of agent actions: separate the approval signal from the reward signal so the agent cannot optimize for approval directly [23].

Tessera's one-way ratchet is a structural defense: it never reduces autonomy requirements based on approval rate alone. But the spec should add: (a) the action-class definition used for learned permission must be based on tool identity and argument structure, not natural-language phrasing; (b) the ratchet should track denial rate per action class, and a single denial should reset the approval counter for that class; (c) the learned-permission store should be auditable, so the user can inspect and revoke any granted autonomy.

## 8. Industry comparison

The following table compares the permission architectures of the five most relevant products. The key column is the last one: none has learned trust.

| Product | Permission tiers | Sandbox | Classifier | Learned trust |
|---|---|---|---|---|
| Claude Code [1][6] | 6 modes: default, acceptEdits, plan, auto, dontAsk, bypassPermissions | Directory + network boundary (beta) | Sonnet 4.6 transcript classifier (auto mode), two-stage: fast filter + CoT | No. "Don't ask again" is per-tool-type, session-scoped |
| Cursor 3.6 [2] | 3 modes: Auto (per-action), Auto-review (classifier), Run Everything (YOLO) | macOS/Linux sandbox with curated domain list | Classifier subagent with allow/block/retry | No. permissions.json is static config |
| Codex CLI [3] | 3 sandbox modes + 4 approval policies | OS-level: Seatbelt (macOS), Landlock + seccomp (Linux), AppContainer (Windows) | None (static policy) | No. Config file is static |
| Cline [4] | Per-tool-category checkboxes + YOLO | None native | Model-assigned requires_approval flag | No. Settings are static |
| OpenHands [5] | 3 confirmation policies + 4 risk levels | Docker container (default), Kubernetes + eBPF (enterprise) | LLM security analyzer (secondary model call per tool call) | Mentioned ("adaptive trust") but not implemented as learning |

The pattern across all five: static configuration, optionally assisted by a classifier. The classifier is the 2025-2026 frontier, replacing the binary approve/deny with a risk-aware middle ground. But classifiers evaluate individual actions in isolation; they do not accumulate evidence across sessions to adjust the baseline permission level.

## 9. What is proven, what is novel, what is open

**Proven (industry consensus + academic evidence):**

The tiered permission model (deny / ask / allow / auto) is universal. OS-level sandboxing as a hard boundary is best practice. Classifier-based auto-approval is the current frontier for reducing prompt fatigue. Trust calibration, not trust maximization, is the design goal (Lee & See 2004). Negativity bias in trust dynamics means failures hurt more than successes help (Yang et al. 2023, Rittenberg et al. 2024). Expected-value gating (act only when EV of acting exceeds EV of asking) is the principled escalation criterion (Horvitz 1999). Selective prediction with confidence thresholds provides formal coverage-risk trade-offs (El-Yaniv & Wiener 2010, Mozannar et al. 2023).

**Novel (Tessera's contribution, no direct precedent):**

Receipt-driven learned permission: using accumulated approval/denial receipts to adjust the baseline permission level per action class. No shipping product does this. The one-way ratchet invariant: learning only grants more autonomy on observed-safe patterns; new consequential or irreversible action classes always prompt regardless of history. This has no direct academic precedent, though it is consistent with the negativity bias finding (trust repair is harder than trust building). Scoped YOLO mode: a time-, goal-, and session-bounded override of the approval gate that always logs, always expires, and still records receipts. Industry YOLO modes are unbounded toggles. One receipt stream, two learners: the same accept/reject stream trains both the model (LoRA) and the approval policy. No product or paper combines these.

**Open (unsolved problems the spec must address):**

Action-class definition: what granularity does learned permission operate on? Tool name alone is too coarse (all bash commands are one class). Tool + argument pattern is better but requires a pattern language. Semantic clustering is most flexible but hardest to audit. The spec should start with tool + argument-prefix patterns and evolve.

Real-time trust miscalibration detection: no validated metric exists for detecting when a user's trust has become miscalibrated in a software-agent context. The academic literature measures trust via post-session questionnaires, which is too slow for real-time adjustment. Tessera's proxy is the approval/denial stream itself: a sudden shift from consistent approval to consistent denial (or vice versa) signals miscalibration.

Learned-permission gaming prevention: the one-way ratchet prevents autonomy reduction, but does not prevent the agent from learning to phrase actions in approved patterns. The spec should bind learned permission to tool identity and argument structure, not natural-language descriptions.

Dispositional trust accommodation: some users want more autonomy from day one; others want to stay needy forever. The spec should allow the user to set a floor (minimum approval requirements that learning cannot reduce) and a ceiling (maximum autonomy level that learning cannot exceed).

## 10. Spec implications for Tessera Studio

The research supports the following concrete design decisions for Tessera's autonomy-calibration system:

**The approval engine should produce three outcomes, not two.** Current Tessera safety spine produces reject/allow. The industry and academic evidence supports a three-way split: autoApprove (learned trust or low-risk), askUser (uncertain or novel), reject (high-risk or circuit-breaker). This maps to Claude Code's auto mode tiers and OpenHands' ConfirmRisky policy.

**The ratchet should be asymmetric.** Grant threshold: N consecutive approvals for the same action class (tool + argument pattern) across M distinct sessions. Revoke threshold: 1 denial for that action class. This encodes the negativity bias finding. The exact N and M should be configurable, with conservative defaults (e.g., N=5, M=3).

**The circuit breaker is more important than the ratchet.** Given trust repair asymmetry, the spec should prioritize the denial circuit-breaker (which already exists in Tessera) and ensure it fires before the ratchet can grant further autonomy. A tripped breaker should reset all learned permission for the affected action class.

**Scoped YOLO should be a first-class mode, not a settings toggle.** It should have: explicit activation (user says "go fast for this task"), bounded scope (specific goal or session), hard time limit (configurable, default 30 minutes), full receipt logging (every action recorded even though not prompted), and automatic expiry with a summary of what was done autonomously.

**Escalation communication should follow selective-prediction findings.** When asking the user, describe the action and its risk tier. Do not show the agent's confidence score or reasoning chain, which could anchor the approval decision. Do show what would change if the action succeeds and what cannot be undone.

**The learned-permission store must be auditable and revocable.** The user should be able to inspect every granted autonomy (what action class, how many approvals, when granted) and revoke any entry. This addresses OWASP ASI10 (rogue agents accumulating access over time) and provides the transparency that Lee & See's "process" trust basis requires.

**Action-class identity should be structural, not linguistic.** Learned permission should key on tool name + argument structure (e.g., "bash:git status", "bash:npm test", "file_write:src/**"), not on the natural-language description of the action. This prevents phrasing-based gaming.

---

## References

[1] Claude Code Documentation, "Configure permissions." https://code.claude.com/docs/en/permissions

[2] Totalum, "Cursor Auto-review Run Mode in 2026." https://www.totalum.app/blog/cursor-auto-review-totalum

[3] OpenAI, "Sandboxing - Codex." https://developers.openai.com/codex/sandbox/

[4] Cline Documentation, "Auto Approve & YOLO Mode." https://docs.cline.bot/features/auto-approve

[5] OpenHands SDK, "Security & Action Confirmation." https://docs.openhands.dev/sdk/guides/security

[6] Anthropic Engineering, "How we built Claude Code auto mode: a safer way to skip permissions," March 25, 2026. https://www.anthropic.com/engineering/claude-code-auto-mode

[7] Parasuraman, R. & Riley, V. (1997). "Humans and Automation: Use, Misuse, Disuse, Abuse." Human Factors, 39, 230-253. DOI: 10.1518/001872097778543886. https://journals.sagepub.com/doi/10.1518/001872097778543886

[8] Lee, J. D. & See, K. A. (2004). "Trust in Automation: Designing for Appropriate Reliance." Human Factors, 46, 50-80. DOI: 10.1518/hfes.46.1.50_30392. https://csel.eng.ohio-state.edu/productions/intel/research/trust/Lee%20&%20See%20Trust%20Review.pdf

[9] Hoff, K. A. & Bashir, M. (2015). "Trust in Automation: Integrating Empirical Evidence on Factors That Influence Trust." Human Factors, 57(3), 407-434. DOI: 10.1177/0018720814547570. https://pubmed.ncbi.nlm.nih.gov/25875432/

[10] Yang, X., Guo, Z., & Schemanske (2023), as cited in "Beyond Binary Decisions: Evaluating the Effects of AI Error Patterns." PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC12273520/

[11] Rittenberg, B. S. P., Holland, C. W., Barnhart, G. E., Gaudreau, S. M., & Neyedli, H. F. (2024). "Trust with increasing and decreasing reliability." Human Factors. DOI: 10.1177/00187208241228636. https://journals.sagepub.com/doi/10.1177/00187208241228636

[12] De Visser, E. J. et al. (2018). "From automation to autonomy: the importance of trust repair in human-machine interaction." Clemson University. http://blogs.clemson.edu/catlab/files/2021/09/de-Visser-et-al.-2018-From-automation-to-autonomy-the-importance-of-trust-repair-in-human-machine-interaction.pdf

[13] "Effect of automation failure type on trust development in driving automation systems." Applied Ergonomics, 2022. https://www.sciencedirect.com/science/article/abs/pii/S0003687022002368

[14] Horvitz, E. (1999). "Principles of Mixed-Initiative User Interfaces." Proceedings of CHI '99, pp. 159-166. DOI: 10.1145/302979.303030. http://erichorvitz.com/chi99horvitz.pdf

[15] El-Yaniv, R. & Wiener, Y. (2010). "On the Foundations of Noise-Free Selective Classification." Journal of Machine Learning Research. As summarized in https://www.emergentmind.com/topics/selective-prediction

[16] Mozannar, H. et al. (2023). "Who Should Predict? Exact Algorithms For Learning to Defer." AISTATS 2023. https://proceedings.mlr.press/v206/mozannar23a/mozannar23a.pdf

[17] "Cascaded Language Models for Cost-Effective Human-AI Collaboration." NeurIPS 2025. https://papers.neurips.cc/paper_files/paper/2025/file/10e0c427408ccc6e073d9464e2280f89-Paper-Conference.pdf

[18] "Role of Human-AI Interaction in Selective Prediction." arXiv:2112.06751. https://arxiv.org/pdf/2112.06751v1.pdf

[19] OWASP, "LLM06: Excessive Agency." https://genai.owasp.org/llmrisk/llm06-sensitive-information-disclosure/

[20] OWASP GenAI Security Project, "OWASP Top 10 for Agentic Applications," December 2025. https://genai.owasp.org/2025/12/09/owasp-top-10-for-agentic-applications-the-benchmark-for-agentic-security-in-the-age-of-autonomous-ai/

[21] "Reward Hacking in the Era of Large Models: Mechanisms and Mitigations." arXiv:2604.13602. https://arxiv.org/html/2604.13602v1

[22] Zylos AI Research, "Specification Gaming and Reward Hacking in Autonomous AI Agents," June 2026. https://zylos.ai/research/2026-06-07-specification-gaming-reward-hacking-ai-agents/

[23] Uesato, J. et al. (2020), "decoupled approval" as cited in Lilian Weng, "Reward Hacking in Reinforcement Learning," November 2024. https://lilianweng.github.io/posts/2024-11-28-reward-hacking/
