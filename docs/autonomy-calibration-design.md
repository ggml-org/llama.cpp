# Autonomy Calibration Design: The Learned-Permission Ratchet, Scoped YOLO, and the Leashed Neural Approver

_Date: 2026-07-31. Status: Phases A-C LANDED and tested (section 18).
This is the detailed engineering spec for the outward
agent's autonomy system. It EXTENDS `tessera-studio-design.md` section
15.5 (the overview) and SUPERSEDES the per-session-only approval posture
of section 14.5. Evidence base: `research-autonomy-calibration-2026-07-31.md`
(industry practice + thirty years of human-factors trust calibration +
2025-2026 agentic-security standards). Where this document and a plan doc
disagree, this document wins until the plan doc is updated._

_Implementation notes (as-built, 2026-07-31): (1) the section 12 regime
shift is detected with a "high-regime latch" (tighten when a class that
was once above `hi` falls below `lo`); the naive first-half-vs-second-half
window comparison is unsatisfiable when `lo < hi/2` and was not shipped.
(2) A failed FIRST net training keeps the net cold (rule-based ratchet
alone) rather than rolling back to untrained random weights. (3) YOLO
scoping takes an explicit `sessionID` through `gateCheck`; the loop
publishes its session id so the Settings UI can bind a YOLO session to
the live loop._

_The inward half of "one machine, two payloads" lives in
`self-improving-loop-design.md`. The approval receipts this spec produces
are the shared substrate for both halves (section 14)._

---

## 0. Purpose, scope, non-goals

Studio starts NEEDY and earns autonomy. This spec defines exactly how:
the action-class identity scheme, the learned-permission store, the
asymmetric ratchet, the circuit-breaker interaction, the dispositional
floor/ceiling, scoped YOLO, the leashed neural approver, miscalibration
detection, audit/revocation, and the receipt integration.

In scope: everything needed to build the autonomy system against the
as-built safety spine (section 2).

Non-goals:
- A psychological model of the user. Learned permission is approval-HISTORY
  learning, not trust modeling. The proxy is the receipt stream, nothing
  deeper (research note section 5, section 9).
- A generic per-action classifier in the Claude Code / Cursor mold. Those
  judge each action in isolation and are not trained on the user's history
  (research note section 1, section 8). Tessera's learned layer is
  per-user and longitudinal; that is the differentiator.
- Autonomy that can cross the user's explicit walls. Learning moves within
  a dispositional floor/ceiling; it never crosses them (section 9).

## 1. Definitions

- **Action class.** A structural identity for a family of actions, derived
  from tool name + argument STRUCTURE only, never from natural language
  (section 3). Example: `bash:git`, `file_write:src/**`.
- **Gate.** The three-outcome verdict every action passes through:
  `autoApprove` / `askUser` / `reject` (as-built, section 2).
- **Ratchet.** The learned projection over approval history that promotes
  `askUser` to `autoApprove` for action classes the user consistently
  allows. ONE-WAY: it only grants on observed-safe patterns and revokes on
  a single denial (section 6).
- **Irreversible class.** An action class that can never be learned
  (destructive verbs, high/forbidden risk). Always prompts, forever
  (section 4).
- **Floor / ceiling.** The user-set band learning moves within (section 9).
- **Scoped YOLO.** A time-, goal-, and session-bounded override that
  auto-approves within scope, always logged, always expired (section 10).
- **Approver network.** A small, continuously-trained, per-user confidence
  estimator that modulates the ratchet WITHIN the safe envelope and acts as
  the smart-YOLO approver. Leashed: it predicts, never grants; it fails
  closed (section 11).
- **Receipt.** A `TesseraLearningReceipt` (kind `"approval"`) recording
  every gate decision. The single stream that trains the ratchet, the
  approver network, and the base-model LoRA (section 14).

## 2. The as-built three-outcome gate (grounding)

This spec builds on landed code. The relevant surface:

- `TesseraSafetyCheck`: `autoApprove` / `askUser` / `reject`.
- `TesseraSafetyDecision.check` (pure): `reject` if risk is `.forbidden`
  or policy is `.denied`; `askUser` if policy is `.prompt`; `askUser` if
  profile is `.restricted`; `autoApprove` if `sandboxEnforceable && risk ==
  .low`; else `askUser`.
- `TesseraActionRisk`: `low` / `medium` / `high` / `forbidden` (Comparable).
- `TesseraPermissionProfile`: `restricted` / `standard` / `elevated`.
- `ApprovalLevel`: `auto` / `notify` / `prompt` / `denied` (the MANUAL,
  per-tool override layer, persisted in UserDefaults).
- `TesseraDenialCircuitBreaker`: trips on 3 consecutive denials OR 10
  denials in the last 50 actions (`consecutiveLimit=3`, `windowSize=50`,
  `windowLimit=10`).
- `TesseraApprovalEngine`: `overrides`, `safetyCheck(...)`,
  `requestApproval(...)`, `requestApprovalForced(...)`, `circuitBreaker`.
- `TesseraActionVerifier.ruleBasedRisk`: destructive verbs -> `.high`,
  mutating verbs -> `.medium`, read-only verbs -> `.low`, unknown -> `.medium`.
- `PendingAction(toolName, arguments)`.
- The agent loop honors all three outcomes: `reject` blocks; `askUser`
  forces a prompt via `requestApprovalForced` (even for a generally-auto
  tool) and a user denial feeds the breaker; `autoApprove` runs directly.

The ratchet (section 6) is the ONLY thing that may promote `askUser` to
`autoApprove`. Nothing promotes `reject`. Nothing demotes `autoApprove`.

## 3. Action-class identity scheme (the granularity decision)

Identity is STRUCTURAL. The classifier reads `toolName` and `arguments`
only. It never reads the agent's natural-language description of the
action. This is the primary defense against approval-gaming (research note
section 7): an agent cannot rephrase a dangerous action into an approved
pattern, because phrasing is not an input.

```
classify(PendingAction) -> String   // the action-class id
```

Three pattern shapes, chosen by tool kind:

1. **Verb-prefix class** (shell-like tools: names containing `bash`,
   `shell`, `run`, `execute`, `terminal`). Take the command-string argument,
   split on whitespace, and keep the program token plus a known multi-word
   head. `git status`, `git diff`, `git log` all map to `bash:git`.
   `npm test`, `npm install` map to `bash:npm`. The multi-word head list is
   small and explicit: `git`, `npm`, `cargo`, `docker`, `swift`, `make`,
   `gh`, `uv`, `pip`. Unknown programs use the single program token.
   A bare program with no safe verb (e.g. `bash:rm`) is still a valid class
   id; it simply lands on the irreversible guard (section 4).

2. **Path-glob class** (file tools: names containing `file`, `write`,
   `read`, `edit`). Take the path argument and reduce it to a glob by
   keeping the first `D` path segments and replacing the rest with `**`.
   Default `D = 1`: `src/Agent/Loop.swift` -> `file_write:src/**`,
   `docs/spec.md` -> `file_write:docs/**`. `D` is configurable (section 16).
   Writes OUTSIDE the project root (absolute paths, `~`, `..`) collapse to
   a single `file_write:<external>` class that the irreversible guard
   treats as high-risk by default.

3. **Arg-shape class** (everything else). Tool name plus a stable hash of
   the sorted argument KEYS (structure, not values): `quantize#<hash>`.
   Two calls with the same argument keys land in the same class regardless
   of values, so the class is about the SHAPE of the call.

Fallback: a tool with no recognizable structure maps to the tool-only class
(`toolname`). Tool-only is coarse and used only as a last resort.

Design rules:
- Deterministic and pure: same action -> same class, always. Unit-testable
  with no I/O.
- Coarse on purpose. The class is meant to generalize across a family the
  user has demonstrated they trust. Over-specific classes (one per exact
  command) would never accumulate enough approvals to grant.
- Values are NOT part of the id (except the command head and the path
  prefix). This is also a privacy property (section 15): the store holds
  coarse patterns, not the user's actual commands and paths.

## 4. The irreversible-class guard (load-bearing, never delegated)

This is the invariant the whole system rests on. It is RULES, not ML, and
no learned component may override it.

An action class is irreversible if ANY of:
- Its verb head is in the destructive denylist: `rm`, `rmdir`, `del`,
  `delete`, `drop`, `purge`, `erase`, `format`, `mkfs`, `dd`, `shred`,
  `sudo`, `chmod`, `chown`, `kill`, `shutdown`, `reboot`.
- `TesseraActionVerifier.ruleBasedRisk` rates it `.high` or `.forbidden`
  (reuses the existing verifier; no new classifier).
- It is an external-path file write (`file_write:<external>`).
- It is on the user's manual denylist (section 13).

Properties:
- An irreversible class ALWAYS resolves to `askUser` or `reject`, never
  `autoApprove`, regardless of history, ratchet state, floor/ceiling, or
  scoped YOLO. "You approved 200 file edits" never becomes "auto-approve
  this `rm -rf`."
- The ratchet does not accumulate grant-progress for irreversible classes
  (it still records approvals/denials for audit).
- Scoped YOLO does NOT bypass this guard (section 10).
- The approver network cannot promote an irreversible class (section 11).

This is the structural defense against OWASP ASI10 (rogue agents
accumulating access over time) and the reason persisted approval is safe
here where the v1 per-session-only rule (14.5) assumed it was dangerous
(research note section 6).

## 5. The learned-permission store (data model + persistence)

Persisted via the existing `TesseraLearningStore` (file-backed JSON under
`ApplicationSupport/TesseraStudio/learning/`, atomic writes, corrupt-file
tolerant). New file: `learned-permissions.json`. No new persistence layer.

```swift
struct TesseraLearnedPermission: Codable, Sendable, Identifiable, Equatable {
    var id: String { actionClass }
    let actionClass: String              // "bash:git", "file_write:src/**"
    let irreversible: Bool               // frozen at first sight (section 4)
    let riskAtFirstSeen: TesseraActionRisk
    var consecutiveApprovals: Int        // current run; resets on any denial
    var distinctSessions: Int            // sessions with >= 1 approval
    var totalApprovals: Int
    var totalDenials: Int
    var granted: Bool                    // crossed the grant threshold?
    var grantedAt: Date?
    var revoked: Bool                    // user manually revoked
    var revokedAt: Date?
    var lastSessionID: String?
    var lastSeen: Date
}

struct TesseraPermissionConfig: Codable, Sendable {
    var grantThresholdN: Int             // consecutive approvals (default 5)
    var sessionThresholdM: Int           // distinct sessions (default 3)
    var floor: TesseraPermissionProfile  // default .standard
    var ceiling: AutonomyCeiling         // default .containedLowRiskOnly
    var pathGlobDepth: Int               // D for path-glob classes (default 1)
    var yoloDefaultMinutes: Int          // default 30
}

enum AutonomyCeiling: String, Codable, Sendable, CaseIterable {
    case containedLowRiskOnly   // ratchet promotes only sandbox-contained low-risk
    case anyNonIrreversible     // ratchet promotes any granted non-irreversible class
}

struct TesseraLearnedPermissionStore: Codable, Sendable {
    static let currentSchemaVersion = 1
    var schemaVersion: Int
    var config: TesseraPermissionConfig
    var entries: [String: TesseraLearnedPermission]   // keyed by actionClass
}
```

Synchronization: the owning service guards read-modify-write with a lock,
per the `TesseraLearningStore` contract (load + save are separate ops).
Missing/corrupt file decodes to an empty store with default config.

## 6. The ratchet algorithm (asymmetric)

Trust repairs slower than it builds; failures cut more than successes
restore (research note section 3). Encode this directly: grant is slow,
revoke is instant.

Grant: a class becomes `granted` when
`consecutiveApprovals >= N` AND `distinctSessions >= M` AND NOT
`irreversible` AND NOT `revoked` AND `risk < .high`. Defaults `N = 5`,
`M = 3` (section 16).

Revoke (automatic): a SINGLE denial of a class sets `granted = false` and
`consecutiveApprovals = 0`.

Revoke (manual): the user sets `revoked = true`; a revoked class never
auto-grants until the user un-revokes (section 13).

```
record(actionClass, approved, sessionID, risk):
    entry = entries[actionClass] ?? new(actionClass, risk)
    entry.irreversible = entry.irreversible || isIrreversible(actionClass, risk)
    entry.lastSeen = now

    if approved:
        if sessionID != entry.lastSessionID:
            entry.distinctSessions += 1
            entry.lastSessionID = sessionID
        entry.consecutiveApprovals += 1
        entry.totalApprovals += 1
        if !entry.granted && !entry.revoked && !entry.irreversible
           && entry.consecutiveApprovals >= N
           && entry.distinctSessions >= M
           && risk < .high:
            entry.granted = true
            entry.grantedAt = now
    else:
        entry.consecutiveApprovals = 0      // asymmetric: one denial resets
        entry.totalDenials += 1
        if entry.granted:
            entry.granted = false           // auto-revoke on single denial

    entries[actionClass] = entry
    emit approval receipt (section 14)
```

The ratchet is monotonic in the safe direction: over time it only ever adds
autonomy on observed-safe patterns, and any negative signal removes it.
This is the academic endorsement of "continue to learn by observing"
(Horvitz 1999 principle 12) bounded by the negativity bias.

## 7. Precedence model

The gate resolves top-down; the first match wins. This keeps manual intent
above learning, and the hard invariants above everything.

1. `risk == .forbidden` OR manual override == `.denied` -> `reject`.
2. Circuit breaker tripped -> `reject` (and learned grants are SUSPENDED,
   section 8).
3. Manual override == `.prompt` -> `askUser`. (An explicit "always ask"
   wins over learning; this is the floor in action, section 9.)
4. Irreversible class (section 4) -> `askUser` (never autoApprove).
5. Base decision `autoApprove` (sandbox-contained low-risk under an
   auto/notify policy) -> `autoApprove`.
6. Base decision `askUser` AND class `granted` AND within floor/ceiling
   AND (no scoped-YOLO conflict) -> `autoApprove` (LEARNED trust).
7. Otherwise -> `askUser`.

The approver network (section 11) only ever influences step 6 (whether a
granted class is confidently auto-approved vs re-prompted) and step 10's
YOLO path. It never reaches steps 1-4.

## 8. Circuit-breaker interaction (suspension, not deletion)

The breaker is more important than the ratchet (research note section 10).
Two distinct jobs, kept separate:

- **Per-class reset** is handled by the ratchet's single-denial auto-revoke
  (section 6). The denial that contributes to a trip already revokes its own
  class.
- **Loop interrupt** is the breaker's existing job: 3 consecutive denials or
  10-in-50 interrupts the agent loop.

Additional rule when the breaker trips: learned permission is SUSPENDED,
not deleted. While tripped, every `granted` class falls back to `askUser`
(precedence step 2). When the breaker is reset (user re-arms, or a new
session starts clean), grants RESTORE unless the user revoked them. This
honors "breaker outranks ratchet" (no autonomy while tripped) and the
trust-repair finding (a bad cluster tightens everything) without destroying
the audit trail. Suspension vs deletion vs per-class-only is a tunable
(section 20); suspension is the default.

## 9. Dispositional floor and ceiling

Some users want more autonomy on day one; others want to stay needy forever
(Hoff & Bashir 2015 dispositional layer; research note section 2). Learning
moves within a user-set band; it never crosses the walls.

- **Floor** (`TesseraPermissionProfile`, default `.standard`): the minimum
  approval requirement learning cannot reduce. Set floor to `.restricted`
  and NOTHING ever auto-approves via the ratchet (restricted never
  auto-approves) - the user stays permanently needy. Precedence step 3
  (manual `.prompt`) is the per-tool expression of the floor.
- **Ceiling** (`AutonomyCeiling`, default `.containedLowRiskOnly`): the
  maximum autonomy learning cannot exceed.
  - `.containedLowRiskOnly`: the ratchet may only promote a granted class
    when the action is sandbox-contained AND low-risk (mirrors the base
    autoApprove rule). Conservative default.
  - `.anyNonIrreversible`: the ratchet may promote any granted
    non-irreversible class regardless of sandbox containment. The user opts
    up to this knowingly.

Both are user-tunable at any time and take effect immediately.

## 10. Scoped YOLO mode (rule-based, first-class)

Not a settings toggle. A bounded, explicit, logged override.

```swift
struct TesseraYoloSession: Codable, Sendable, Identifiable {
    let id: String
    let goal: String?            // the task it is scoped to (user-stated)
    let sessionID: String        // hard session bound
    let startedAt: Date
    let expiresAt: Date          // hard time bound
    let reason: String           // user's stated reason (audit)
    var actionCount: Int         // actions auto-approved under it
}
```

Activation: an explicit user action ("go fast for this task"), with a
stated goal/session and a hard expiry (default 30 min, configurable).

While active, unexpired, and in-scope:
- The gate returns `autoApprove` for actions it would otherwise `askUser`,
  EXCEPT irreversible classes (section 4), which ALWAYS prompt even in
  YOLO. YOLO reduces prompt fatigue; it does not arm `rm -rf`.
- `reject` is unchanged (forbidden/denied/breaker still reject).
- Every action under YOLO is receipt-logged at full fidelity with
  `yoloActive = true` (section 14). A YOLO session is the richest training
  data the harness gets - many actions, fast - and feeds the loop rather
  than escaping it.

Expiry: automatic deactivation at `expiresAt` or on session end, whichever
first, plus a summary of what ran autonomously (count, classes, any
denials). YOLO never persists across sessions.

Industry YOLO modes are unbounded toggles; the bounding (time + goal +
session + irreversible guard + full receipts) is the point.

## 11. The learned approver network (leashed, Phase C+)

The industry frontier is a per-action classifier (Claude Code's Sonnet
transcript classifier; Cursor's classifier subagent; research note section
1, section 8). Those are generic and stateless. Tessera's version is
per-user and CONTINUOUSLY TRAINED on the local receipt stream - the same
stream that trains the base-model LoRA (section 14). That is the novelty.

But a naive "the net is the approver" is unsafe (research note section 3,
section 7). So the net is LEASHED.

### 11.1 The safety layering (the central principle)

Two layers, strictly ordered:

- **Load-bearing invariant layer (rules, no ML, auditable, monotonic).**
  Sections 4, 6, 8: the irreversible guard, the one-way ratchet's hard
  rules, the breaker. These can never be violated by any learned component.
  They are the answer to "why is persisted approval safe here."
- **Learned modulation layer (ML allowed, fails closed, bounded).** The
  approver network. It operates ONLY within the envelope the rules define.
  Where the rules say "prompt" or "reject," the net has no say. Where the
  rules permit autonomy, the net modulates HOW confidently.

The net FAILS CLOSED: any error, low confidence, missing features, or cold
start resolves to `askUser` (prompt), never to `autoApprove`. This mirrors
the fail-closed verifier (S2).

### 11.2 What the net is, precisely

A calibrated confidence estimator. For a pending action it outputs
`P(user approves | action features, context)` - the selective-prediction
confidence of research note section 5. It is NOT a grant authority. Per
"decoupled approval" (Uesato et al. 2020; research note section 7), the
prediction signal is separated from the grant decision, so the net cannot
optimize its way into gaming the grants.

### 11.3 Inputs (structural only)

The net sees the action-class features and context, NEVER the agent's
natural-language reasoning:
- One-hot / embedding of the action class (section 3).
- `TesseraActionRisk` level.
- Sandbox-contained boolean.
- Ratchet state for the class (consecutiveApprovals, distinctSessions,
  granted, totalDenials).
- Session context: scoped-YOLO active, recent denial rate in this session,
  time since last denial.
- Dispositional band (floor/ceiling).

Excluding the NL description blocks both the anchoring bias (research note
section 5: revealing the AI's prediction degrades the human's decision) and
the phrasing-gaming surface (section 7).

### 11.4 Outputs (confidence, not grants)

The net's confidence is used in three bounded ways:
1. **Ratchet modulation.** For a `granted` class at precedence step 6, a
   high-confidence net auto-approves; a low-confidence net re-prompts even
   though the class is granted. The net can TIGHTEN a grant (re-prompt) but
   cannot LOOSEN the rules (never promotes an un-granted or irreversible
   class). Effective grant thresholds can adapt: a class the net is very
   confident about may grant at a lower `N`; an uncertain one may need a
   higher `N`. The hard floor on `N` (section 9) still holds.
2. **Prompt triage.** When several actions queue, order/deprioritize by
   predicted-approval so the user sees the genuinely uncertain ones first.
3. **Smart YOLO (the headline role, section 11.7).**

### 11.5 Training (continuous, idle window, one stream three learners)

The net trains on `TesseraLearningReceipt` records of kind `"approval"`
(section 14): features -> the user's actual choice (approved/denied). This
is the SAME receipt stream that trains the base-model LoRA
(`self-improving-loop-design.md` 4.4-4.5) and the ratchet (section 6): one
receipt stream, THREE learners.

Training runs in the existing idle-training window
(`TesseraTrainingOrchestrator`), incrementally on new receipts. The net is
small (a shallow MLP over the structural features), trains fast, fully
local, no egress. Cold start: until the user has enough receipts (a
configurable warmup, e.g. 50 approval receipts), the net is inactive and
the pure rule-based ratchet (sections 6-10) runs alone. The system is fully
functional with no neural component; the net is an enhancement, not a
dependency.

### 11.6 Collapse guard for the net (calibration rollback)

A continuously-trained net can drift - track a user's changing mind
(good) or drift toward gaming / miscalibration (bad). Guard it with the
SAME collapse-guard pattern the LoRA uses (`self-improving-loop-design.md`
capability-eval gate):
- Hold out the most recent K receipts as a calibration set.
- Periodically measure the net's calibration (predicted vs actual approval
  rate, e.g. expected calibration error) and its denial recall (does it
  still flag the things the user denies?).
- If calibration degrades beyond epsilon, or denial recall drops, ROLL BACK
  to the previous net weights and fall back to the rule-based ratchet until
  recalibrated. The net, like the model, is not allowed to get worse.

### 11.7 Smart YOLO (the headline role)

Plain scoped YOLO (section 10) is a blanket: approve everything in scope
except irreversible classes. The net makes YOLO CALIBRATED: within a YOLO
session, the net auto-approves actions it is confident the user would
approve, and still PROMPTS the outliers it is unsure about - even though
YOLO is active. "Go fast, but still catch the one thing I would actually
say no to." This is strictly better than blanket YOLO and remains safe
because YOLO is an explicit, bounded, logged user opt-in, and the
irreversible guard still always prompts. Smart YOLO is the net's killer
application and the reason to build it.

### 11.8 Hazards and mitigations

| Hazard | Why it bites | Mitigation |
|---|---|---|
| Base-rate / disuse bias | ~93% of actions are approved (research note section 1); a naive net learns "approve everything" = trust maximization, the opposite of calibration | Cost-sensitive training: weight denials by the negativity-bias factor (section 6); optimize calibration and denial recall, not raw accuracy; the collapse guard (11.6) gates on denial recall |
| Approval gaming / Goodhart | A net that controls grants and trains on grants can learn to game them | Decoupled approval (11.2): the net predicts, the ratchet/user grants; structural-only inputs (11.3) remove the phrasing surface |
| Unauditable black box | "The net said approve" is not inspectable; conflicts with the auditable/revocable requirement and OWASP ASI10 transparency | The net never grants; every grant is still a ratchet entry (section 5) the user can inspect and revoke (section 13). The net's confidence is logged in the receipt for inspection but is not the authority |
| Dangerous generalization to unseen classes | The net's strength (generalize to new classes) is exactly what the irreversible/new-class guard forbids | The net cannot promote an un-granted or irreversible class (11.1, 11.4). It may SURFACE a recommendation ("you usually approve X-like things; pre-approve this class?") but the user confirms; it never silently auto-approves a new class |
| Cold start | Too little data to generalize early | Warmup threshold (11.5); rule-based ratchet runs alone until then; fail closed throughout |
| Drift | Continuous training tracks noise or gaming | Calibration collapse guard with rollback (11.6) |

### 11.9 Recommendation-confirmation: UX and training signal

The safe form of generalization is not "the net auto-approves a new
class" but "the system asks the user to pre-approve a class and records
the answer." This is first-class, and it earns its keep twice.

**The mechanism.** The system surfaces a recommendation: "You have
approved `bash:git` 4 times across 2 sessions. Pre-approve this class so
Tessera stops asking?" with Confirm / Not now / Never for this class.
- **Confirm** grants the class immediately (an explicit user grant,
  stronger than an accumulated one) and records it.
- **Not now** leaves the class accumulating normally.
- **Never** adds the class to the manual denylist (irreversible, section
  4).

**Two triggers, one flow.**
- *Rule-based trigger (Phase A, no net).* A class is worth recommending
  when it is not granted, not irreversible, not revoked, and shows a strong
  pattern: `consecutiveApprovals >= recommendationFloor` (default
  `max(2, N - 2)`) across `>= 1` session. The ratchet state alone supplies
  this; no ML required. So the flow exists from the first build.
- *Net trigger (Phase C).* The net additionally recommends NEW classes it
  is confident the user would approve (generalization), still gated on user
  confirmation. This is the only place generalization reaches autonomy, and
  it never bypasses the confirm step.

**Why it is also training material.** A confirmation is a CLASS-LEVEL
label: one explicit "pre-approve `bash:git`" is a cleaner, denser signal
than many per-action approvals, and a "Never" is an explicit negative. The
recommendation stream (section 14, `source = "recommendation"`) is
therefore high-value ground truth for the future approver network: it
labels whole classes, including the user's NEGATIVES, which the imbalanced
per-action stream is sparse on. The UX that makes autonomy feel
collaborative is the same mechanism that bootstraps a better classifier.
This is deliberate: the product surface and the training pipeline are the
same receipt stream.

## 12. Miscalibration detection (regime-shift, not psychology)

No validated real-time trust-miscalibration metric exists for software
agents; the literature measures trust via post-session questionnaires, too
slow for real time (research note section 9). Tessera's proxy is the
receipt stream itself.

Mechanism (dumb, auditable, receipt-only):
- Track a rolling window of the user's recent gate outcomes (reuse the
  breaker's window concept).
- Compute the approval rate over the last K decisions, per class and
  globally.
- Detect a regime shift: a class (or the global stream) that was
  consistently approved (rate > `hi`, e.g. 0.8) flips to consistently
  denied (rate < `lo`, e.g. 0.3) within a short window.
- Response: TIGHTEN. Suspend learned promotion for the affected class (or
  globally), and surface a NON-ANCHORING notice ("your recent responses
  changed; Tessera will ask more for a while"). Do not show a confidence
  score or the agent's reasoning (research note section 5).

This complements the breaker (which reacts to denial CLUSTERS) by reacting
to denial REGIME SHIFTS. Thresholds (`hi`, `lo`, K) are placeholders to
tune with real streams (section 20).

## 13. Audit and revocation

The store is a plain JSON file, inspectable and deletable. A UI surface
(in Settings, with a link from the Runs audit log) lists every entry:
`actionClass`, `irreversible`, `riskAtFirstSeen`, `consecutiveApprovals`,
`distinctSessions`, `totalApprovals`, `totalDenials`, `granted`,
`grantedAt`, `revoked`.

User actions per entry:
- **Revoke** (`revoked = true`): the class stops auto-approving and never
  re-grants until un-revoked.
- **Un-revoke** (`revoked = false`): resumes accumulation from zero.
- **Add to denylist**: marks the class irreversible (section 4), forever
  prompting.

Global actions:
- **Reset all**: clears every grant (entries kept for audit, `granted =
  false`).
- **Purge**: deletes the store and the approval receipts (privacy, section
  15).

This is the transparency Lee & See's "process" trust basis requires and the
antidote to access accumulating out of sight (research note section 6,
section 10).

## 14. Receipt integration (one stream, three learners)

Every gate decision emits a `TesseraLearningReceipt` with a new kind value
`"approval"` (added to the existing kind vocabulary; no new receipt type).
Payload:

```
{
  "actionClass":   "bash:git",
  "toolName":      "bash",
  "risk":          "low",
  "sandboxed":     true,
  "decision":      "autoApprove" | "askUser" | "reject",
  "userChoice":    "approved" | "denied" | "none",   // none = no prompt shown
  "source":        "rule" | "ratchet" | "yolo" | "net" | "recommendation",
  "grantedBefore": false,
  "grantedAfter":  true,
  "netConfidence": 0.93,        // null until the net exists
  "yoloActive":    false,
  "sessionID":     "..."
}
```

This single stream feeds all three learners:
1. The ratchet (section 6) - updates grant state per class.
2. The approver network (section 11) - trains features -> userChoice.
3. The base-model LoRA (`self-improving-loop-design.md` 4.4-4.5) - the
   accept/reject + outcome signal.

Capture is on by default and local; learning and egress are opt-in
(`self-improving-loop-design.md` section 6). The longitudinal local
approval history is the moat no cloud product can replicate (15.6).

## 15. Privacy and egress posture

- The store and the approval receipts are LOCAL-ONLY, under
  ApplicationSupport. They never egress.
- The store holds COARSE PATTERNS (action classes), not full argument
  values (section 3). The user's actual commands and paths live only in the
  journal receipts (also local), not in the learned-permission store.
- The approver network's weights are local model artifacts; training is
  local; no features leave the machine.
- Everything is deletable (section 13 purge). This is the no-egress-by-default
  doctrine applied to the autonomy system.

## 16. Migration, defaults, persistence

Fresh install:
- Empty store; `config = { N: 5, M: 3, floor: .standard, ceiling:
  .containedLowRiskOnly, pathGlobDepth: 1, yoloDefaultMinutes: 30 }`.
- The existing per-tool `overrides` (`ApprovalLevel`, UserDefaults) remain
  the MANUAL layer; the ratchet is the LEARNED layer beneath. Precedence in
  section 7 defines their interaction.
- Approver network absent until warmup (section 11.5); rule-based system is
  fully functional without it.

Schema-versioned (`schemaVersion = 1`); a future migration bumps the
version and migrates entries. Corrupt/missing file decodes to defaults
(`TesseraLearningStore` contract).

Defaults are conservative on purpose ("start needy"): high-ish grant
thresholds, contained-low-risk ceiling, standard floor. Users loosen them
knowingly.

## 17. Test plan

Pure/offline first (the spine's existing test style):
- **Classifier**: deterministic; structural; never reads NL; verb-prefix,
  path-glob, arg-shape, and fallback cases; external-path collapse.
- **Irreversible guard**: denylist verbs, high/forbidden risk, external
  writes, manual denylist; guard holds regardless of history/YOLO/net.
- **Ratchet state machine**: grant after N approvals across M sessions;
  single-denial revoke; irreversible never grants; revoked stays revoked;
  distinct-session counting; monotonic-in-safe-direction property test.
- **Precedence**: the seven-step table, especially manual `.prompt` beating
  a grant, and `.denied`/forbidden beating everything.
- **Breaker interaction**: trip suspends grants (askUser fallback); reset
  restores non-revoked grants; single-denial auto-revoke is independent.
- **Floor/ceiling**: restricted floor blocks all learned auto-approve;
  containedLowRiskOnly vs anyNonIrreversible.
- **Scoped YOLO**: auto-approves in scope; irreversible still prompts;
  expiry deactivates + emits summary; session-bound; receipts logged.
- **Miscalibration**: regime shift (hi -> lo) tightens; stable stream does
  not.
- **Persistence**: round-trip via `TesseraLearningStore`; corrupt-file
  tolerance; schema version.
- **Receipt emission**: every decision emits a kind `"approval"` receipt
  with the right payload.

Neural (Phase C+):
- **Fail-closed**: low confidence / error / cold start -> askUser.
- **Leash**: net cannot promote un-granted or irreversible classes
  (property test over random net outputs).
- **Collapse guard**: injected miscalibration triggers rollback; denial
  recall gate.
- **Smart YOLO**: confident actions auto-approve in YOLO; uncertain ones
  still prompt; irreversible always prompts.
- **Decoupling**: changing the agent's NL description does not change the
  net's output (NL is not an input).

## 18. Phasing

- **Phase A (landed) - the rule-based core.** Action-class classifier
  (section 3); irreversible guard (section 4); store + persistence (section
  5); ratchet state machine (section 6); precedence integration into the
  gate (section 7, the `askUser -> autoApprove` promotion); receipt
  emission (section 14); and the rule-based recommendation-confirmation
  flow (section 11.9, rule-based trigger only - the logic and the
  class-level training signal, not yet a UI surface). This is load-bearing
  and ships with no ML.
- **Phase B (landed) - bounded autonomy UX.** Scoped YOLO (section 10); breaker
  suspension semantics (section 8); dispositional floor/ceiling (section
  9); audit + revocation UI (section 13).
- **Phase C (landed) - the leashed neural approver.** Confidence estimator (section
  11.2-11.4); continuous idle training (11.5); calibration collapse guard
  (11.6); smart YOLO (11.7); miscalibration regime-shift detector (section
  12). Gated on enough receipt volume (warmup) and on Phase A/B being
  stable. The rule-based system remains the fallback throughout.

Each phase keeps the system safe and functional; the neural layer is an
enhancement that can be removed without breaking autonomy.

## 19. Ratified decisions log

Decisions ratified through 2026-07-31 (user direction + this spec):
1. Action-class identity is STRUCTURAL (tool + argument structure), never
   natural language. Granularity: verb-prefix / path-glob / arg-shape,
   config-driven, no ML in the classifier.
2. Ratchet defaults N=5 consecutive approvals across M=3 distinct sessions;
   user-tunable; conservative ("start needy").
3. The ratchet is ASYMMETRIC: grant slow, revoke on a single denial
   (negativity bias).
4. The circuit breaker outranks the ratchet; a trip SUSPENDS learned grants
   (restored on reset unless revoked), and interrupts the loop.
5. Irreversible classes (destructive verbs, high/forbidden risk, external
   writes, manual denylist) are permanently un-trainable and always prompt,
   even under scoped YOLO and the neural approver.
6. Dispositional floor (default `.standard`) and ceiling (default
   `.containedLowRiskOnly`) bound learning; the user can stay needy forever
   or opt up knowingly.
7. Scoped YOLO is first-class: time- + goal- + session-bounded, always
   logged, always expired, irreversible guard intact.
8. The learned-permission store is auditable and revocable, local-only,
   holds coarse patterns, and is deletable.
9. The approval stream reuses `TesseraLearningReceipt` (kind `"approval"`);
   one stream, three learners (ratchet, approver net, base-model LoRA).
10. The neural approver is LEASHED: a fail-closed confidence estimator that
   modulates within the rules' envelope, predicts but never grants
   (decoupled approval), trains continuously in the idle window, and is
   gated by a calibration collapse guard. Smart YOLO is its headline role.
   It is an enhancement, never the load-bearing safety mechanism.
11. Miscalibration is detected via a receipt-stream regime shift, not a
   psychological model; the response is to tighten and to notify without
   anchoring.
12. Generalization reaches autonomy ONLY through recommendation-
   confirmation (section 11.9): the system asks the user to pre-approve a
   class and records the answer; it never silently auto-approves a new
   class. The rule-based trigger ships in Phase A; the net trigger lands in
   Phase C. A confirmation is a class-level label, so the recommendation
   stream is also the future approver network's cleanest training signal
   (including explicit negatives). The collaborative UX and the training
   pipeline are the same receipt stream, by design.

## 20. Residual open questions

- **Verb-prefix coverage.** Which multi-word programs get two-token classes
  beyond the starter list (`git`, `npm`, `cargo`, `docker`, `swift`,
  `make`, `gh`, `uv`, `pip`)? Lean: extend only by evidence from the
  receipt stream.
- **Path-glob depth D.** Default 1; is 2 better for monorepos? Tune with
  data; keep it coarse.
- **Breaker-trip semantics.** Suspension (default) vs deletion vs
  per-class-only. Lean: suspension; revisit with real trust-repair data.
- **Miscalibration thresholds.** `hi=0.8`, `lo=0.3`, window K are
  placeholders. Tune against real streams.
- **Net warmup size.** 50 approval receipts is a guess; calibrate.
- **Per-runtime grant scope.** Should a class granted under the local model
  transfer to a cloud-teacher session? Lean: keep grants global for v1;
  revisit if teacher sessions show different approval behavior.
- **Denial-weight for net training.** The negativity-bias weighting factor
  (section 11.8) needs an empirical value; start high (denials cost several
  approvals' worth) and tune by calibration.
