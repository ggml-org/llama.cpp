import Foundation

/// One teacher's routing decision for a fan-out. `effectiveWeight` is the
/// assessor's live estimate (or the configured prior when the teacher has no
/// assessment yet); `selected` marks the top-N that actually get queried.
/// Surfaced by TesseraEscalationService.routeTeachers so the selection and its
/// rationale are inspectable rather than implicit.
public struct TesseraTeacherRoute: Sendable, Equatable {
    public let teacherId: String
    public let effectiveWeight: Double
    public let selected: Bool
    public let reason: String
}

/// Concrete escalation ensemble. Fans one frame out to the available teachers
/// and collects their proposals, rather than betting on a single oracle
/// (design 4.1). Reuses the existing RemoteLLMProvider remote path for each
/// teacher call. Stateless apart from the configured teacher pool, so it is a
/// plain Sendable struct.
///
/// Ordering (design Phase 2): tier 1 first probes the LOCAL knowledge stores
/// and only escalates to remote teachers when still stuck, purifying the
/// escalation corpus toward genuinely reasoning-bound problems. Fan-out is
/// weighted by the assessor's per-teacher estimate, so the recurring
/// assessment steers usage - the structural defense against R3 (teacher bias).
public struct TesseraEscalationService: TesseraEscalating {
    private let teachers: [TesseraTeacherConfig]

    public init(teachers: [TesseraTeacherConfig] = TesseraSettings.learningTeachers) {
        self.teachers = teachers
    }

    public func availableTeachers() -> [TesseraTeacherConfig] {
        teachers.filter { !$0.apiKey.isEmpty }
    }

    // MARK: - Tier 1 (frame only, retrieve-before-escalate)

    public func escalate(frame: TesseraEscalationFrame) async throws -> TesseraEscalationResult {
        // Retrieve-before-escalate: a high-confidence LOCAL hit resolves the
        // frame with no egress at all.
        if let local = localResolution(for: frame) {
            return local
        }

        // WEB RETRIEVAL PLUG-IN POINT (later wave): a real web-search step
        // (TesseraWebSearch / Tavily, design Phase 2) would run here, cache its
        // hits into the reference store, and short-circuit exactly like the
        // local check above. This wave orders against the existing LOCAL stores
        // only; no web egress happens on this path.

        return try await fanOut(frame: frame, anonymizedPayload: nil)
    }

    // MARK: - Tier 2 (frame + scrubbed code)

    /// Fan the frame plus an already-scrubbed code payload out to the
    /// assessment-weighted teachers. The caller scrubs the payload before
    /// calling (see EscalateWithCodeTool); this method performs the egress and
    /// registers every proposal it creates. No retrieve-before-escalate here:
    /// tier 2 is the explicit "the frame could not capture it, a teacher needs
    /// the code" path.
    public func escalateWithCode(frame: TesseraEscalationFrame, anonymizedPayload: String) async throws -> TesseraEscalationResult {
        try await fanOut(frame: frame, anonymizedPayload: anonymizedPayload)
    }

    // MARK: - Assessment-weighted routing

    /// Order the pool by effective weight (highest first) and mark the top
    /// `cap` as selected. When the pool is no larger than the cap, everyone is
    /// selected (the "fan out to all when there are few" path). A teacher with
    /// no assessment yet falls back to its configured prior weight, so a cold
    /// teacher is never starved out before it has been tried. Equal weights
    /// break ties by teacher id so routing is deterministic.
    public func routeTeachers(_ pool: [TesseraTeacherConfig], cap: Int) -> [TesseraTeacherRoute] {
        let assessments = Self.assessmentWeights()
        let ranked = pool
            .map { teacher -> (teacher: TesseraTeacherConfig, weight: Double, assessed: Bool) in
                if let assessment = assessments[teacher.id] {
                    return (teacher, assessment.effectiveWeight, true)
                }
                return (teacher, teacher.weight, false)
            }
            .sorted { lhs, rhs in
                if lhs.weight != rhs.weight { return lhs.weight > rhs.weight }
                return lhs.teacher.id < rhs.teacher.id
            }

        let limit = cap > 0 ? cap : ranked.count
        return ranked.enumerated().map { index, entry in
            let selected = index < limit
            let weightText = String(format: "%.2f", entry.weight)
            let source = entry.assessed ? "assessment" : "configured prior"
            let reason: String
            if selected {
                reason = ranked.count > limit
                    ? "top-\(limit) by effective weight \(weightText) (\(source))"
                    : "all \(ranked.count) teacher(s) used; effective weight \(weightText) (\(source))"
            } else {
                reason = "below top-\(limit) cap; effective weight \(weightText) (\(source))"
            }
            return TesseraTeacherRoute(
                teacherId: entry.teacher.id,
                effectiveWeight: entry.weight,
                selected: selected,
                reason: reason
            )
        }
    }

    /// The assessor owns the live per-teacher estimates; the escalation
    /// service just surfaces them.
    public func assessTeachers() async throws -> [TesseraTeacherAssessment] {
        TesseraLearningCenter.shared.assessor.assessments()
    }

    // MARK: - Local resolution (retrieve-before-escalate)

    /// Probe the local knowledge stores for an answer to the frame. Returns a
    /// result built entirely from local proposals (no egress) when either store
    /// has a hit, or nil when still stuck. Proposals are tagged with synthetic
    /// teacher ids ("local-playbook" / "local-reference") so downstream
    /// attribution and receipts can tell them apart from remote teachers.
    private func localResolution(for frame: TesseraEscalationFrame) -> TesseraEscalationResult? {
        let center = TesseraLearningCenter.shared
        var proposals: [TesseraTeacherProposal] = []

        // Reasoning playbook: meta-strategies keyed by problem class.
        let strategies = center.playbook.strategies(forProblemClass: frame.problemClass)
        if !strategies.isEmpty {
            let body = strategies.map { "- \($0)" }.joined(separator: "\n")
            proposals.append(TesseraTeacherProposal(
                teacherId: "local-playbook",
                reasoning: "Local reasoning playbook for \"\(frame.problemClass)\":\n\(body)",
                metaMethod: strategies.joined(separator: "\n")
            ))
        }

        // Reference knowledge store: cached docs/examples (TTL-filtered),
        // matched against the natural-language summary.
        for hit in center.reference.lookup(query: frame.summary) {
            proposals.append(TesseraTeacherProposal(
                teacherId: "local-reference",
                reasoning: hit
            ))
        }

        guard !proposals.isEmpty else { return nil }
        for proposal in proposals {
            TesseraProposalRegistry.shared.register(proposal)
        }
        let sources = proposals.map(\.teacherId).reduce(into: [String]()) { acc, id in
            if !acc.contains(id) { acc.append(id) }
        }
        // Foraging capture: a locally-resolved frame is one less escalation.
        // Telemetry only - a store failure must not break the resolution.
        for id in sources {
            let source: TesseraForagingSource = (id == "local-reference") ? .localReference : .localPlaybook
            try? center.foraging.record(problemClass: frame.problemClass, source: source, teacherIds: [id])
        }
        return TesseraEscalationResult(frame: frame, proposals: proposals, fannedOutTo: sources)
    }

    // MARK: - Fan-out (shared by tier 1 and tier 2)

    private func fanOut(frame: TesseraEscalationFrame, anonymizedPayload: String?) async throws -> TesseraEscalationResult {
        let pool = availableTeachers()
        guard !pool.isEmpty else { throw TesseraLearningError.noTeachersAvailable }

        // Assessment-weighted routing: query the top-N by effective weight,
        // highest first (or all of them when the pool is small).
        let routes = routeTeachers(pool, cap: TesseraSettings.learningMaxConcurrentAgents)
        let selectedIds = routes.filter(\.selected).map(\.teacherId)
        let selected = selectedIds.compactMap { id in pool.first { $0.id == id } }

        let system = Self.systemPrompt
        let user = Self.userMessage(for: frame, anonymizedPayload: anonymizedPayload)

        let proposals = try await withThrowingTaskGroup(of: TesseraTeacherProposal?.self) { group in
            for teacher in selected {
                group.addTask {
                    do {
                        return try await Self.query(teacher: teacher, system: system, user: user)
                    } catch {
                        // One teacher failing must not sink the whole fan-out.
                        return nil
                    }
                }
            }
            var collected: [TesseraTeacherProposal] = []
            for try await proposal in group {
                if let proposal { collected.append(proposal) }
            }
            return collected
        }

        // Attribute every proposal to its teacher so a later world outcome can
        // be routed back to the right teacher (kills the "unknown" bucket).
        for proposal in proposals {
            TesseraProposalRegistry.shared.register(proposal)
        }

        // Foraging capture: this frame could not be resolved locally and fanned
        // out to remote teachers. Telemetry only - never sinks the fan-out.
        try? TesseraLearningCenter.shared.foraging.record(
            problemClass: frame.problemClass, source: .remote, teacherIds: selectedIds)

        // fannedOutTo carries the selected teachers in weight order (highest
        // first); routeTeachers exposes the full weight + rationale per teacher.
        return TesseraEscalationResult(frame: frame, proposals: proposals, fannedOutTo: selectedIds)
    }

    // MARK: - Teacher call

    private static func query(
        teacher: TesseraTeacherConfig,
        system: String,
        user: String
    ) async throws -> TesseraTeacherProposal {
        let provider = RemoteLLMProvider(
            baseURL: teacher.baseURL,
            apiKey: teacher.apiKey,
            modelName: teacher.model,
            useStreaming: false
        )
        let start = Date()
        let response = try await provider.complete(
            system: system,
            messages: [LLMMessage(role: "user", content: user)],
            tools: []
        )
        return TesseraTeacherProposal(
            teacherId: teacher.id,
            reasoning: response.content,
            metaMethod: extractMetaMethod(from: response.content),
            tokenCount: response.tokenCount,
            elapsedSeconds: Date().timeIntervalSince(start)
        )
    }

    // MARK: - Prompt building

    private static let systemPrompt = """
        You are a senior engineering teacher. A coding agent is stuck on a problem it cannot solve alone.
        First, reason about the problem out loud (object-layer reasoning): diagnose the likely cause, lay
        out the candidate fixes, and explain the trade-offs between them. Show your work.
        Then, on a line starting with "Meta-method:", give an explicit meta-method describing HOW to reason
        about this whole class of problem, so the agent learns the skill rather than just this one instance.
        Do not ask clarifying questions; reason from what you are given.
        """

    private static func userMessage(for frame: TesseraEscalationFrame, anonymizedPayload: String? = nil) -> String {
        var parts: [String] = [
            "Problem class: \(frame.problemClass)",
            "Summary: \(frame.summary)",
        ]
        if !frame.observedVsExpected.isEmpty {
            parts.append("Observed vs expected: \(frame.observedVsExpected)")
        }
        if !frame.failingTests.isEmpty {
            parts.append("Failing tests:\n" + frame.failingTests.map { "- \($0)" }.joined(separator: "\n"))
        }
        if !frame.redactedErrors.isEmpty {
            parts.append("Redacted errors:\n" + frame.redactedErrors.map { "- \($0)" }.joined(separator: "\n"))
        }
        if !frame.stackShape.isEmpty {
            parts.append("Stack shape: \(frame.stackShape)")
        }
        let base = parts.joined(separator: "\n\n")
        guard let anonymizedPayload, !anonymizedPayload.isEmpty else { return base }
        return base + "\n\nAnonymized code context (symbols scrubbed; de-anonymize the answer locally):\n\(anonymizedPayload)"
    }

    /// Live per-teacher estimates keyed by teacher id, read from the assessor.
    private static func assessmentWeights() -> [String: TesseraTeacherAssessment] {
        var map: [String: TesseraTeacherAssessment] = [:]
        for assessment in TesseraLearningCenter.shared.assessor.assessments() {
            map[assessment.teacherId] = assessment
        }
        return map
    }

    /// Best-effort extraction of the "Meta-method:" section from a teacher's
    /// response. Returns "" when the teacher did not externalize a method.
    private static func extractMetaMethod(from content: String) -> String {
        let markers = ["meta-method:", "meta method:", "how to reason", "reasoning method:"]
        let lines = content.components(separatedBy: .newlines)
        for (index, line) in lines.enumerated() {
            let lower = line.lowercased()
            guard markers.contains(where: { lower.contains($0) }) else { continue }
            var chunk: [String] = []
            for follow in lines[index...] {
                if follow.trimmingCharacters(in: .whitespaces).isEmpty && !chunk.isEmpty { break }
                chunk.append(follow)
                if chunk.count >= 6 { break }
            }
            return chunk.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return ""
    }
}
