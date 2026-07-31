import Foundation

/// Concrete escalation ensemble. Fans one frame out to EVERY available
/// teacher concurrently and collects their proposals, rather than betting on
/// a single oracle (design 4.1). Reuses the existing RemoteLLMProvider
/// remote path for each teacher call. Stateless apart from the configured
/// teacher pool, so it is a plain Sendable struct.
public struct TesseraEscalationService: TesseraEscalating {
    private let teachers: [TesseraTeacherConfig]

    public init(teachers: [TesseraTeacherConfig] = TesseraSettings.learningTeachers) {
        self.teachers = teachers
    }

    public func availableTeachers() -> [TesseraTeacherConfig] {
        teachers.filter { !$0.apiKey.isEmpty }
    }

    public func escalate(frame: TesseraEscalationFrame) async throws -> TesseraEscalationResult {
        let pool = availableTeachers()
        guard !pool.isEmpty else { throw TesseraLearningError.noTeachersAvailable }

        let system = Self.systemPrompt
        let user = Self.userMessage(for: frame)

        let proposals = try await withThrowingTaskGroup(of: TesseraTeacherProposal?.self) { group in
            for teacher in pool {
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

        return TesseraEscalationResult(frame: frame, proposals: proposals, fannedOutTo: pool.map(\.id))
    }

    /// The assessor owns the live per-teacher estimates; the escalation
    /// service just surfaces them.
    public func assessTeachers() async throws -> [TesseraTeacherAssessment] {
        TesseraLearningCenter.shared.assessor.assessments()
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

    private static func userMessage(for frame: TesseraEscalationFrame) -> String {
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
        return parts.joined(separator: "\n\n")
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
