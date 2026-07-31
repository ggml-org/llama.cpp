import Foundation

/// The transparency surface: inspect the learning subsystem's configuration,
/// teachers, assessments, recent outcomes, and playbook. Local read only.
public struct InspectLearningTool: TesseraTool {
    public let name = "inspect_learning"
    public let description = "Inspect the learning subsystem: configuration, teachers, assessments, recent outcomes, and the playbook."
    public let defaultApprovalLevel = ApprovalLevel.auto

    public let parameters = JSONSchema()

    public init() {}

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        let center = TesseraLearningCenter.shared
        let configured = center.isConfigured
        let teacherCount = center.escalation.availableTeachers().count
        let assessments = center.assessor.assessments()
        let recentCount = center.worldSignals.recent(limit: 10).count
        let playbook = center.playbook.all()
        let playbookEntries = playbook.values.reduce(0) { $0 + $1.count }
        let foraging = center.foraging.summary()
        let curation = center.curation.summary()
        let adaptation = center.scheduler.lastAdaptation()

        var lines: [String] = [
            "Learning subsystem",
            "- configured: \(configured)",
            "- available teachers: \(teacherCount)",
            "- teacher assessments: \(assessments.count)",
        ]
        for assessment in assessments {
            lines.append("  - \(assessment.teacherId): pass=\(String(format: "%.2f", assessment.worldGatePassFraction)) samples=\(assessment.samples) weight=\(String(format: "%.2f", assessment.effectiveWeight))")
        }
        lines.append("- recent outcomes (last 10): \(recentCount)")
        lines.append("- playbook: \(playbook.count) class(es), \(playbookEntries) strategy(ies)")
        lines.append("- foraging: \(foraging.total) event(s) - local-playbook: \(foraging.localPlaybook), local-reference: \(foraging.localReference), remote: \(foraging.remote) (resolved locally: \(foraging.resolvedLocally))")
        lines.append("- curation: \(curation.stored) outcome(s), \(curation.preferencePairs) preference pair(s), dedup-skipped \(curation.dedupSkipped), mean quality \(String(format: "%.2f", curation.meanQuality))")
        if let adaptation {
            lines.append("- last adaptation: guard=\(adaptation.guardPassed ? "pass" : "fail") adapted=\(adaptation.adapted) dryRun=\(adaptation.dryRun) backend=\(adaptation.backend)")
        } else {
            lines.append("- last adaptation: none")
        }

        var data: [String: JSONValue] = [
            "configured": .bool(configured),
            "teachers": .number(Double(teacherCount)),
            "assessments": .number(Double(assessments.count)),
            "recent_outcomes": .number(Double(recentCount)),
            "playbook_classes": .number(Double(playbook.count)),
            "foraging_total": .number(Double(foraging.total)),
            "foraging_resolved_locally": .number(Double(foraging.resolvedLocally)),
            "foraging_remote": .number(Double(foraging.remote)),
            "curation_stored": .number(Double(curation.stored)),
            "curation_preference_pairs": .number(Double(curation.preferencePairs)),
            "curation_dedup_skipped": .number(Double(curation.dedupSkipped)),
            "curation_mean_quality": .number(curation.meanQuality),
        ]
        if let adaptation {
            data["last_adaptation_guard_passed"] = .bool(adaptation.guardPassed)
            data["last_adaptation_adapted"] = .bool(adaptation.adapted)
            data["last_adaptation_backend"] = .string(adaptation.backend)
        }

        return .ok(lines.joined(separator: "\n"), data: data)
    }
}
