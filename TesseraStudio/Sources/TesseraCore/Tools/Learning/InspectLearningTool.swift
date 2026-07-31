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

        return .ok(lines.joined(separator: "\n"), data: [
            "configured": .bool(configured),
            "teachers": .number(Double(teacherCount)),
            "assessments": .number(Double(assessments.count)),
            "recent_outcomes": .number(Double(recentCount)),
            "playbook_classes": .number(Double(playbook.count)),
        ])
    }
}
