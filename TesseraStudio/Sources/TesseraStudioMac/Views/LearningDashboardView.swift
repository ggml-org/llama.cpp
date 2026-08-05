import SwiftUI
import TesseraCore

/// Read-only transparency surface for the self-improving learning loop. Reads
/// the latest state from TesseraLearningCenter.shared on appear and on demand.
/// Plain labels only (no Charts framework); macOS only.
struct LearningDashboardView: View {
    @State private var capability: TesseraCapabilityEvalRecord?
    @State private var adaptation: TesseraAdaptationRecord?
    @State private var teachers: [TesseraTeacherAssessment] = []
    @State private var foraging = TesseraForagingSummary()
    @State private var curation = TesseraCurationSummary()
    /// Regenerated on Refresh so the training section reloads its
    /// read-only state without disturbing any in-flight run.
    @State private var refreshID = UUID()

    var body: some View {
        List {
            LearningTrainingSection(refreshID: refreshID)
            capabilitySection
            adaptationSection
            teachersSection
            foragingSection
            curationSection
        }
        .navigationTitle("Learning")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button("Refresh", systemImage: "arrow.clockwise") { load() }
            }
        }
        .onAppear { load() }
    }

    private var capabilitySection: some View {
        Section("Capability") {
            if let capability {
                ForEach(TesseraCapabilityScore.axisNames, id: \.self) { axis in
                    LabeledContent(axis, value: String(format: "%.2f", capability.score[axis]))
                }
                LabeledContent("weighted sum", value: String(format: "%.2f", capability.weightedSum))
                LabeledContent("backend", value: capability.backend)
                LabeledContent("evaluated", value: capability.timestamp.formatted())
            } else {
                Text("No eval on record").foregroundStyle(.secondary)
            }
        }
    }

    private var adaptationSection: some View {
        Section("Adaptation") {
            if let adaptation {
                LabeledContent("guard passed", value: adaptation.guardPassed ? "yes" : "no")
                LabeledContent("adapted", value: adaptation.adapted ? "yes" : "no")
                LabeledContent("when", value: adaptation.timestamp.formatted())
            } else {
                Text("No adaptation yet").foregroundStyle(.secondary)
            }
        }
    }

    private var teachersSection: some View {
        Section("Teachers") {
            if teachers.isEmpty {
                Text("No teacher assessments yet").foregroundStyle(.secondary)
            } else {
                ForEach(teachers) { teacher in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(teacher.teacherId).font(.headline)
                        LabeledContent("world-gate pass", value: String(format: "%.2f", teacher.worldGatePassFraction))
                        LabeledContent("samples", value: "\(teacher.samples)")
                        LabeledContent("effective weight", value: String(format: "%.2f", teacher.effectiveWeight))
                    }
                    .padding(.vertical, 2)
                }
            }
        }
    }

    private var foragingSection: some View {
        Section("Foraging") {
            LabeledContent("local-playbook", value: "\(foraging.localPlaybook)")
            LabeledContent("local-reference", value: "\(foraging.localReference)")
            LabeledContent("remote", value: "\(foraging.remote)")
            LabeledContent("total", value: "\(foraging.total)")
        }
    }

    private var curationSection: some View {
        Section("Curation") {
            LabeledContent("stored items", value: "\(curation.stored)")
            LabeledContent("dedup hits", value: "\(curation.dedupSkipped)")
            LabeledContent("preference pairs", value: "\(curation.preferencePairs)")
            LabeledContent("mean quality", value: String(format: "%.2f", curation.meanQuality))
        }
    }

    private func load() {
        refreshID = UUID()
        capability = TesseraCapabilityEvalStore().latest()
        adaptation = TesseraLearningCenter.shared.scheduler.lastAdaptation()
        teachers = TesseraLearningCenter.shared.assessor.assessments()
        foraging = TesseraLearningCenter.shared.foraging.summary()
        curation = TesseraLearningCenter.shared.curation.summary()
    }
}
