import Foundation

/// Composition root for the learning subsystem. Constructs the concrete
/// services and installs them into a TesseraLearningCenter, replacing the
/// no-op defaults. Wired at app launch by the orchestrator; nothing in this
/// module calls installDefaults itself.
public enum TesseraLearningServices {
    public static func installDefaults(into center: TesseraLearningCenter = .shared) {
        center.install(escalation: TesseraEscalationService())
        center.install(curation: TesseraCurationService())
        center.install(playbook: TesseraReasoningPlaybookStore())
        center.install(reference: TesseraReferenceKnowledgeStore())
        center.install(worldSignals: TesseraWorldSignalObserver())
        center.install(scheduler: TesseraAdaptationScheduler())
        center.install(assessor: TesseraTeacherAssessor())
    }
}
