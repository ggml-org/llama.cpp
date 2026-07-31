import Foundation

/// Composition root for the learning subsystem. Constructs the concrete
/// services and installs them into a TesseraLearningCenter, replacing the
/// no-op defaults. Wired at app launch by the orchestrator; nothing in this
/// module calls installDefaults itself.
public enum TesseraLearningServices {
    /// Held for the process lifetime so the recurring assessment loop is not
    /// deallocated; installDefaults owns the only reference.
    private static var assessmentScheduler: TesseraAssessmentScheduler?

    public static func installDefaults(into center: TesseraLearningCenter = .shared) {
        center.install(escalation: TesseraEscalationService())
        center.install(curation: TesseraCurationService())
        center.install(playbook: TesseraReasoningPlaybookStore())
        center.install(reference: TesseraReferenceKnowledgeStore())
        center.install(worldSignals: TesseraWorldSignalObserver())
        center.install(scheduler: TesseraAdaptationScheduler())
        center.install(assessor: TesseraTeacherAssessor())
        center.install(foraging: TesseraForagingStore())
        center.install(headRouting: TesseraTrackRScaffold())

        let scheduler = TesseraAssessmentScheduler(assessor: center.assessor)
        scheduler.start()
        assessmentScheduler = scheduler
    }
}
