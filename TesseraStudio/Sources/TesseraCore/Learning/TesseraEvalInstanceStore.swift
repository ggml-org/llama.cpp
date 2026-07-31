import Foundation

/// File-backed store for the held-out capability-eval instance set, keyed by
/// axis (design 4.7 / 8). The instance set is the ruler every candidate is
/// judged against; it is curated, not harvested, and is deletable via purge.
public final class TesseraEvalInstanceStore: TesseraPurgeable, @unchecked Sendable {
    private let store: TesseraLearningStore
    private let lock = NSLock()
    private static let file = "eval-instances.json"

    public init() {
        self.store = TesseraLearningStore()
    }

    /// Install a small built-in seed set if the store is empty, so every axis
    /// is represented from day 1. SEEDS ONLY: these are compact stand-ins to
    /// be replaced by a real curated held-out set before any serious run.
    public func seedDefaultsIfNeeded() {
        lock.lock(); defer { lock.unlock() }
        let existing = loadLocked()
        guard existing.isEmpty else { return }
        try? store.save(Self.seedInstances, to: Self.file)
    }

    public func instances(forAxis axis: String) -> [TesseraEvalInstance] {
        lock.lock(); defer { lock.unlock() }
        return loadLocked().filter { $0.axis == axis }
    }

    public func allInstances() -> [TesseraEvalInstance] {
        lock.lock(); defer { lock.unlock() }
        return loadLocked()
    }

    public func add(_ instance: TesseraEvalInstance) throws {
        lock.lock(); defer { lock.unlock() }
        var instances = loadLocked()
        instances.append(instance)
        try store.save(instances, to: Self.file)
    }

    public func purgeTrainingData() throws -> Int {
        lock.lock(); defer { lock.unlock() }
        let count = loadLocked().count
        try store.delete(Self.file)
        return count
    }

    // Caller must hold `lock`.
    private func loadLocked() -> [TesseraEvalInstance] {
        store.load([TesseraEvalInstance].self, from: Self.file, default: [])
    }

    // Compact seed set. mechanical carries the first proof (red -> green,
    // binary reward); the other axes get 1-2 placeholders each so the vector
    // is fully populated. Replace with a real curated held-out set.
    private static let seedInstances: [TesseraEvalInstance] = [
        // mechanical: failing-test resolution, binary reward
        TesseraEvalInstance(
            id: "seed-mechanical-1",
            axis: "mechanical",
            prompt: "The test expects sum(2, 3) == 5 but sum currently subtracts and returns 4. Fix sum so the test passes.",
            expectedSignal: "test target exits 0; assertion passes (red -> green)"
        ),
        TesseraEvalInstance(
            id: "seed-mechanical-2",
            axis: "mechanical",
            prompt: "testReverse fails: reverse(\"abc\") returns \"abc\" unchanged. Fix reverse so it returns \"cba\".",
            expectedSignal: "test passes (red -> green)"
        ),
        TesseraEvalInstance(
            id: "seed-mechanical-3",
            axis: "mechanical",
            prompt: "testMax fails: max(1, 2) returns 1. Correct the comparison so the larger value is returned.",
            expectedSignal: "test passes (red -> green)"
        ),
        TesseraEvalInstance(
            id: "seed-mechanical-4",
            axis: "mechanical",
            prompt: "Build fails with 'cannot convert value of type Int to expected argument type String' in greet(). Fix the type mismatch.",
            expectedSignal: "build succeeds and the test passes"
        ),
        // apiCurrency: deprecated-API migration
        TesseraEvalInstance(
            id: "seed-api-1",
            axis: "apiCurrency",
            prompt: "Migrate the deprecated UIColor(red:green:blue:alpha:) call in ProfileView to the current Color API.",
            expectedSignal: "no deprecation warnings; builds clean against the current SDK"
        ),
        TesseraEvalInstance(
            id: "seed-api-2",
            axis: "apiCurrency",
            prompt: "Replace the removed String(contentsOf:) overload with the current throwing initializer.",
            expectedSignal: "compiles against the current SDK"
        ),
        // hardTail: escalation-class reasoning
        TesseraEvalInstance(
            id: "seed-hardtail-1",
            axis: "hardTail",
            prompt: "TSan reports a data race in the shared cache read path. Diagnose it and propose a fix.",
            expectedSignal: "TSan-clean run"
        ),
        // personalStyle: trunk / personal-distribution fit
        TesseraEvalInstance(
            id: "seed-style-1",
            axis: "personalStyle",
            prompt: "Refactor this function to match the repo convention of early returns over nested conditionals.",
            expectedSignal: "matches trunk style; reviewer accepts"
        ),
        // generalCompetence: broad held-out guard
        TesseraEvalInstance(
            id: "seed-general-1",
            axis: "generalCompetence",
            prompt: "Find and fix the off-by-one error in this binary search over a sorted array.",
            expectedSignal: "correct on boundary cases (empty, single element, first, last)"
        ),
        TesseraEvalInstance(
            id: "seed-general-2",
            axis: "generalCompetence",
            prompt: "State the time complexity of the given merge-sort routine and justify it.",
            expectedSignal: "correct O(n log n) answer with a sound justification"
        ),
    ]
}
