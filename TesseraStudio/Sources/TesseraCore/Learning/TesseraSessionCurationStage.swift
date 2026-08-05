import Foundation

/// Outcome of one curation sweep (runtime-traces spec section 12.5). Every
/// counter is honest: a degraded sweep reports what it did NOT do via note.
public struct TesseraSessionCurationReport: Sendable, Equatable {
    public var sessionsSeen = 0
    public var analyzed = 0
    public var promoted = 0
    public var quarantined = 0
    public var dropped = 0
    public var replayedSessions = 0
    public var replayRecords = 0
    /// Sids promoted but not yet replayed (replay degrades open and retries
    /// on the next sweep).
    public var pendingReplay: [String] = []
    /// Why the sweep stopped short, when it did.
    public var note: String?

    public init() {}
}

public enum TesseraSessionCurationError: LocalizedError, Equatable {
    case replayUnavailable(String)

    public var errorDescription: String? {
        switch self {
        case .replayUnavailable(let message): return message
        }
    }
}

/// Resumable curation state persisted at
/// <learningStoreDir>/session-curation-state.json: the dedup fingerprint set
/// plus promoted sids whose replay has not landed yet.
struct TesseraSessionCurationStateFile: Codable, Equatable {
    var fingerprints: [String] = []
    var pendingReplaySids: [String] = []
}

/// The session replay: analysis and curation stage (runtime-traces spec
/// section 12). Order per sweep: read runtime sessions -> analyze the
/// uncurated ones -> append verdicts to the ledger -> replay promoted
/// sessions through the existing imatrix calibration loop at topk 64 and
/// append the replay-provenance records. Every dependency degrades open:
/// missing trunk/drafter/imatrix means the sweep reports and retries later,
/// never judges blind, and never loses a verdict.
public final class TesseraSessionCurationStage: @unchecked Sendable {
    public static let replayTopkDefault = 64
    public static let stateFileName = "session-curation-state.json"

    /// The replay driver turns a decoded session corpus into raw imatrix
    /// telemetry lines (unstamped). Tests inject a fake; the default drives
    /// llama-imatrix (zero new native binaries, spec section 12.2).
    public typealias ReplayDriver = (_ corpus: String, _ topk: Int) async throws -> [String]

    private let store: TesseraTraceStore
    private let ledger: TesseraCurationLedger
    private let decoderProvider: (String) -> (any TesseraSessionDecoder)?
    private let injectedReplayDriver: ReplayDriver?
    private let replayTopk: Int
    private let trunkPathProvider: () -> String
    private let lock = NSLock()

    public init(
        store: TesseraTraceStore = TesseraTraceStore(),
        ledger: TesseraCurationLedger? = nil,
        decoderProvider: @escaping (String) -> (any TesseraSessionDecoder)? = { TesseraVocabDecoder.open(modelPath: $0) },
        replayDriver: ReplayDriver? = nil,
        replayTopk: Int = TesseraSessionCurationStage.replayTopkDefault,
        trunkPathProvider: @escaping () -> String = { TesseraSettings.learningBaseModelPath }
    ) {
        self.store = store
        self.ledger = ledger
            ?? TesseraCurationLedger(directory: store.directoryURL.deletingLastPathComponent())
        self.decoderProvider = decoderProvider
        self.injectedReplayDriver = replayDriver
        self.replayTopk = replayTopk
        self.trunkPathProvider = trunkPathProvider
    }

    /// One idle curation sweep. Resumable: sessions already in the ledger
    /// are skipped, promoted-but-unreplayed sids carry over, and the
    /// fingerprint set persists across launches.
    public func sweep() async -> TesseraSessionCurationReport {
        var report = TesseraSessionCurationReport()

        let sessions = TesseraSessionTraceReader.sessions(in: store)
        report.sessionsSeen = sessions.count
        let sessionsBySid = Dictionary(uniqueKeysWithValues: sessions.map { ($0.sid, $0) })

        var state = loadState()
        var fingerprints = Set(state.fingerprints)
        let known = ledger.latestVerdicts()
        let uncurated = sessions.filter { known[$0.sid] == nil }

        // Pending replay sids whose runtime files were trimmed before the
        // replay landed cannot be decoded anymore; drop them honestly.
        var pendingReplay = state.pendingReplaySids.filter { sessionsBySid[$0] != nil }
        let lostReplay = state.pendingReplaySids.count - pendingReplay.count
        if lostReplay > 0 {
            report.note = "\(lostReplay) promoted session(s) trimmed before replay"
        }

        report.pendingReplay = pendingReplay
        if uncurated.isEmpty && pendingReplay.isEmpty {
            if report.note == nil { report.note = "no uncurated sessions" }
            return report
        }

        // Compatibility + decode need the current trunk's vocab. Degrade
        // open: judge nothing, keep every session for the next sweep.
        let trunkPath = trunkPathProvider()
        guard !trunkPath.isEmpty else {
            report.note = "no trunk model configured; curation deferred"
            return report
        }
        guard let decoder = decoderProvider(trunkPath) else {
            report.note = "trunk vocab unavailable; curation deferred"
            return report
        }
        defer { (decoder as? TesseraVocabDecoder)?.close() }

        var decodedTexts: [String: String] = [:]
        var newlyPromoted: [String] = []

        for session in uncurated {
            let tokens = session.acceptedTokens
            let compatible = !tokens.isEmpty
                && tokens.allSatisfy { $0 >= 0 && $0 < decoder.nVocab }

            var text = ""
            var pieces: [String] = []
            if compatible {
                text = decoder.detokenize(tokens) ?? ""
                pieces = tokens.map { decoder.piece(for: $0) ?? "" }
            }

            let analysis = TesseraSessionAnalysis(
                tokenCount: tokens.count,
                acceptanceRate: TesseraSessionScorecard.acceptanceRate(steps: session.steps.map {
                    TesseraSessionStepStats(drafted: $0.drafted, accepted: $0.accepted)
                }),
                meanAcceptedRun: TesseraSessionScorecard.meanAcceptedRun(steps: session.steps.map {
                    TesseraSessionStepStats(drafted: $0.drafted, accepted: $0.accepted)
                }),
                repetitionRatio: TesseraSessionScorecard.repetitionRatio(of: text),
                garbageRatio: TesseraSessionScorecard.garbageRatio(pieces: pieces),
                probeHits: TesseraScrubRules.probe(text),
                fingerprint: TesseraSessionScorecard.fingerprint(of: text),
                modelCompatible: compatible)

            let isDuplicate = compatible && fingerprints.contains(analysis.fingerprint)
            let judgement = TesseraSessionScorecard.judge(analysis, isDuplicate: isDuplicate)

            do {
                try ledger.append(TesseraCurationLedgerEntry(
                    sid: session.sid,
                    verdict: judgement.verdict,
                    reasons: judgement.reasons,
                    score: TesseraCurationLedgerEntry.Score(
                        acceptance: analysis.acceptanceRate,
                        tokens: analysis.tokenCount,
                        repetition: analysis.repetitionRatio)))
            } catch {
                report.note = "ledger append failed: \(error.localizedDescription)"
                break
            }

            report.analyzed += 1
            if compatible { fingerprints.insert(analysis.fingerprint) }
            switch judgement.verdict {
            case .promoted:
                report.promoted += 1
                newlyPromoted.append(session.sid)
                decodedTexts[session.sid] = text
            case .quarantined: report.quarantined += 1
            case .dropped: report.dropped += 1
            case .purged: break // only user-initiated purge emits this
            }
        }

        pendingReplay.append(contentsOf: newlyPromoted)
        state.fingerprints = Array(fingerprints)
        state.pendingReplaySids = pendingReplay
        saveState(state)
        report.pendingReplay = pendingReplay

        // Replay step: promoted sessions become an imatrix corpus, re-run at
        // deepened topk (spec section 12.2). Failure keeps the sids pending.
        if !pendingReplay.isEmpty {
            var texts: [String] = []
            for sid in pendingReplay {
                if let cached = decodedTexts[sid] {
                    texts.append(cached)
                } else if let session = sessionsBySid[sid],
                          let text = decoder.detokenize(session.acceptedTokens) {
                    texts.append(text)
                }
            }
            let corpus = texts.joined(separator: "\n\n")
            let driver = injectedReplayDriver ?? defaultReplayDriver(trunkPath: trunkPath)
            do {
                let lines = try await driver(corpus, replayTopk)
                let stamped = lines.compactMap { Self.stampReplayLine($0) }
                try store.appendReplay(
                    records: stamped, exemptSids: ledger.quarantinedSids())
                report.replayedSessions = pendingReplay.count
                report.replayRecords = stamped.count
                state.pendingReplaySids = []
                saveState(state)
                report.pendingReplay = []
            } catch {
                let suffix = "replay deferred: \(error.localizedDescription)"
                report.note = report.note.map { "\($0); \(suffix)" } ?? suffix
            }
        }

        return report
    }

    /// Stamp one imatrix calibration record with the replay provenance.
    /// Calibration emits no provenance/sid fields (spec section 12.2: the
    /// sid is stripped at promotion), so the fields are appended before the
    /// closing brace. Idempotent: a line already stamped as replay passes
    /// through byte-identical. A line carrying any other provenance is
    /// dropped (the egress guard only accepts the exact promotion stamp).
    /// Lines that are not JSON objects are dropped.
    public static func stampReplayLine(_ line: String) -> String? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasSuffix("}"),
              let data = trimmed.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data),
              let dict = obj as? [String: Any] else { return nil }
        if let provenance = dict["provenance"] as? String {
            return provenance == "replay" ? trimmed : nil
        }
        return String(trimmed.dropLast())
            + ",\"provenance\":\"replay\",\"replayed_from\":\"runtime\"}"
    }

    // MARK: - Default replay driver (llama-imatrix over the decoded corpus)

    private func defaultReplayDriver(trunkPath: String) -> ReplayDriver {
        return { corpus, topk in
            let imatrix = TesseraTrainBinaryResolver.resolveImatrix(
                trainOverride: TesseraSettings.learningTrainBinary)
            guard FileManager.default.isExecutableFile(atPath: imatrix) else {
                throw TesseraSessionCurationError.replayUnavailable(
                    CollectTrainingTracesTool.missingImatrixNote(path: imatrix))
            }
            guard let drafter = TesseraRuntimeDrafterResolver.resolvedDrafter(
                setting: TesseraSettings.learningRuntimeDraftModel, trunkPath: trunkPath) else {
                throw TesseraSessionCurationError.replayUnavailable("no runtime drafter resolved for replay")
            }

            let tmp = NSTemporaryDirectory() as NSString
            let corpusPath = tmp.appendingPathComponent("tessera-replay-corpus-\(UUID().uuidString).txt")
            let telemetryPath = tmp.appendingPathComponent("tessera-replay-\(UUID().uuidString).jsonl")
            let imatrixOut = tmp.appendingPathComponent("tessera-replay-imatrix-\(UUID().uuidString).gguf")
            try corpus.write(toFile: corpusPath, atomically: true, encoding: .utf8)

            let result = try await ProcessRunner().run(
                executable: imatrix,
                arguments: [
                    "-m", NSString(string: trunkPath).expandingTildeInPath,
                    "--model-draft", NSString(string: drafter).expandingTildeInPath,
                    "-f", corpusPath,
                    "-c", String(Self.replayContextSize(for: corpus)),
                    "-o", imatrixOut,
                    "--telemetry-out", telemetryPath,
                    "--telemetry-topk", String(topk),
                ]
            )
            // The imatrix file is a byproduct of replay; drop it either way.
            try? FileManager.default.removeItem(atPath: imatrixOut)
            try? FileManager.default.removeItem(atPath: corpusPath)

            guard result.exitCode == 0 else {
                try? FileManager.default.removeItem(atPath: telemetryPath)
                throw TesseraSessionCurationError.replayUnavailable(
                    "llama-imatrix exited \(result.exitCode): \(result.stderr)")
            }

            var lines: [String] = []
            if let text = try? String(contentsOfFile: telemetryPath, encoding: .utf8) {
                text.enumerateLines { line, _ in
                    if !line.trimmingCharacters(in: .whitespaces).isEmpty { lines.append(line) }
                }
            }
            try? FileManager.default.removeItem(atPath: telemetryPath)
            return lines
        }
    }

    /// Context size for the replay run. Calibration needs a corpus of at
    /// least n_ctx + 4 tokens, so size the context from a conservative
    /// chars-per-token estimate of the corpus and leave headroom.
    static func replayContextSize(for corpus: String) -> Int {
        let estimate = corpus.utf8.count / 5
        return max(32, min(4096, estimate))
    }

    // MARK: - State file

    private var stateURL: URL {
        store.directoryURL
            .deletingLastPathComponent()
            .appendingPathComponent(Self.stateFileName)
    }

    private func loadState() -> TesseraSessionCurationStateFile {
        lock.lock(); defer { lock.unlock() }
        guard let data = try? Data(contentsOf: stateURL) else { return TesseraSessionCurationStateFile() }
        return (try? JSONDecoder().decode(TesseraSessionCurationStateFile.self, from: data))
            ?? TesseraSessionCurationStateFile()
    }

    private func saveState(_ state: TesseraSessionCurationStateFile) {
        lock.lock(); defer { lock.unlock() }
        guard let data = try? JSONEncoder().encode(state) else { return }
        try? FileManager.default.createDirectory(
            at: stateURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: stateURL, options: .atomic)
    }
}
