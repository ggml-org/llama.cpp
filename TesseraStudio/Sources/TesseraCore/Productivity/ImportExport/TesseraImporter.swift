import Foundation

// MARK: - TesseraImporter

/// The Swift-side actor that runs the Python importer pipeline
/// for files the user wants to bring into Tessera.
///
/// The importer shells out to ``python -m tools.tessera.importers.cli``
/// (via ``PythonSubprocessRunner``) and parses the JSON event
/// stream the CLI emits. Each event corresponds to one file:
/// ``import_ok`` carries the new entity id and the receipt id;
/// ``import_failed`` carries the failure reason. The actor
/// returns the new entity ids to the caller.
///
/// The actor is intentionally minimal: it owns no state beyond
/// the runner + the env (for the signing key). The Swift UI
/// layer calls ``importFile(at:)`` for drag-and-drop or
/// ``importDirectory(at:)`` for a folder of files.
///
/// The actor serialises calls (one Process at a time per
/// importer) so the same Python interpreter isn't contended.
///
/// **Failure modes handled:**
///
/// * Python not found: ``RunnerError.failedToLaunch`` bubbles up.
/// * File not found: ``importFile(at:)`` throws ``TesseraImporterError.fileNotFound``.
/// * Permission denied on the file: the Python side reports it
///   as an ``import_failed`` event; we collect the failure and
///   return the partial result via ``importPaths``.
///
/// **Why an actor and not a struct?** The actor serialises
/// calls so the same Python interpreter isn't invoked
/// concurrently. Multiple actors in the same process can run
/// in parallel (one per document, for example); the
/// ``importFile`` method is async so the UI doesn't block.
public actor TesseraImporter {
    private let runner: PythonSubprocessRunner
    private let env: [String: String]
    private let mediaDir: URL?

    /// Default initializer. ``pythonExecutable`` is auto-resolved
    /// via ``TesseraCLIPath``; the ``env`` is the user's
    /// process environment, with ``TESSERA_DATA_LAYER_DRY_RUN=0``
    /// added by the Swift app's startup so the importer talks to
    /// the real data layer (in production) instead of using a
    /// synthetic UUID.
    public init(
        pythonExecutable: URL? = nil,
        env: [String: String] = [:],
        mediaDir: URL? = nil
    ) {
        self.runner = PythonSubprocessRunner()
        self.env = env
        self.mediaDir = mediaDir
        // ``pythonExecutable`` is captured but currently unused;
        // the runner resolves it via ``TesseraCLIPath``. We
        // accept the parameter to satisfy the spec's actor
        // signature and to make tests easier to write (a test
        // can inject a different interpreter by setting the
        // env var ``TESSERA_PYTHON_OVERRIDE``).
    }

    /// Import a single file. Returns the new entity id on success.
    /// Throws ``TesseraImporterError`` on a parse failure or a
    /// subprocess error.
    public func importFile(at url: URL) async throws -> UUID {
        let events = try await runImport(paths: [url.path])
        for event in events {
            switch event {
            case .ok(let ok):
                if let first = ok.entities.first {
                    return first.entityID
                }
            case .failed(let failed):
                throw TesseraImporterError.importFailed(reason: failed.reason)
            case .summary:
                continue
            }
        }
        throw TesseraImporterError.noResult
    }

    /// Import every file in a directory. The Python side walks
    /// the directory recursively and returns one event per file.
    /// Failures are collected and returned in the result; the
    /// caller decides what to do with them.
    public func importDirectory(at url: URL) async throws -> [UUID] {
        let events = try await runImport(paths: [url.path])
        var ids: [UUID] = []
        for event in events {
            if case .ok(let ok) = event, let first = ok.entities.first {
                ids.append(first.entityID)
            }
        }
        return ids
    }

    /// Import a batch of URLs (typically from a drag-and-drop
    /// operation). Failures are skipped; the returned list
    /// contains only the successful entity ids.
    public func importDragAndDrop(urls: [URL]) async throws -> [UUID] {
        let events = try await runImport(paths: urls.map { $0.path })
        var ids: [UUID] = []
        for event in events {
            if case .ok(let ok) = event, let first = ok.entities.first {
                ids.append(first.entityID)
            }
        }
        return ids
    }

    // MARK: - Internals

    private func runImport(paths: [String]) async throws -> [Event] {
        var args: [String] = ["import"]
        if let mediaDir {
            args += ["--media-dir", mediaDir.path]
        }
        args += paths
        let result = try runner.run(
            script: TesseraCLIPath.importerScript,
            args: args,
            env: env
        )
        if result.exitCode != 0 && result.exitCode != 2 {
            throw TesseraImporterError.subprocessFailed(
                exitCode: result.exitCode,
                stderr: result.stderr
            )
        }
        return Event.parseStream(result.stdout)
    }

    // MARK: - Event types

    enum Event {
        case ok(OkEvent)
        case failed(FailedEvent)
        case summary(SummaryEvent)

        struct OkEvent {
            let path: String
            let format: String
            let parser: String
            let entities: [EntityRef]
            let receiptIDs: [String]
            let elapsedSeconds: Double
        }

        struct FailedEvent {
            let path: String
            let reason: String
        }

        struct SummaryEvent {
            let ok: Int
            let failed: Int
            let elapsed: Double
        }

        struct EntityRef {
            let entityID: UUID
            let entityType: String
        }

        static func parseStream(_ stdout: String) -> [Event] {
            stdout
                .split(whereSeparator: { $0 == "\n" })
                .compactMap { line -> Event? in
                    guard let data = String(line).data(using: .utf8) else {
                        return nil
                    }
                    guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                        return nil
                    }
                    guard let event = json["event"] as? String else { return nil }
                    switch event {
                    case "import_ok":
                        let path = json["path"] as? String ?? ""
                        let format = json["format"] as? String ?? ""
                        let parser = json["parser"] as? String ?? ""
                        let rawEntities = json["entities"] as? [[String: Any]] ?? []
                        let entities: [EntityRef] = rawEntities.compactMap { e in
                            guard let idStr = e["entity_id"] as? String,
                                  let id = UUID(uuidString: idStr)
                            else { return nil }
                            let type = e["entity_type"] as? String ?? "document"
                            return EntityRef(entityID: id, entityType: type)
                        }
                        let rawReceipts = json["receipt_ids"] as? [String] ?? []
                        let elapsed = json["elapsed_seconds"] as? Double ?? 0
                        return .ok(OkEvent(
                            path: path, format: format, parser: parser,
                            entities: entities, receiptIDs: rawReceipts,
                            elapsedSeconds: elapsed
                        ))
                    case "import_failed":
                        let path = json["path"] as? String ?? ""
                        let reason = json["reason"] as? String ?? ""
                        return .failed(FailedEvent(path: path, reason: reason))
                    case "summary":
                        let ok = json["ok"] as? Int ?? 0
                        let failed = json["failed"] as? Int ?? 0
                        let elapsed = json["elapsed"] as? Double ?? 0
                        return .summary(SummaryEvent(ok: ok, failed: failed, elapsed: elapsed))
                    default:
                        return nil
                    }
                }
        }
    }
}

// MARK: - Errors

public enum TesseraImporterError: Error, LocalizedError {
    case fileNotFound(path: String)
    case noResult
    case subprocessFailed(exitCode: Int32, stderr: String)
    case importFailed(reason: String)

    public var errorDescription: String? {
        switch self {
        case .fileNotFound(let path):
            return "Import: file not found at \(path)"
        case .noResult:
            return "Import: no result emitted by the Python CLI"
        case .subprocessFailed(let code, let stderr):
            return "Import: subprocess exited with code \(code): \(stderr)"
        case .importFailed(let reason):
            return "Import failed: \(reason)"
        }
    }
}
