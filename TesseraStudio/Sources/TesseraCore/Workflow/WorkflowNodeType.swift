import Foundation

/// The contract a workflow node type implements. A node is one
/// step in a workflow: it has typed inputs (which the executor
/// fills from upstream node outputs), a set of per-instance
/// parameters (serialised in the workflow JSON), and typed
/// outputs (which downstream nodes consume).
///
/// A node type is a Swift type that conforms to ``WorkflowNodeType``
/// and is registered in a ``WorkflowNodeRegistry``. The
/// protocol is intentionally close to ``TesseraTool`` (the
/// LLM-facing tool surface) so the existing 18 tools can be
/// wrapped as workflow nodes with little ceremony, but the
/// contract here is richer: the tool surface is one-shot
/// (LLM calls a tool, gets a result), the workflow surface is
/// dataflow (typed values flow on edges between nodes).
public protocol WorkflowNodeType: Sendable {
    /// The node type's stable id (e.g. ``"load_model"``). Used as
    /// the discriminator in the workflow JSON; renaming this
    /// value is a breaking change.
    static var typeId: String { get }

    /// Human-readable name shown in the editor.
    static var displayName: String { get }

    /// Short description of what the node does.
    static var summary: String { get }

    /// Typed input ports.
    static var inputs: [WorkflowPort] { get }

    /// Typed output ports.
    static var outputs: [WorkflowPort] { get }

    /// JSON Schema for the per-instance parameters that don't
    /// flow on edges (e.g. ``n_ctx``, ``awq_alpha``).
    static var parameterSchema: JSONSchema { get }

    /// Run the node. ``inputs`` is keyed by ``WorkflowPort.id``
    /// and contains one value for every declared input; the
    /// executor enforces that the type matches ``inputs[i].type``.
    /// The returned dictionary is keyed by ``WorkflowPort.id`` for
    /// the declared outputs; values must match ``outputs[i].type``.
    /// The context is the per-run environment (file system
    /// access, logging, cancellation). Throwing aborts the
    /// workflow with the error surfaced to the caller.
    static func execute(
        parameters: [String: JSONValue],
        inputs: [String: WorkflowPortValue],
        context: WorkflowExecutionContext
    ) async throws -> [String: WorkflowPortValue]
}

/// Convenience: default ``typeId`` from the type name (e.g.
/// ``LoadModelNode`` -> ``"load_model"``). Conformers that
/// prefer an explicit value can still provide one.
extension WorkflowNodeType {
    static var typeId: String {
        let name = String(describing: Self.self)
        // Drop the trailing "Node" suffix if present, then
        // snake_case the rest. This is the convention used by
        // every node shipped from Tessera Core; third-party
        // nodes are free to override.
        let stripped = name.hasSuffix("Node") ? String(name.dropLast(4)) : name
        return Self.snakeCase(stripped)
    }

    private static func snakeCase(_ value: String) -> String {
        var out = ""
        for (i, ch) in value.enumerated() {
            if ch.isUppercase {
                if i > 0 { out.append("_") }
                out.append(ch.lowercased())
            } else {
                out.append(ch)
            }
        }
        return out
    }
}

/// Per-run context passed to every node execution. The context
/// gives nodes access to the file system (via ``TesseraFileSystem``)
/// and a logging sink (so node progress is surfaced in the editor).
/// The context is ``Sendable``; nodes may run on any actor.
public struct WorkflowExecutionContext: Sendable {
    public let fileSystem: any TesseraFileSystem
    public let logger: any WorkflowLogger

    public init(fileSystem: any TesseraFileSystem = LocalTesseraFileSystem(),
                logger: any WorkflowLogger = SilentWorkflowLogger()) {
        self.fileSystem = fileSystem
        self.logger = logger
    }
}

/// Minimal file system abstraction the executor hands to nodes.
/// The default implementation is the real file system; tests
/// pass an in-memory stub.
public protocol TesseraFileSystem: Sendable {
    func fileExists(at path: String) -> Bool
    func readString(at path: String) async throws -> String
    func writeString(_ content: String, to path: String) async throws
    func expandPath(_ path: String) -> String
}

/// Local-disk file system. The default. ``expandPath`` resolves
/// ``~`` against the user's home; for the macOS app this is
/// ``NSString.expandingTildeInPath`` semantics.
public struct LocalTesseraFileSystem: TesseraFileSystem {
    public init() {}

    public func fileExists(at path: String) -> Bool {
        FileManager.default.fileExists(atPath: expandPath(path))
    }

    public func readString(at path: String) async throws -> String {
        try String(contentsOfFile: expandPath(path), encoding: .utf8)
    }

    public func writeString(_ content: String, to path: String) async throws {
        let expanded = expandPath(path)
        try content.write(toFile: expanded, atomically: true, encoding: .utf8)
    }

    public func expandPath(_ path: String) -> String {
        NSString(string: path).expandingTildeInPath
    }
}

/// Logger surface nodes call into for progress + diagnostics.
/// The executor routes log lines to the editor's progress pane
/// (or, in headless mode, to stderr).
public protocol WorkflowLogger: Sendable {
    func log(_ message: String, level: WorkflowLogLevel)
}

public enum WorkflowLogLevel: String, Codable, Sendable {
    case debug, info, warn, error
}

/// Logger that drops everything. Use for batch / headless runs
/// where progress chatter is noise.
public struct SilentWorkflowLogger: WorkflowLogger {
    public init() {}
    public func log(_ message: String, level: WorkflowLogLevel) {}
}

/// Logger that writes to stderr. Default for the workflow CLI.
public struct StderrWorkflowLogger: WorkflowLogger {
    public init() {}
    public func log(_ message: String, level: WorkflowLogLevel) {
        let prefix: String
        switch level {
        case .debug: prefix = "DEBUG"
        case .info:  prefix = "INFO"
        case .warn:  prefix = "WARN"
        case .error: prefix = "ERROR"
        }
        let ts = ISO8601DateFormatter().string(from: Date())
        FileHandle.standardError.write(Data("[workflow \(ts) \(prefix)] \(message)\n".utf8))
    }
}
