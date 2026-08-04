import Foundation

/// The type system for workflow node ports.
///
/// A workflow is a directed graph of nodes; edges connect output
/// ports to input ports. The runtime enforces that an edge is
/// only legal when the source port type and the target port type
/// are equal (or one is a subtype of the other via
/// ``canFlowInto``). Types are the source of truth for both the
/// executor (which port value to pass downstream) and the
/// SwiftUI graph editor (which edges to allow in the UI).
///
/// This is a closed enum on purpose: node types are defined at
/// compile time, and we want the editor and the runtime to agree
/// without negotiation. The string raw value is the canonical
/// type name in the workflow JSON; clients compare via the
/// ``rawValue`` for stability across Swift versions.
public enum WorkflowPortType: String, Codable, Sendable, CaseIterable, Equatable {
    /// A free-text string.
    case string
    /// A 64-bit floating-point number.
    case number
    /// A boolean.
    case boolean
    /// A file system path (absolute, with `~` expanded by the
    /// executor before being passed to the node).
    case path
    /// A path to a GGUF file. Treated as ``path`` at the wire
    /// level; the type is preserved in the JSON for the editor
    /// to render a "GGUF" badge on the port.
    case gguf
    /// A path to a JSON file (e.g. a calibration policy, a
    /// sidecar). Same wire handling as ``gguf``; separate type
    /// for the editor.
    case json
    /// A structured tool result (``ToolResult``-shaped). Carried
    /// as a JSON object in the workflow runtime; the executor
    /// materializes it as ``[String: JSONValue]`` for the node.
    case toolResult = "tool_result"
    /// A free-form key/value bag. The executor passes it through
    /// as ``[String: JSONValue]``; the node decides which keys
    /// it cares about. Use this when the upstream output shape
    /// is heterogeneous (the LLM's "evaluation result" has
    /// several fields and downstream nodes only need a subset).
    case bag

    /// Whether a value of ``self`` can flow into a port of type
    /// ``target``. Strict equality is the default; the only
    /// allowed widening is ``path -> gguf`` / ``path -> json``
    /// (a generic path can be specialised at the receiving port,
    /// not the other way).
    public func canFlowInto(_ target: WorkflowPortType) -> Bool {
        if self == target { return true }
        if self == .path && (target == .gguf || target == .json) {
            return true
        }
        return false
    }
}

/// A typed value carried on a workflow edge. The runtime holds
/// these in memory between node executions; they are not
/// serialised to the workflow JSON (the JSON carries only the
/// graph topology + per-node parameters; values are produced at
/// execution time).
public enum WorkflowPortValue: Sendable, Equatable {
    case string(String)
    case number(Double)
    case boolean(Bool)
    case path(String)
    case toolResult([String: JSONValue])
    case bag([String: JSONValue])

    public init?(type: WorkflowPortType, raw: JSONValue) {
        switch (type, raw) {
        case (.string, .string(let v)): self = .string(v)
        case (.number, .number(let v)): self = .number(v)
        case (.boolean, .bool(let v)): self = .boolean(v)
        case (.path, .string(let v)): self = .path(v)
        case (.gguf, .string(let v)): self = .path(v)
        case (.json, .string(let v)): self = .path(v)
        case (.toolResult, .object(let v)): self = .toolResult(v)
        case (.bag, .object(let v)): self = .bag(v)
        case (.bag, .string(let v)): self = .bag(["value": .string(v)])
        default: return nil
        }
    }

    public var asJSONValue: JSONValue {
        switch self {
        case .string(let v): return .string(v)
        case .number(let v): return .number(v)
        case .boolean(let v): return .bool(v)
        case .path(let v): return .string(v)
        case .toolResult(let v): return .object(v)
        case .bag(let v): return .object(v)
        }
    }

    public var shortDescription: String {
        switch self {
        case .string(let v): return v.count > 80 ? String(v.prefix(77)) + "..." : v
        case .number(let v): return String(format: "%g", v)
        case .boolean(let v): return v ? "true" : "false"
        case .path(let v): return v
        case .toolResult(let v): return "{tool_result \(v.count) keys}"
        case .bag(let v): return "{bag \(v.count) keys}"
        }
    }
}

/// A single input or output port on a workflow node. The ``id``
/// is local to the node; edges reference ports by ``nodeId.portId``.
public struct WorkflowPort: Codable, Sendable, Equatable, Hashable {
    public let id: String
    public let label: String
    public let type: WorkflowPortType
    public let description: String?

    public init(id: String, label: String, type: WorkflowPortType, description: String? = nil) {
        self.id = id
        self.label = label
        self.type = type
        self.description = description
    }
}
