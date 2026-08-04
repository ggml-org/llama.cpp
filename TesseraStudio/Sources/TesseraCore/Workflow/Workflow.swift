import Foundation

/// The serialised form of a workflow. A workflow is a graph:
/// a set of nodes, a set of edges, and per-node parameters
/// (the values that don't flow on edges — runtime configuration
/// the user typed in the editor's side panel).
///
/// The ``schema`` field pins the JSON shape to a version. The
/// executor refuses to run a workflow whose schema it doesn't
/// recognise. New fields are additive (default values); new
/// node types are additive (unrecognised types produce a
/// validation error at load time, not at execute time).
public struct Workflow: Codable, Sendable, Equatable {
    public static let currentSchema = "tessera.workflow.v1"

    public let schema: String
    public let name: String
    public let nodes: [WorkflowNode]
    public let edges: [WorkflowEdge]

    public init(schema: String = Workflow.currentSchema,
                name: String,
                nodes: [WorkflowNode],
                edges: [WorkflowEdge]) {
        self.schema = schema
        self.name = name
        self.nodes = nodes
        self.edges = edges
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.schema = try c.decodeIfPresent(String.self, forKey: .schema) ?? Workflow.currentSchema
        self.name = try c.decode(String.self, forKey: .name)
        self.nodes = try c.decode([WorkflowNode].self, forKey: .nodes)
        self.edges = try c.decodeIfPresent([WorkflowEdge].self, forKey: .edges) ?? []
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(schema, forKey: .schema)
        try c.encode(name, forKey: .name)
        try c.encode(nodes, forKey: .nodes)
        try c.encode(edges, forKey: .edges)
    }

    private enum CodingKeys: String, CodingKey {
        case schema, name, nodes, edges
    }
}

/// One node in a workflow. The ``id`` is local to the workflow
/// and is referenced by edges; the ``type`` is the discriminator
/// in the ``WorkflowNodeRegistry``. ``parameters`` carries the
/// per-instance configuration (e.g. ``n_ctx = 4096``); the
/// default values come from ``WorkflowNodeType.parameterSchema``.
public struct WorkflowNode: Codable, Sendable, Equatable, Hashable {
    public let id: String
    public let type: String
    public let parameters: [String: JSONValue]

    public init(id: String, type: String, parameters: [String: JSONValue] = [:]) {
        self.id = id
        self.type = type
        self.parameters = parameters
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try c.decode(String.self, forKey: .id)
        self.type = try c.decode(String.self, forKey: .type)
        self.parameters = try c.decodeIfPresent([String: JSONValue].self, forKey: .parameters) ?? [:]
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(type, forKey: .type)
        try c.encode(parameters, forKey: .parameters)
    }

    private enum CodingKeys: String, CodingKey {
        case id, type, parameters
    }
}

/// One execution event surfaced from the workflow executor.
/// The editor subscribes to these for the live progress pane
/// (per-node status + log lines); headless runs surface them
/// to stderr.
public enum WorkflowEvent: Sendable, Equatable {
    case started(workflowName: String, totalNodes: Int)
    case nodeStarted(nodeId: String, typeId: String)
    case nodeFinished(nodeId: String, success: Bool, message: String?)
    case log(nodeId: String?, level: WorkflowLogLevel, message: String)
    case finished(success: Bool, message: String?)
}
