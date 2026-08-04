import Foundation

/// A registry of workflow node types. The registry is the
/// runtime's catalogue: when the executor encounters a node
/// in a workflow JSON, it looks up the ``WorkflowNodeType`` by
/// ``typeId`` and dispatches to it. The registry is also the
/// surface the editor uses to render the node palette (every
/// registered type becomes a draggable node in the UI).
///
/// Registries are ``Sendable``; the executor and the editor
/// share one instance per process. The default registry
/// (``WorkflowNodeRegistry.default``) bundles every wrapped
/// node shipped from Tessera Core; tests build their own
/// minimal registry.
public final class WorkflowNodeRegistry: @unchecked Sendable {
    private let types: [String: any WorkflowNodeType.Type]

    public init(types: [any WorkflowNodeType.Type]) {
        var map: [String: any WorkflowNodeType.Type] = [:]
        for type in types {
            // Protocol-metatype dispatch: ``typeId`` is a static
            // property of the metatype. We use a do/catch-free
            // pattern because ``typeId`` is required.
            let id = type.typeId
            if map[id] != nil {
                preconditionFailure("duplicate node typeId: \(id)")
            }
            map[id] = type
        }
        self.types = map
    }

    public func nodeType(for typeId: String) -> (any WorkflowNodeType.Type)? {
        types[typeId]
    }

    public var allTypeIds: [String] {
        types.keys.sorted()
    }

    public var allNodeTypes: [any WorkflowNodeType.Type] {
        types.values.sorted { $0.typeId < $1.typeId }
    }

    /// Build a workflow node palette entry for the editor. The
    /// entry has the human-readable display name, the
    /// declared ports, and the parameter schema; the editor
    /// uses these to render the draggable node + its side panel.
    public func paletteEntry(for typeId: String) -> WorkflowNodePaletteEntry? {
        guard let type = types[typeId] else { return nil }
        return WorkflowNodePaletteEntry(
            typeId: type.typeId,
            displayName: type.displayName,
            summary: type.summary,
            inputs: type.inputs,
            outputs: type.outputs,
            parameterSchema: type.parameterSchema,
        )
    }
}

/// A snapshot of a registered node type, suitable for the
/// editor's palette. The palette is rebuilt from the registry
/// at editor open time; node types that aren't registered are
/// not draggable.
public struct WorkflowNodePaletteEntry: Sendable, Equatable, Hashable {
    public let typeId: String
    public let displayName: String
    public let summary: String
    public let inputs: [WorkflowPort]
    public let outputs: [WorkflowPort]
    public let parameterSchema: JSONSchema
}

/// Static descriptor for an edge in a workflow. Edges are
/// (sourceNodeId, sourcePortId) -> (targetNodeId, targetPortId)
/// with the executor enforcing type compatibility. The
/// executor also rejects self-loops and parallel edges
/// (two edges connecting the same pair of ports).
public struct WorkflowEdge: Codable, Sendable, Equatable, Hashable {
    public let fromNode: String
    public let fromPort: String
    public let toNode: String
    public let toPort: String

    public init(fromNode: String, fromPort: String, toNode: String, toPort: String) {
        self.fromNode = fromNode
        self.fromPort = fromPort
        self.toNode = toNode
        self.toPort = toPort
    }
}
