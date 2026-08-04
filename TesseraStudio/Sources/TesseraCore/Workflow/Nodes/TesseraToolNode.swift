import Foundation

/// Helpers for wrapping a `TesseraTool` as a `WorkflowNodeType`.
///
/// The protocol is `static` everywhere (the executor dispatches on
/// the metatype, not on an instance), so each wrapped tool is a
/// distinct Swift type. The pattern is:
///   1. Each node (e.g. `LoadModelNode`) is a zero-state struct
///      that holds a static reference to the underlying tool.
///   2. `TesseraToolNode` itself is not a type — it's a namespace
///      with the schema-splitting logic (`splitSchema`,
///      `portType`, `humanName`) and the canonical `execute` body.
///   3. Adding a new wrapped tool is ~20 lines: declare the
///      struct, supply the static `typeId` / `displayName` / etc.,
///      and forward `execute` to the helper.
///
/// Splitting rule:
///   - Each `required` schema property becomes an input port
///     (typed from the schema's `type` field; `_path` suffix
///     maps to `WorkflowPortType.path`).
///   - Each optional property (has a `defaultValue`, or not in
///     `required`) becomes a node-level parameter edited in the
///     side panel, not wired.
///   - The node has one synthetic output port `result` typed
///     `toolResult`, carrying the `ToolResult.data` map (or a
///     synthesised payload if `data` is nil).
public enum TesseraToolNode {
    /// Split a tool's `JSONSchema` into (input ports, parameter
    /// schema). Required properties become ports; optional ones
    /// stay in the parameter schema for the editor's side panel.
    public static func splitSchema(_ schema: JSONSchema)
        -> ([WorkflowPort], JSONSchema)
    {
        let required = Set(schema.required ?? [])
        let allProperties = schema.properties ?? [:]
        let ports: [WorkflowPort] = allProperties
            .filter { required.contains($0.key) }
            .sorted { $0.key < $1.key }
            .map { (key, prop) in
                WorkflowPort(
                    id: key,
                    label: humanLabel(key),
                    type: portType(for: prop, name: key),
                    description: prop.description
                )
            }
        var remainingProperties: [String: SchemaProperty] = [:]
        for (key, prop) in allProperties where !required.contains(key) {
            remainingProperties[key] = prop
        }
        let paramSchema = JSONSchema(
            type: schema.type,
            properties: remainingProperties.isEmpty ? nil : remainingProperties,
            required: nil
        )
        return (ports, paramSchema)
    }

    /// Map a `SchemaProperty` to a `WorkflowPortType`. Path-typed
    /// properties (name ends in `_path`, or the schema's JSON type
    /// is "string" with a path-shaped description) become `.path`
    /// so the editor can offer a file picker; everything else
    /// falls through to the JSON-Schema type.
    public static func portType(for prop: SchemaProperty, name: String) -> WorkflowPortType {
        if name.hasSuffix("_path") || name == "path" {
            return .path
        }
        switch prop.type {
        case "string": return .string
        case "integer", "number": return .number
        case "boolean": return .boolean
        case "array", "object": return .bag
        default: return .string
        }
    }

    /// "load_model" -> "Load Model". Used for the editor's palette
    /// label; `typeId` stays snake_case for stable workflow JSON.
    public static func humanName(_ snake: String) -> String {
        snake.split(separator: "_")
            .map { $0.prefix(1).uppercased() + $0.dropFirst() }
            .joined(separator: " ")
    }

    public static func humanLabel(_ snake: String) -> String {
        humanName(snake)
    }

    /// Canonical execute body for a wrapped tool. Inputs are
    /// coerced to JSONValue, merged with the node's parameters
    /// (parameters win on conflict — the editor surfaces them
    /// as "this node's settings"), and the tool is invoked.
    /// `ToolResult.data` becomes the `result` port payload; if
    /// the tool returns nil data, a synthesised `{success,
    /// output, error}` map is produced so downstream nodes can
    /// always read the canonical fields.
    public static func execute(
        tool: any TesseraTool,
        parameters: [String: JSONValue],
        inputs: [String: WorkflowPortValue],
        context: WorkflowExecutionContext
    ) async throws -> [String: WorkflowPortValue] {
        var arguments: [String: JSONValue] = [:]
        for (key, value) in inputs {
            arguments[key] = value.asJSONValue
        }
        for (key, value) in parameters {
            arguments[key] = value
        }
        context.logger.log("workflow: tool \(tool.name) starting", level: .info)
        let result: ToolResult
        do {
            result = try await tool.execute(arguments: arguments)
        } catch {
            context.logger.log("workflow: tool \(tool.name) threw: \(error)", level: .error)
            return [
                "result": .toolResult([
                    "success": .bool(false),
                    "output": .string(""),
                    "error": .string(String(describing: error)),
                ])
            ]
        }
        context.logger.log("workflow: tool \(tool.name) finished ok=\(result.success)", level: .info)
        let payload: [String: JSONValue]
        if let data = result.data {
            payload = data
        } else {
            var synthesized: [String: JSONValue] = [
                "success": .bool(result.success),
                "output": .string(result.output),
            ]
            if let error = result.error {
                synthesized["error"] = .string(error)
            }
            payload = synthesized
        }
        return ["result": .toolResult(payload)]
    }

    /// The canonical output port (synthetic `result`).
    public static let resultPort: [WorkflowPort] = [
        WorkflowPort(
            id: "result",
            label: "Result",
            type: .toolResult,
            description: "Structured result of the tool call."
        )
    ]
}

// MARK: - Wrapped node types

/// `LoadModelTool` as a workflow node. Required input: `model_path`.
/// Parameters (side panel): `sidecar_path`, `runtime`, `n_ctx`.
public struct LoadModelNode: WorkflowNodeType {
    public static let typeId = "load_model"
    public static let displayName = TesseraToolNode.humanName("load_model")
    public static let summary = "Load a GGUF model into the Tessera engine."
    public static let tool: any TesseraTool = LoadModelTool()
    public static let inputs: [WorkflowPort] = TesseraToolNode
        .splitSchema(tool.parameters).0
    public static let outputs: [WorkflowPort] = TesseraToolNode.resultPort
    public static let parameterSchema: JSONSchema = TesseraToolNode
        .splitSchema(tool.parameters).1

    public static func execute(
        parameters: [String: JSONValue],
        inputs: [String: WorkflowPortValue],
        context: WorkflowExecutionContext
    ) async throws -> [String: WorkflowPortValue] {
        try await TesseraToolNode.execute(
            tool: tool, parameters: parameters, inputs: inputs, context: context)
    }
}

/// `CalibrateTool` as a workflow node. Required inputs:
/// `model_path`, `corpus_path`, `output_path`.
public struct CalibrateNode: WorkflowNodeType {
    public static let typeId = "calibrate"
    public static let displayName = TesseraToolNode.humanName("calibrate")
    public static let summary = "Run imatrix calibration on a BF16/FP16 model."
    public static let tool: any TesseraTool = CalibrateTool()
    public static let inputs: [WorkflowPort] = TesseraToolNode
        .splitSchema(tool.parameters).0
    public static let outputs: [WorkflowPort] = TesseraToolNode.resultPort
    public static let parameterSchema: JSONSchema = TesseraToolNode
        .splitSchema(tool.parameters).1

    public static func execute(
        parameters: [String: JSONValue],
        inputs: [String: WorkflowPortValue],
        context: WorkflowExecutionContext
    ) async throws -> [String: WorkflowPortValue] {
        try await TesseraToolNode.execute(
            tool: tool, parameters: parameters, inputs: inputs, context: context)
    }
}

/// `QuantizeTool` as a workflow node. Required inputs:
/// `model_path`, `output_path`, `policy_path`.
public struct QuantizeNode: WorkflowNodeType {
    public static let typeId = "quantize"
    public static let displayName = TesseraToolNode.humanName("quantize")
    public static let summary = "Quantize a GGUF model with a calibration policy."
    public static let tool: any TesseraTool = QuantizeTool()
    public static let inputs: [WorkflowPort] = TesseraToolNode
        .splitSchema(tool.parameters).0
    public static let outputs: [WorkflowPort] = TesseraToolNode.resultPort
    public static let parameterSchema: JSONSchema = TesseraToolNode
        .splitSchema(tool.parameters).1

    public static func execute(
        parameters: [String: JSONValue],
        inputs: [String: WorkflowPortValue],
        context: WorkflowExecutionContext
    ) async throws -> [String: WorkflowPortValue] {
        try await TesseraToolNode.execute(
            tool: tool, parameters: parameters, inputs: inputs, context: context)
    }
}

/// `EvaluateTool` as a workflow node. Required inputs: `model_path`,
/// `eval_corpus`.
public struct EvaluateNode: WorkflowNodeType {
    public static let typeId = "evaluate"
    public static let displayName = TesseraToolNode.humanName("evaluate")
    public static let summary = "Run perplexity / power / capability evaluation."
    public static let tool: any TesseraTool = EvaluateTool()
    public static let inputs: [WorkflowPort] = TesseraToolNode
        .splitSchema(tool.parameters).0
    public static let outputs: [WorkflowPort] = TesseraToolNode.resultPort
    public static let parameterSchema: JSONSchema = TesseraToolNode
        .splitSchema(tool.parameters).1

    public static func execute(
        parameters: [String: JSONValue],
        inputs: [String: WorkflowPortValue],
        context: WorkflowExecutionContext
    ) async throws -> [String: WorkflowPortValue] {
        try await TesseraToolNode.execute(
            tool: tool, parameters: parameters, inputs: inputs, context: context)
    }
}

/// `InspectSidecarTool` as a workflow node. Required input: `path`.
public struct InspectSidecarNode: WorkflowNodeType {
    public static let typeId = "inspect_sidecar"
    public static let displayName = TesseraToolNode.humanName("inspect_sidecar")
    public static let summary = "Read a calibration-policy JSON sidecar and summarise it."
    public static let tool: any TesseraTool = InspectSidecarTool()
    public static let inputs: [WorkflowPort] = TesseraToolNode
        .splitSchema(tool.parameters).0
    public static let outputs: [WorkflowPort] = TesseraToolNode.resultPort
    public static let parameterSchema: JSONSchema = TesseraToolNode
        .splitSchema(tool.parameters).1

    public static func execute(
        parameters: [String: JSONValue],
        inputs: [String: WorkflowPortValue],
        context: WorkflowExecutionContext
    ) async throws -> [String: WorkflowPortValue] {
        try await TesseraToolNode.execute(
            tool: tool, parameters: parameters, inputs: inputs, context: context)
    }
}

// MARK: - Default registry

extension WorkflowNodeRegistry {
    /// The default registry for Tessera Studio workflows. Bundles
    /// every wrapped `TesseraTool` that ships from Tessera Core;
    /// tests build their own minimal registry to avoid pulling
    /// the full tool implementations (and to control side effects).
    public static var `default`: WorkflowNodeRegistry {
        WorkflowNodeRegistry(types: [
            LoadModelNode.self,
            CalibrateNode.self,
            QuantizeNode.self,
            EvaluateNode.self,
            InspectSidecarNode.self,
        ])
    }
}
