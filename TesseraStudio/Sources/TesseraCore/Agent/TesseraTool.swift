import Foundation

// MARK: - JSON Value

/// A type-erased JSON value for tool arguments and results.
public enum JSONValue: Codable, Sendable, Equatable, Hashable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case array([JSONValue])
    case object([String: JSONValue])
    case null

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let v = try? container.decode(Bool.self) {
            self = .bool(v)
        } else if let v = try? container.decode(Double.self) {
            self = .number(v)
        } else if let v = try? container.decode(String.self) {
            self = .string(v)
        } else if let v = try? container.decode([JSONValue].self) {
            self = .array(v)
        } else if let v = try? container.decode([String: JSONValue].self) {
            self = .object(v)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported JSON value")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let v): try container.encode(v)
        case .number(let v): try container.encode(v)
        case .bool(let v): try container.encode(v)
        case .array(let v): try container.encode(v)
        case .object(let v): try container.encode(v)
        case .null: try container.encodeNil()
        }
    }

    public var stringValue: String? {
        if case .string(let v) = self { return v }
        return nil
    }

    public var numberValue: Double? {
        if case .number(let v) = self { return v }
        return nil
    }

    public var boolValue: Bool? {
        if case .bool(let v) = self { return v }
        return nil
    }

    /// A short human-readable description, used across the UI.
    public var shortDescription: String {
        switch self {
        case .string(let s): return s
        case .number(let n): return String(format: "%g", n)
        case .bool(let b): return b ? "true" : "false"
        case .null: return "null"
        case .array(let a): return "[\(a.count) items]"
        case .object(let o): return "{\(o.count) keys}"
        }
    }
}

// MARK: - JSON Schema

/// A JSON Schema descriptor for tool parameters.
public struct JSONSchema: Codable, Sendable, Equatable, Hashable {
    public let type: String
    public let properties: [String: SchemaProperty]?
    public let required: [String]?

    public init(type: String = "object", properties: [String: SchemaProperty]? = nil, required: [String]? = nil) {
        self.type = type
        self.properties = properties
        self.required = required
    }

    public func toJSON() -> String {
        guard let data = try? JSONEncoder().encode(self),
              let str = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return str
    }
}

public struct SchemaProperty: Codable, Sendable, Equatable, Hashable {
    public let type: String
    public let description: String?
    public let enumValues: [String]?
    public let defaultValue: String?
    public let minimum: Double?
    public let maximum: Double?

    enum CodingKeys: String, CodingKey {
        case type
        case description
        case enumValues = "enum"
        case defaultValue = "default"
        case minimum
        case maximum
    }

    public init(type: String, description: String? = nil, enumValues: [String]? = nil, defaultValue: String? = nil, minimum: Double? = nil, maximum: Double? = nil) {
        self.type = type
        self.description = description
        self.enumValues = enumValues
        self.defaultValue = defaultValue
        self.minimum = minimum
        self.maximum = maximum
    }
}

// MARK: - Tool Result

/// The result of executing a tool.
public struct ToolResult: Sendable {
    public let success: Bool
    public let output: String
    public let data: [String: JSONValue]?
    public let error: String?

    public init(success: Bool, output: String, data: [String: JSONValue]? = nil, error: String? = nil) {
        self.success = success
        self.output = output
        self.data = data
        self.error = error
    }

    public static func ok(_ output: String, data: [String: JSONValue]? = nil) -> ToolResult {
        ToolResult(success: true, output: output, data: data)
    }

    public static func fail(_ error: String) -> ToolResult {
        ToolResult(success: false, output: "", error: error)
    }

    public var payload: ToolResultPayload {
        ToolResultPayload(success: success, output: output, error: error)
    }
}

// MARK: - TesseraTool Protocol

/// A tool that the agent can invoke. Each tool declares its name,
/// description, parameter schema, default approval level, and
/// an async execute method.
public protocol TesseraTool: Sendable {
    var name: String { get }
    var description: String { get }
    var parameters: JSONSchema { get }
    var defaultApprovalLevel: ApprovalLevel { get }
    func execute(arguments: [String: JSONValue]) async throws -> ToolResult
}
