import Foundation

// MARK: - JSON Value

/// A type-erased JSON value for tool arguments and results.
enum JSONValue: Codable, Sendable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case array([JSONValue])
    case object([String: JSONValue])
    case null

    init(from decoder: Decoder) throws {
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

    func encode(to encoder: Encoder) throws {
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

    var stringValue: String? {
        if case .string(let v) = self { return v }
        return nil
    }

    var numberValue: Double? {
        if case .number(let v) = self { return v }
        return nil
    }
}

// MARK: - JSON Schema

/// A JSON Schema descriptor for tool parameters.
struct JSONSchema: Codable, Sendable {
    let type: String
    let properties: [String: SchemaProperty]?
    let required: [String]?

    init(type: String = "object", properties: [String: SchemaProperty]? = nil, required: [String]? = nil) {
        self.type = type
        self.properties = properties
        self.required = required
    }

    func toJSON() -> String {
        guard let data = try? JSONEncoder().encode(self),
              let str = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return str
    }
}

struct SchemaProperty: Codable, Sendable {
    let type: String
    let description: String?
    let enumValues: [String]?
    let defaultValue: String?

    enum CodingKeys: String, CodingKey {
        case type
        case description
        case enumValues = "enum"
        case defaultValue = "default"
    }

    init(type: String, description: String? = nil, enumValues: [String]? = nil, defaultValue: String? = nil) {
        self.type = type
        self.description = description
        self.enumValues = enumValues
        self.defaultValue = defaultValue
    }
}

// MARK: - Tool Result

/// The result of executing a tool.
struct ToolResult: Sendable {
    let success: Bool
    let output: String
    let data: [String: JSONValue]?
    let error: String?

    init(success: Bool, output: String, data: [String: JSONValue]? = nil, error: String? = nil) {
        self.success = success
        self.output = output
        self.data = data
        self.error = error
    }

    static func ok(_ output: String, data: [String: JSONValue]? = nil) -> ToolResult {
        ToolResult(success: true, output: output, data: data)
    }

    static func fail(_ error: String) -> ToolResult {
        ToolResult(success: false, output: "", error: error)
    }

    var payload: ToolResultPayload {
        ToolResultPayload(success: success, output: output, error: error)
    }
}

// MARK: - TesseraTool Protocol

/// A tool that the agent can invoke. Each tool declares its name,
/// description, parameter schema, default approval level, and
/// an async execute method.
protocol TesseraTool: Sendable {
    var name: String { get }
    var description: String { get }
    var parameters: JSONSchema { get }
    var defaultApprovalLevel: ApprovalLevel { get }
    func execute(arguments: [String: JSONValue]) async throws -> ToolResult
}
