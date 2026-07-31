import Foundation
import SwiftData

public enum ChatRole: String, Codable, Sendable {
    case user
    case assistant
    case system
    case tool
}

/// A single tool call recorded within a chat message.
public struct ToolCallRecord: Codable, Sendable, Identifiable {
    public let id: UUID
    public let toolName: String
    public let arguments: [String: JSONValue]
    public let result: ToolResultPayload?
    public let timestamp: Date

    public init(toolName: String, arguments: [String: JSONValue], result: ToolResultPayload? = nil) {
        self.id = UUID()
        self.toolName = toolName
        self.arguments = arguments
        self.result = result
        self.timestamp = Date()
    }
}

/// Codable payload for a tool result stored in SwiftData.
public struct ToolResultPayload: Codable, Sendable {
    public let success: Bool
    public let output: String
    public let error: String?

    public init(success: Bool, output: String, error: String? = nil) {
        self.success = success
        self.output = output
        self.error = error
    }
}

@Model
public final class ChatMessage {
    public var role: ChatRole
    public var content: String
    public var toolCalls: [ToolCallRecord]
    public var timestamp: Date
    public var conversationID: UUID

    public init(
        role: ChatRole,
        content: String,
        toolCalls: [ToolCallRecord] = [],
        conversationID: UUID = UUID()
    ) {
        self.role = role
        self.content = content
        self.toolCalls = toolCalls
        self.timestamp = Date()
        self.conversationID = conversationID
    }
}

@Model
public final class RunRecord {
    public var modelName: String
    public var runtime: TesseraRuntime
    public var configJSON: String
    public var metricsJSON: String
    public var timestamp: Date
    public var durationSeconds: Double
    public var status: RunStatus

    public init(
        modelName: String,
        runtime: TesseraRuntime,
        config: [String: JSONValue] = [:],
        metrics: [String: JSONValue] = [:],
        durationSeconds: Double = 0,
        status: RunStatus = .running
    ) {
        self.modelName = modelName
        self.runtime = runtime
        self.configJSON = (try? JSONEncoder().encode(config)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        self.metricsJSON = (try? JSONEncoder().encode(metrics)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
        self.timestamp = Date()
        self.durationSeconds = durationSeconds
        self.status = status
    }

    public var config: [String: JSONValue] {
        guard let data = configJSON.data(using: .utf8) else { return [:] }
        return (try? JSONDecoder().decode([String: JSONValue].self, from: data)) ?? [:]
    }

    public var metrics: [String: JSONValue] {
        guard let data = metricsJSON.data(using: .utf8) else { return [:] }
        return (try? JSONDecoder().decode([String: JSONValue].self, from: data)) ?? [:]
    }
}

public enum RunStatus: String, Codable, Sendable {
    case running
    case completed
    case failed
    case cancelled
}
