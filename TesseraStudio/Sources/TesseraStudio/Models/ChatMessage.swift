import Foundation
import SwiftData

enum ChatRole: String, Codable, Sendable {
    case user
    case assistant
    case system
    case tool
}

/// A single tool call recorded within a chat message.
struct ToolCallRecord: Codable, Sendable, Identifiable {
    let id: UUID
    let toolName: String
    let arguments: [String: JSONValue]
    let result: ToolResultPayload?
    let timestamp: Date

    init(toolName: String, arguments: [String: JSONValue], result: ToolResultPayload? = nil) {
        self.id = UUID()
        self.toolName = toolName
        self.arguments = arguments
        self.result = result
        self.timestamp = Date()
    }
}

/// Codable payload for a tool result stored in SwiftData.
struct ToolResultPayload: Codable, Sendable {
    let success: Bool
    let output: String
    let error: String?
}

@Model
final class ChatMessage {
    var role: ChatRole
    var content: String
    var toolCalls: [ToolCallRecord]
    var timestamp: Date
    var conversationID: UUID

    init(
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
final class RunRecord {
    var modelName: String
    var runtime: TesseraRuntime
    var configJSON: String
    var metricsJSON: String
    var timestamp: Date
    var durationSeconds: Double
    var status: RunStatus

    init(
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

    var config: [String: JSONValue] {
        guard let data = configJSON.data(using: .utf8) else { return [:] }
        return (try? JSONDecoder().decode([String: JSONValue].self, from: data)) ?? [:]
    }

    var metrics: [String: JSONValue] {
        guard let data = metricsJSON.data(using: .utf8) else { return [:] }
        return (try? JSONDecoder().decode([String: JSONValue].self, from: data)) ?? [:]
    }
}

enum RunStatus: String, Codable, Sendable {
    case running
    case completed
    case failed
    case cancelled
}
