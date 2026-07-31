import Foundation
import SwiftData

/// A persisted conversation, the unit shown in the chat history drawer.
/// Individual messages (ChatMessage) link back via `conversationID == id`.
@Model
public final class Conversation {
    @Attribute(.unique) public var id: UUID
    public var title: String
    public var modelName: String
    public var toolNames: [String]
    public var createdAt: Date
    public var updatedAt: Date

    public init(
        id: UUID = UUID(),
        title: String,
        modelName: String = "",
        toolNames: [String] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.title = title
        self.modelName = modelName
        self.toolNames = toolNames
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}
