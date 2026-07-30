import SwiftUI

/// Renders a single chat message as a bubble.
struct ChatBubbleView: View {
    let role: ChatRole
    let content: String
    let toolCalls: [ToolCallRecord]
    let isStreaming: Bool

    init(message: ChatMessage) {
        self.role = message.role
        self.content = message.content
        self.toolCalls = message.toolCalls
        self.isStreaming = false
    }

    init(role: ChatRole, content: String, isStreaming: Bool = false) {
        self.role = role
        self.content = content
        self.toolCalls = []
        self.isStreaming = isStreaming
    }

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            if role == .user { Spacer(minLength: 60) }

            VStack(alignment: role == .user ? .trailing : .leading, spacing: 6) {
                // Role label
                Text(roleLabel)
                    .font(.caption2.bold())
                    .foregroundStyle(.secondary)

                // Message content
                if !content.isEmpty {
                    Text(content)
                        .textSelection(.enabled)
                        .padding(10)
                        .background(bubbleColor, in: RoundedRectangle(cornerRadius: 12))
                }

                // Tool calls
                ForEach(toolCalls) { call in
                    ToolCallView(record: call)
                }

                // Streaming indicator
                if isStreaming {
                    HStack(spacing: 4) {
                        ProgressView()
                            .controlSize(.mini)
                        Text("Generating...")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            if role != .user { Spacer(minLength: 60) }
        }
    }

    private var roleLabel: String {
        switch role {
        case .user: "You"
        case .assistant: "Tessera Agent"
        case .system: "System"
        case .tool: "Tool"
        }
    }

    private var bubbleColor: Color {
        switch role {
        case .user: .blue.opacity(0.15)
        case .assistant: .gray.opacity(0.1)
        case .system: .yellow.opacity(0.1)
        case .tool: .green.opacity(0.1)
        }
    }
}
