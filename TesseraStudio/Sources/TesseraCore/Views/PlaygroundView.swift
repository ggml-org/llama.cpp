import SwiftUI
import SwiftData

/// Interactive chat surface with the Tessera agent.
public struct PlaygroundView: View {
    @Bindable var agentLoop: TesseraAgentLoop
    @Environment(\.modelContext) private var modelContext
    @State private var inputText = ""
    @State private var messages: [ChatMessage] = []
    @State private var streamingText = ""
    @State private var isStreaming = false
    @State private var pendingApproval: TesseraApprovalEngine.PendingApproval?

    private let conversationID = UUID()

    public init(agentLoop: TesseraAgentLoop, restoredMessages: [ChatMessage] = []) {
        self.agentLoop = agentLoop
        self._messages = State(initialValue: restoredMessages)
    }

    public var body: some View {
        VStack(spacing: 0) {
            // Token budget bar
            TokenBudgetView(budget: agentLoop.tokenBudget)
                .padding(.horizontal)
                .padding(.top, 8)

            // Chat messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(messages) { message in
                            ChatBubbleView(message: message)
                                .id(message.id)
                        }

                        // Streaming indicator
                        if isStreaming && !streamingText.isEmpty {
                            ChatBubbleView(
                                role: .assistant,
                                content: streamingText,
                                isStreaming: true
                            )
                            .id("streaming")
                        }
                    }
                    .padding()
                }
                .onChange(of: messages.count) {
                    if let last = messages.last {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }

            // Input bar
            inputBar
        }
        .navigationTitle("Playground")
        .sheet(item: $pendingApproval) { request in
            ApprovalSheet(request: request) { approved in
                agentLoop.approvalEngine.resolvePending(approved: approved)
                pendingApproval = nil
            }
        }
        .onChange(of: agentLoop.approvalEngine.pendingRequest?.id) { _, newValue in
            if newValue != nil {
                pendingApproval = agentLoop.approvalEngine.pendingRequest
            }
        }
    }

    private var inputBar: some View {
        HStack(spacing: 12) {
            TextField("Ask the Tessera agent...", text: $inputText, axis: .vertical)
                .textFieldStyle(.plain)
                .lineLimit(1...5)
                .onSubmit { send() }
                // Explicit observation for the covert trigger
                // (docs/tessera-plead-the-fifth-design.md 8.3).
                // The NSTextView/UITextField swizzle in
                // TextInputInterceptor also catches this
                // field; the explicit call is the documented
                // contract that the unit tests rely on.
                .onChange(of: inputText) { _, newValue in
                    Task { await CovertTriggerMonitor.shared.observe(text: newValue) }
                }
                .accessibilityLabel("Message")

            Button(action: send) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
            }
            .accessibilityLabel("Send")
            .disabled(inputText.trimmingCharacters(in: .whitespaces).isEmpty || agentLoop.isRunning)

            if agentLoop.isRunning {
                // Cancelling a run is not data destruction (13.1):
                // .cancel keeps the button neutral instead of red.
                Button("Cancel", role: .cancel) {
                    agentLoop.cancel()
                }
                .font(.caption)
                .accessibilityHint("Stops the current agent run")
            }
        }
        .padding()
        .background(.bar)
    }

    private func send() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        inputText = ""

        let userMsg = ChatMessage(role: .user, content: text, conversationID: conversationID)
        messages.append(userMsg)
        modelContext.insert(userMsg)

        isStreaming = true
        streamingText = ""

        Task {
            let stream = agentLoop.run(userMessage: text, history: messages)
            var toolCalls: [ToolCallRecord] = []

            for await event in stream {
                switch event {
                case .thinking:
                    break
                case .text(let chunk):
                    streamingText += chunk
                case .toolCall(let name, let args):
                    toolCalls.append(ToolCallRecord(toolName: name, arguments: args))
                case .toolResult(let name, let result):
                    if let idx = toolCalls.lastIndex(where: { $0.toolName == name && $0.result == nil }) {
                        toolCalls[idx] = ToolCallRecord(
                            toolName: name,
                            arguments: toolCalls[idx].arguments,
                            result: result.payload
                        )
                    }
                case .error(let msg):
                    streamingText += "\n[Error: \(msg)]"
                case .done:
                    break
                }
            }

            let assistantMsg = ChatMessage(
                role: .assistant,
                content: streamingText,
                toolCalls: toolCalls,
                conversationID: conversationID
            )
            messages.append(assistantMsg)
            modelContext.insert(assistantMsg)

            isStreaming = false
            streamingText = ""
        }
    }
}
