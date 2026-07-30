import SwiftUI

/// Sheet presented when a tool requires explicit user approval.
struct ApprovalSheet: View {
    let request: TesseraApprovalEngine.PendingApproval
    let onResolve: (Bool) -> Void

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 16) {
                // Header
                Label("Tool Approval Required", systemImage: "hand.raised")
                    .font(.headline)

                // Tool name
                VStack(alignment: .leading, spacing: 4) {
                    Text("Tool")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(request.toolName)
                        .font(.system(.body, design: .monospaced).bold())
                }

                // Arguments
                if !request.arguments.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Arguments")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        ForEach(request.arguments.sorted(by: { $0.key < $1.key }), id: \.key) { key, value in
                            HStack(alignment: .top) {
                                Text(key)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundStyle(.secondary)
                                Text(describe(value))
                                    .font(.system(.caption, design: .monospaced))
                                    .textSelection(.enabled)
                            }
                        }
                    }
                    .padding()
                    .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
                }

                Spacer()

                // Action buttons
                HStack {
                    Button("Deny", role: .destructive) {
                        onResolve(false)
                        dismiss()
                    }
                    .buttonStyle(.bordered)

                    Spacer()

                    Button("Approve") {
                        onResolve(true)
                        dismiss()
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .padding()
            .navigationTitle("Approval")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        onResolve(false)
                        dismiss()
                    }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 400, minHeight: 300)
        #endif
    }

    private func describe(_ value: JSONValue) -> String {
        switch value {
        case .string(let s): s
        case .number(let n): String(format: "%g", n)
        case .bool(let b): b ? "true" : "false"
        case .null: "null"
        case .array(let a): "[\(a.count) items]"
        case .object(let o): "{\(o.count) keys}"
        }
    }
}
