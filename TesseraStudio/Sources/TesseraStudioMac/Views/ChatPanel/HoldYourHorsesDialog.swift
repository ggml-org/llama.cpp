import SwiftUI
import TesseraCore

// MARK: - HoldYourHorsesDialog

/// The "Hold your horses" dialog (per spec §6.8). When the
/// user clicks the button in the chat panel footer, the
/// queue pauses and this dialog is shown. The user can
/// describe what's working and what's not, and the agent
/// can suggest reorderings (the reordering suggestions
/// appear as pending items in the queue).
///
/// **macOS:** the dialog is a SwiftUI sheet. The parent
/// view presents it via `.sheet(...)` when the model
/// sets `holdDialog` to non-nil.
///
/// **iOS:** the dialog is a modal sheet with the same
/// content. iOS uses a `.sheet` rather than a `.popover`
/// to match the platform convention.
///
/// **VoiceOver.** The dialog has a single label
/// ("Hold your horses — is something wrong?") and a
/// "Resume" button at the bottom. The text field has its
/// own label.
public struct HoldYourHorsesDialog: View {

    @Binding public var response: String
    public let state: ChatPanelViewModel.HoldDialogState
    public let onSubmit: () -> Void
    public let onResume: () -> Void
    public let onCancel: () -> Void

    @FocusState private var responseFocused: Bool

    public init(
        response: Binding<String>,
        state: ChatPanelViewModel.HoldDialogState,
        onSubmit: @escaping () -> Void,
        onResume: @escaping () -> Void,
        onCancel: @escaping () -> Void
    ) {
        self._response = response
        self.state = state
        self.onSubmit = onSubmit
        self.onResume = onResume
        self.onCancel = onCancel
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 8) {
                Image(systemName: "pause.circle.fill")
                    .font(.system(size: 24))
                    .foregroundStyle(.orange)
                Text(state.title)
                    .font(.system(size: 18, weight: .semibold))
            }

            Text(state.message)
                .font(.system(size: 13))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            VStack(alignment: .leading, spacing: 4) {
                Text("What's working? What's not?")
                    .font(.system(size: 12, weight: .medium))
                TextEditor(text: $response)
                    .font(.system(size: 13))
                    .frame(minHeight: 80, maxHeight: 160)
                    .padding(4)
                    .background(
                        RoundedRectangle(cornerRadius: 5)
                            .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                    )
                    .focused($responseFocused)
                    .accessibilityLabel("Response to the agent")
            }

            HStack {
                Spacer()
                Button("Cancel", action: onCancel)
                    .keyboardShortcut(.cancelAction)
                Button("Submit & continue", action: onSubmit)
                    .keyboardShortcut(.defaultAction)
                    .disabled(response.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Button {
                    onResume()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "play.fill")
                        Text("Resume")
                    }
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut("r", modifiers: [.command])
            }
        }
        .padding(20)
        .frame(minWidth: 360, idealWidth: 420, maxWidth: 520)
        .onAppear { responseFocused = true }
    }
}
