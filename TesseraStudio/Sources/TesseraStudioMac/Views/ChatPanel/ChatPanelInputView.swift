import SwiftUI
import TesseraCore

// MARK: - ChatPanelInputView

/// The input region of the chat panel. The user types a
/// command in the text field; pressing return enqueues
/// the text as a new pending item. The "Hold your horses"
/// button is on the right and is always present (per
/// architect decision, spec §6.8).
public struct ChatPanelInputView: View {

    @Binding public var text: String
    public let holdMode: HoldMode
    public let isInProgress: Bool
    public let onSubmit: () -> Void
    public let onHoldYourHorses: () -> Void
    public let onCancelInProgress: () -> Void

    @FocusState private var inputFocused: Bool

    public init(
        text: Binding<String>,
        holdMode: HoldMode,
        isInProgress: Bool,
        onSubmit: @escaping () -> Void,
        onHoldYourHorses: @escaping () -> Void,
        onCancelInProgress: @escaping () -> Void
    ) {
        self._text = text
        self.holdMode = holdMode
        self.isInProgress = isInProgress
        self.onSubmit = onSubmit
        self.onHoldYourHorses = onHoldYourHorses
        self.onCancelInProgress = onCancelInProgress
    }

    public var body: some View {
        HStack(spacing: 8) {
            inputField
            if isInProgress {
                cancelButton
            } else {
                holdButton
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.thinMaterial)
    }

    // MARK: - Pieces

    private var inputField: some View {
        TextField("Type a command…", text: $text, axis: .vertical)
            .textFieldStyle(.roundedBorder)
            .font(.system(size: 13))
            .lineLimit(1...3)
            .focused($inputFocused)
            .onSubmit(onSubmit)
            .disabled(isInProgress)
            .help(isInProgress
                ? "Agent is working — wait for the current task to finish or cancel it"
                : "Type a command for the agent (return to send)")
    }

    private var holdButton: some View {
        Button(action: onHoldYourHorses) {
            HStack(spacing: 4) {
                Image(systemName: holdIcon)
                    .font(.system(size: 11))
                Text(holdMode.footerButtonLabel)
                    .font(.system(size: 11, weight: .medium))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(
                RoundedRectangle(cornerRadius: 5)
                    .fill(holdMode.isUserPaused ? Color.orange.opacity(0.25) : Color.orange.opacity(0.18))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 5)
                    .stroke(Color.orange.opacity(holdMode.isUserPaused ? 0.6 : 0.4), lineWidth: 1)
            )
            .foregroundStyle(.orange)
        }
        .buttonStyle(.plain)
        .help(holdMode.isUserPaused
            ? "Resume the agent"
            : "Pause the agent and open a conversation")
    }

    private var cancelButton: some View {
        Button(action: onCancelInProgress) {
            HStack(spacing: 4) {
                Image(systemName: "stop.circle")
                    .font(.system(size: 11))
                Text("Stop")
                    .font(.system(size: 11, weight: .medium))
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(
                RoundedRectangle(cornerRadius: 5)
                    .fill(Color.red.opacity(0.18))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 5)
                    .stroke(Color.red.opacity(0.4), lineWidth: 1)
            )
            .foregroundStyle(.red)
        }
        .buttonStyle(.plain)
        .help("Stop the agent's current work")
    }

    private var holdIcon: String {
        switch holdMode {
        case .running, .holdRequested: return "pause.fill"
        case .hold, .resuming: return "play.fill"
        }
    }
}
