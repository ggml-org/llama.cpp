import SwiftUI
import TesseraCore

// MARK: - ReceiptRowView

/// One row in the receipts drawer's chain list. The row
/// shows the actor (user / agent), the timestamp, the
/// summary, and a small icon for the receipt's status
/// (applied, voided, export). Tapping the row opens the
/// receipt in the detail view.
public struct ReceiptRowView: View {

    public let receipt: Receipt
    public let isSelected: Bool

    public init(receipt: Receipt, isSelected: Bool = false) {
        self.receipt = receipt
        self.isSelected = isSelected
    }

    public var body: some View {
        HStack(alignment: .top, spacing: 8) {
            actorIcon
                .frame(width: 18, height: 18)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(receipt.summary)
                        .font(.system(size: 12, weight: .medium))
                        .lineLimit(1)
                    if receipt.isVoided {
                        Text("voided")
                            .font(.system(size: 9, weight: .medium))
                            .padding(.horizontal, 4)
                            .padding(.vertical, 1)
                            .background(Capsule().fill(Color.red.opacity(0.15)))
                            .foregroundStyle(.red)
                    }
                }
                HStack(spacing: 6) {
                    Text(actorLabel)
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                    Text("·")
                        .font(.system(size: 10))
                        .foregroundStyle(.tertiary)
                    Text(timestampText)
                        .font(.system(size: 10))
                        .foregroundStyle(.secondary)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 8)
        .background(isSelected ? Color.accentColor.opacity(0.12) : Color.clear)
        .contentShape(Rectangle())
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(actorLabel), \(receipt.summary), \(timestampText)")
    }

    // MARK: - Pieces

    private var actorIcon: some View {
        Image(systemName: actorIconName)
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(actorIconTint)
    }

    private var actorIconName: String {
        switch receipt.actor {
        case .user: return "person.fill"
        case .agent: return "cpu"
        }
    }

    private var actorIconTint: Color {
        switch receipt.actor {
        case .user: return .accentColor
        case .agent: return .purple
        }
    }

    private var actorLabel: String {
        switch receipt.actor {
        case .user: return "user"
        case .agent(_, let model, _):
            return "agent · \(model)"
        }
    }

    private var timestampText: String {
        let f = DateFormatter()
        f.dateStyle = .short
        f.timeStyle = .short
        return f.string(from: receipt.timestamp)
    }
}
