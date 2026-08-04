import SwiftUI

/// Visualizes the token budget usage for the current session.
public struct TokenBudgetView: View {
    public let budget: TokenBudget

    public init(budget: TokenBudget) {
        self.budget = budget
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Token Budget")
                    .font(.caption.bold())
                Spacer()
                Text("\(budget.used) / \(budget.limit)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 3)
                        .fill(.quaternary)
                        .frame(height: 6)

                    RoundedRectangle(cornerRadius: 3)
                        .fill(barColor)
                        .frame(width: geo.size.width * budget.fraction, height: 6)
                }
            }
            .frame(height: 6)
            // The bar is visual-only; the used / limit / remaining
            // texts around it carry the same information.
            .accessibilityHidden(true)

            HStack {
                Text("\(budget.remaining) remaining")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
                if budget.fraction > 0.8 {
                    Label("Budget nearly exhausted", systemImage: "exclamationmark.triangle")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }
            }
        }
    }

    private var barColor: Color {
        if budget.fraction > 0.9 { return .red }
        if budget.fraction > 0.7 { return .orange }
        return .green
    }
}
