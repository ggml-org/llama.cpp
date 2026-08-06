import SwiftUI

// MARK: - FlowLayout

/// A simple flowing layout: items wrap onto the next line
/// when they don't fit in the current row. Used by the
/// notes surface's tag chip strip and the editor's
/// linked-entities row. This is a SwiftUI Layout (the
/// Layout protocol landed in iOS 16 / macOS 13) so it
/// works in nested scroll views and respects Dynamic Type.
struct FlowLayout: Layout {

    var spacing: CGFloat = 4
    var lineSpacing: CGFloat = 4

    struct Cache {
        var rows: [Row] = []
        var totalSize: CGSize = .zero
    }

    struct Row {
        var items: [LayoutSubview]
        var sizes: [CGSize]
        var width: CGFloat
        var height: CGFloat
    }

    func makeCache(subviews: Subviews) -> Cache { Cache() }

    func sizeThatFits(
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout Cache
    ) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        let rows = computeRows(subviews: subviews, maxWidth: maxWidth)
        let height = rows.reduce(0) { $0 + $1.height } + CGFloat(max(0, rows.count - 1)) * lineSpacing
        let width = maxWidth.isFinite
            ? maxWidth
            : (rows.map { $0.width }.max() ?? 0)
        return CGSize(width: width, height: height)
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout Cache
    ) {
        let rows = computeRows(subviews: subviews, maxWidth: bounds.width)
        var y = bounds.minY
        for row in rows {
            var x = bounds.minX
            for (index, subview) in row.items.enumerated() {
                let size = row.sizes[index]
                subview.place(
                    at: CGPoint(x: x, y: y),
                    proposal: ProposedViewSize(width: size.width, height: size.height)
                )
                x += size.width + spacing
            }
            y += row.height + lineSpacing
        }
    }

    private func computeRows(subviews: Subviews, maxWidth: CGFloat) -> [Row] {
        var rows: [Row] = []
        var current = Row(items: [], sizes: [], width: 0, height: 0)

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            let needsSpace = current.width + size.width + (current.items.isEmpty ? 0 : spacing)

            if needsSpace > maxWidth && !current.items.isEmpty {
                rows.append(current)
                current = Row(items: [], sizes: [], width: 0, height: 0)
            }

            if !current.items.isEmpty {
                current.width += spacing
            }
            current.items.append(subview)
            current.sizes.append(size)
            current.width += size.width
            current.height = max(current.height, size.height)
        }
        if !current.items.isEmpty {
            rows.append(current)
        }
        return rows
    }
}
