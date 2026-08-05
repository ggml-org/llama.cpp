import SwiftUI
import TesseraCore

/// macOS wrapper around the shared ``GraphView``. The macOS
/// target owns the data-layer wiring (the shared
/// ``GraphView`` is in `TesseraCore` and only knows about
/// the view model). The view is exposed as a window so the
/// user can pop the graph out of the main workspace.
public struct GraphWindowView: View {

    public init(store: GraphStore) {
        self.store = store
        self._viewModel = State(initialValue: GraphViewModel(store: store))
    }

    private let store: GraphStore
    @State private var viewModel: GraphViewModel

    public var body: some View {
        GraphView(viewModel: viewModel)
            .frame(minWidth: 900, minHeight: 600)
    }
}
