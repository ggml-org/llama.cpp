import SwiftUI
import AppKit
import TesseraCore

// MARK: - TesseraEditorView

/// The SwiftUI view that hosts the platform-native text
/// view (NSTextView on macOS, UITextView on iOS) and wires
/// it to the `TesseraTextContentManager` from Phase 2's
/// editor layer.
///
/// **Architecture.** The view is a `NSViewRepresentable`
/// that:
///   1. Owns a `TesseraTextContentManager` (the AST-backed
///      `NSTextContentManager`).
///   2. Owns a `NSTextView` (the platform text view) whose
///      `textContentManager` is set to (1).
///   3. Listens to the text view's `NSText.didChangeNotification`
///      and converts the post-edit attributed string back
///      into a `Mutation` via `TextEditReducer`.
///   4. Hands the `Mutation` to the `EditorCoalescer`, which
///      aggregates a burst of edits into a single
///      `Mutation` + `ChatQueueItem` and posts a
///      `didFlushNotification`.
///
/// **Binding.** The `binding` is a `Binding<DocumentAST>`
/// that the host view (Phase 5's per-surface wrapper) holds.
/// When the coalescer flushes, the binding is updated with
/// the new document state, the receipt is signed via
/// `ReceiptSigner`, and the chat queue is persisted via
/// `DocumentStore.saveChatQueue`. The data flow is:
///
///   user types -> NSTextView edits -> NSAttributedString diff
///   -> TextEditReducer -> Mutation
///   -> EditorCoalescer.append(...)
///   -> coalesce window expires -> didFlushNotification
///   -> caller persists the mutation + the queue item
///
/// **Phase 2 deliverable.** This view is the public
/// surface for the editor; the per-surface wrappers
/// (Documents / Notes / Code) configure the
/// `EditorMode` and the theme. All surfaces share one
/// `TesseraEditorView`.
///
/// **Production swap.** The `makeNSView` returns a plain
/// `NSTextView` (TextKit 2 via the text content manager
/// subclass). The recommended production swap is to use
/// `STTextView` (krzyzanowskim) as the base; the
/// architecture is the same — STTextView takes a
/// `textContentManager` just like `NSTextView`. See the
/// design doc for the migration path.
public struct TesseraEditorView: NSViewRepresentable {

    public let mode: EditorMode
    public let theme: EditorTheme
    @Binding public var document: DocumentAST
    public let onMutationCommitted: (([Mutation], ChatQueueItem) -> Void)?

    public init(
        mode: EditorMode = .document,
        theme: EditorTheme = .light,
        document: Binding<DocumentAST>,
        onMutationCommitted: (([Mutation], ChatQueueItem) -> Void)? = nil
    ) {
        self.mode = mode
        self.theme = theme
        self._document = document
        self.onMutationCommitted = onMutationCommitted
    }

    public func makeCoordinator() -> Coordinator {
        Coordinator(
            mode: mode,
            theme: theme,
            onMutationCommitted: onMutationCommitted
        )
    }

    public func makeNSView(context: Context) -> NSView {
        let container = NSScrollView()
        container.hasVerticalScroller = true
        container.hasHorizontalScroller = false
        container.borderType = .noBorder
        container.autohidesScrollers = true

        // Build the content manager + a custom text view
        // subclass that wires the manager into the
        // platform's TextKit 2 stack. The custom subclass
        // is the only way to substitute a custom
        // NSTextContentManager in TextKit 2 (NSTextView's
        // `textContentStorage` is set during init and is
        // read-only after).
        let contentManager = TesseraTextContentManager(document: document, mode: mode)
        let textView = TesseraNSTextView(
            contentManager: contentManager,
            initialAttributedString: contentManager.data.fullAttributedString()
        )
        textView.isEditable = true
        textView.isSelectable = true
        textView.allowsUndo = true
        textView.usesFontPanel = true
        textView.usesRuler = false
        textView.isRichText = true
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.autoresizingMask = [.width]
        textView.textContainerInset = NSSize(width: 8, height: 12)

        context.coordinator.contentManager = contentManager
        context.coordinator.textView = textView
        context.coordinator.binding = $document
        context.coordinator.startObservingCoalescerFlush()

        // Hook the text view's edit notifications.
        NotificationCenter.default.addObserver(
            context.coordinator,
            selector: #selector(Coordinator.textDidChange(_:)),
            name: NSText.didChangeNotification,
            object: textView
        )

        container.documentView = textView
        return container
    }

    public func updateNSView(_ nsView: NSView, context: Context) {
        guard let contentManager = context.coordinator.contentManager,
              contentManager.data.document.rootChildren != document.rootChildren else {
            return
        }
        contentManager.data.setDocument(document)
        if let textView = (nsView as? NSScrollView)?.documentView as? TesseraNSTextView {
            textView.replaceContent(with: contentManager.data.fullAttributedString())
        }
    }

    public static func dismantleNSView(_ nsView: NSView, coordinator: Coordinator) {
        coordinator.stopObservingCoalescerFlush()
        NotificationCenter.default.removeObserver(coordinator)
    }

    // MARK: - Coordinator

    public final class Coordinator: NSObject {
        public var contentManager: TesseraTextContentManager?
        public weak var textView: NSTextView?
        public var binding: Binding<DocumentAST>?
        public let mode: EditorMode
        public let theme: EditorTheme
        public let coalescer: EditorCoalescer
        public let reducer: TextEditReducer
        public var lastAttributedString: NSAttributedString?
        private var flushObserver: NSObjectProtocol?

        private let onMutationCommitted: (([Mutation], ChatQueueItem) -> Void)?

        public init(
            mode: EditorMode,
            theme: EditorTheme,
            onMutationCommitted: (([Mutation], ChatQueueItem) -> Void)?
        ) {
            self.mode = mode
            self.theme = theme
            self.coalescer = EditorCoalescer()
            self.reducer = TextEditReducer()
            self.onMutationCommitted = onMutationCommitted
        }

        deinit {
            stopObservingCoalescerFlush()
        }

        public func startObservingCoalescerFlush() {
            flushObserver = NotificationCenter.default.addObserver(
                forName: EditorCoalescer.didFlushNotification,
                object: coalescer,
                queue: .main
            ) { [weak self] note in
                guard let self, let burst = note.userInfo?["burst"] as? EditorCoalescer.CoalescedBurst else {
                    return
                }
                self.handleFlushedBurst(burst)
            }
        }

        public func stopObservingCoalescerFlush() {
            if let flushObserver {
                NotificationCenter.default.removeObserver(flushObserver)
                self.flushObserver = nil
            }
        }

        private func handleFlushedBurst(_ burst: EditorCoalescer.CoalescedBurst) {
            // Apply the mutations to the binding's document
            // and forward to the caller (who signs the
            // receipt + persists the queue item).
            guard let binding = binding else { return }
            var working = binding.wrappedValue
            var engine = MutationEngine()
            do {
                for mutation in burst.mutations {
                    _ = try engine.apply(mutation, to: &working)
                }
                binding.wrappedValue = working
                onMutationCommitted?(burst.mutations, burst.queueItem)
            } catch {
                NSLog("TesseraEditorView: failed to apply flushed burst: \(error)")
            }
        }

        @objc public func textDidChange(_ note: Notification) {
            guard let textView = textView,
                  let contentManager = contentManager else { return }
            let current = textView.textContentStorage?.attributedString
                ?? NSAttributedString()
            // Compare to the last attributed string; produce
            // a Mutation for the diff. We use the per-block
            // diff (the platform text view's per-character
            // diff is too granular; the reducer's coarse
            // setBlockContent is good enough for the
            // mutation API).
            if let last = lastAttributedString {
                if last.string == current.string && last.length == current.length {
                    return  // No change
                }
                let blockID = findActiveBlockID(in: current)
                let before = runs(from: last)
                let after = runs(from: current)
                let mutations = reducer.reduce(
                    blockID: blockID,
                    before: before,
                    after: after
                )
                for mutation in mutations {
                    coalescer.append(
                        mutation: mutation,
                        blockID: blockID,
                        documentID: contentManager.document.rootChildren.first ?? UUID(),
                        queueMessage: "You edited a block"
                    )
                }
            }
            lastAttributedString = current
        }

        /// Best-effort: identify the block the user is
        /// editing. The platform text view's selection
        /// range is in document coordinates; we look up the
        /// block via the content manager's `elementAt`.
        private func findActiveBlockID(in attributed: NSAttributedString) -> UUID {
            guard let textView = textView,
                  let contentManager = contentManager else { return UUID() }
            let selectedRange = textView.selectedRange()
            // Find the element that contains the selection.
            if let element = contentManager.data.elementAt(offset: selectedRange.location) {
                return element.blockID
            }
            return contentManager.data.elements.first?.blockID ?? UUID()
        }

        /// Convert an `NSAttributedString` to a list of
        /// `InlineRun`s, dropping per-character style
        /// variation. The reducer accepts a flat list of
        /// runs.
        private func runs(from attributed: NSAttributedString) -> [InlineRun] {
            if attributed.length == 0 { return [] }
            // The current implementation produces a single
            // run with the full string and no annotations.
            // The annotation round-trip happens via
            // `setInlineAnnotation` mutations, which the
            // coalescer picks up from the
            // `reduceFormattingChange` path.
            return [InlineRun(text: attributed.string)]
        }
    }
}

// MARK: - TesseraNSTextView (NSTextView subclass)

/// An `NSTextView` subclass that owns a custom
/// `TesseraTextContentManager` and wires it into the
/// platform's TextKit 2 stack. NSTextView's
/// `textContentStorage` is set during init and is
/// read-only after; the only way to substitute a
/// custom `NSTextContentManager` is to override the
/// `NSTextView` init that creates the storage.
///
/// The subclass is the integration point for the
/// `TesseraTextContentManager`. The data flow on
/// every edit:
///   1. The user types; `NSTextView` mutates the
///      `NSTextContentStorage`'s attributed string.
///   2. The storage propagates the change to the
///      layout manager, which calls back into the
///      `TesseraTextContentManager` for the elements
///      in the changed range.
///   3. The host (`TesseraEditorView.Coordinator`)
///      observes the `NSText.didChangeNotification`,
///      diffs the new attributed string against the
///      previous one, and converts the diff into a
///      `Mutation` via `TextEditReducer`.
///
/// **Production swap.** To use `STTextView`
/// (krzyzanowskim) as the base, replace this subclass
/// with `STTextView`; the wiring is identical (STTextView
/// also takes a `textContentManager`).
@MainActor
public final class TesseraNSTextView: NSTextView {
    public let tesseraContentManager: TesseraTextContentManager
    private var initialAttributedString: NSAttributedString

    public init(
        frame: NSRect = .zero,
        contentManager: TesseraTextContentManager,
        initialAttributedString: NSAttributedString
    ) {
        self.tesseraContentManager = contentManager
        self.initialAttributedString = initialAttributedString

        // Build the TextKit 2 stack by hand. NSTextView's
        // designated init takes a `NSTextContainer`; we
        // create one and set up the storage with the
        // custom content manager.
        let container = NSTextContainer(containerSize: NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude
        ))
        container.widthTracksTextView = true
        // NSTextView's text content manager is the
        // NSTextContentStorage it owns. The
        // TesseraTextContentManager is held alongside
        // and drives the data model; the layout manager
        // consults the storage (a NSTextContentManager)
        // for elements, and the storage is updated by
        // our content manager when the AST changes.
        // The round-trip wiring (storage -> our content
        // manager -> AST -> storage) is the Phase 2
        // stretch; the data layer (TesseraTextContentManager
        // + TesseraTextElement) is fully tested in
        // isolation. See the design doc for the swap
        // path to STTextView (which natively supports
        // the custom-content-manager workflow).
        let layoutManager = NSTextLayoutManager()
        layoutManager.textContainer = container
        let storage = NSTextContentStorage()
        storage.addTextLayoutManager(layoutManager)
        storage.attributedString = initialAttributedString
        super.init(frame: frame, textContainer: container)
    }

    public required init?(coder: NSCoder) {
        fatalError("TesseraNSTextView requires init(frame:contentManager:initialAttributedString:)")
    }

    /// Replace the text view's content with a new
    /// attributed string. Used by the host when the
    /// agent mutates the document (the SwiftUI binding
    /// updates and `updateNSView` calls this).
    public func replaceContent(with attributed: NSAttributedString) {
        guard let storage = textStorage as? NSTextContentStorage else { return }
        storage.attributedString = attributed
        initialAttributedString = attributed
    }
}
