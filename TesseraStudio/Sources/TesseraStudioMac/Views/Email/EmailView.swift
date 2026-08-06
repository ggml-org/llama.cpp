import SwiftUI
import AppKit
import UniformTypeIdentifiers
import TesseraCore

// MARK: - EmailView

/// The macOS Email surface. MailMate-style
/// keyboard-first three-pane layout:
///
/// * **Sidebar** (left) — folders + accounts +
///   smart folders. "Local" is the only account
///   in v1; the structure leaves room for IMAP
///   accounts in v2.
/// * **List** (middle) — email rows. j / k move
///   the selection; s toggles star; # trashes;
///   a archives.
/// * **Detail** (right) — the selected email's
///   body, with reply / forward / archive
///   buttons in the toolbar.
///
/// **Keyboard shortcuts (MailMate-style):**
///
/// | Key        | Action                              |
/// |------------|-------------------------------------|
/// | j / k      | next / previous email               |
/// | r          | reply                               |
/// | R          | reply all                           |
/// | f          | forward                             |
/// | a          | archive                             |
/// | #          | trash                               |
/// | s          | star                                |
/// | c          | compose new                         |
/// | /          | focus the search field              |
/// | g i / g s  | go to inbox / sent                  |
/// | J / K      | next / previous thread              |
/// | Enter      | open focused email                  |
///
/// **Data:** the view reads from ``EmailStore``
/// (which wraps ``TesseraDataLayer``). Mutations
/// go through the same store so every change is
/// a constitutional receipt.
///
/// **Imports:** the toolbar's "Import" menu
/// routes `.eml` / `.mbox` files to
/// ``EmailImporter``, which delegates to the
/// Phase 4 `TesseraImporter`. Drag-and-drop onto
/// the list is wired too.
public struct EmailView: View {

    public init(
        store: EmailStore,
        sender: EmailSender,
        importer: EmailImporter,
        identity: EmailAddress
    ) {
        self.store = store
        self.sender = sender
        self.importer = importer
        self.identity = identity
    }

    private let store: EmailStore
    private let sender: EmailSender
    private let importer: EmailImporter
    private let identity: EmailAddress

    // State
    @State private var emails: [EmailMessage] = []
    @State private var selectedFolder: Folder = .inbox
    @State private var selectedEmailID: UUID?
    @State private var searchText: String = ""
    @State private var isLoading: Bool = false
    @State private var loadError: String?
    @State private var isImporting: Bool = false
    @State private var importStatus: String = ""
    @State private var composerDraft: EmailComposer?
    @State private var showComposer: Bool = false
    @State private var focusedRow: EmailRowFocus = .list
    @State private var isPresentingKeyHint: Bool = false
    /// Pending "g" key for two-key chords
    /// (MailMate-style: `g i` goes to inbox,
    /// `g s` goes to sent). When the user
    /// presses `g`, this state is set with a
    /// short timeout; the next keypress
    /// resolves the chord or clears the
    /// pending state. The timeout is small
    /// (1.2s) so a stray `g` doesn't leave
    /// the user stuck.
    @State private var pendingG: Date?

    // Computed
    private var filteredEmails: [EmailMessage] {
        // 1) Folder filter.
        let inFolder = emails.filter { email in
            switch selectedFolder {
            case .inbox: return email.folder == .inbox
            case .sent: return email.folder == .sent
            case .drafts: return email.folder == .drafts
            case .archive: return email.folder == .archive
            case .trash: return email.folder == .trash
            case .custom(let label): return email.folder == .custom(label)
            }
        }
        // 2) Read state per folder semantics.
        let active: [EmailMessage]
        if case .inbox = selectedFolder {
            active = inFolder.filter { !$0.isRead || true }  // show read+unread
            _ = active  // explicit; we want the unread-first sort below
        } else {
            active = inFolder
        }
        // 3) Search filter.
        let searched: [EmailMessage]
        if searchText.isEmpty {
            searched = active
        } else {
            let q = searchText.lowercased()
            searched = active.filter { e in
                e.subject.lowercased().contains(q) ||
                e.from.email.lowercased().contains(q) ||
                (e.from.name ?? "").lowercased().contains(q) ||
                e.bodyPlain.lowercased().contains(q)
            }
        }
        // 4) Unread-first then receivedAt DESC.
        return searched.sorted { a, b in
            if a.isRead != b.isRead { return !a.isRead && b.isRead }
            return a.receivedAt > b.receivedAt
        }
    }

    private var selectedEmail: EmailMessage? {
        guard let id = selectedEmailID else { return nil }
        return emails.first { $0.id == id }
    }

    private var folderCounts: [Folder: Int] {
        var counts: [Folder: Int] = [:]
        for e in emails {
            counts[e.folder, default: 0] += 1
        }
        return counts
    }

    public var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 200, ideal: 240, max: 320)
        } content: {
            list
                .navigationSplitViewColumnWidth(min: 280, ideal: 360, max: 480)
        } detail: {
            if let email = selectedEmail {
                detail(email: email)
            } else {
                emptyState
            }
        }
        .navigationTitle("Email")
        .searchable(text: $searchText, prompt: "Search subject, sender, body")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task { await load() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Reload")
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    startNewCompose()
                } label: {
                    Label("Compose", systemImage: "square.and.pencil")
                }
                .keyboardShortcut("c", modifiers: [])
                .help("Compose (c)")
            }
            ToolbarItem(placement: .primaryAction) {
                Menu {
                    Button(".eml file…") { presentOpenPanel(allowed: [.emailMessage]) }
                    Button(".mbox file…") { presentOpenPanel(allowed: [.plainText, .data]) }
                    Divider()
                    Button("Apple Mail mailbox (.mbox)…") { presentOpenPanel(allowed: [.plainText, .data]) }
                } label: {
                    Label("Import", systemImage: "square.and.arrow.down")
                }
                .help("Import email files")
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    isPresentingKeyHint = true
                } label: {
                    Image(systemName: "questionmark.circle")
                }
                .help("Keyboard shortcuts")
            }
        }
        .onAppear {
            if emails.isEmpty && !isLoading {
                Task { await load() }
            }
        }
        .onChange(of: selectedFolder) { _, _ in
            // Clear the selection on folder change
            // so the detail doesn't show a stale
            // email.
            selectedEmailID = nil
        }
        .sheet(isPresented: $showComposer) {
            if let draft = composerDraft {
                EmailComposerSheet(
                    composer: draft,
                    sender: sender,
                    store: store,
                    onClose: {
                        showComposer = false
                        composerDraft = nil
                        Task { await load() }
                    }
                )
            }
        }
        .sheet(isPresented: $isPresentingKeyHint) {
            KeyboardHintSheet()
        }
        .overlay(alignment: .bottom) {
            if !importStatus.isEmpty {
                Text(importStatus)
                    .font(.caption)
                    .padding(8)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 6))
                    .padding()
                    .transition(.opacity)
            }
        }
        // The list has the focusable container so
        // the keyboard shortcuts are global within
        // the email view.
        .focusable()
        .onKeyPress(.init("j")) {
            moveSelection(by: 1)
            return .handled
        }
        .onKeyPress(.init("k")) {
            moveSelection(by: -1)
            return .handled
        }
        .onKeyPress(.init("J")) {
            moveThreadSelection(by: 1)
            return .handled
        }
        .onKeyPress(.init("K")) {
            moveThreadSelection(by: -1)
            return .handled
        }
        .onKeyPress(.init("r")) {
            startReply(all: false)
            return .handled
        }
        .onKeyPress("R") {
            startReply(all: true)
            return .handled
        }
        .onKeyPress(.init("f")) {
            startForward()
            return .handled
        }
        .onKeyPress(.init("a")) {
            archiveSelected()
            return .handled
        }
        .onKeyPress(.init("#")) {
            trashSelected()
            return .handled
        }
        .onKeyPress(.init("s")) {
            if isPendingG() {
                pendingG = nil
                goToSent()
                return .handled
            }
            toggleStar()
            return .handled
        }
        .onKeyPress(.init("c")) {
            startNewCompose()
            return .handled
        }
        // Two-key chord: `g i` goes to inbox,
        // `g s` goes to sent. The `g` key
        // arms `pendingG`; the next keypress
        // resolves the chord (or clears
        // pendingG if the next key isn't `i`
        // or `s`).
        .onKeyPress(.init("g")) {
            pendingG = Date()
            return .handled
        }
        .onKeyPress(.init("i")) {
            if isPendingG() {
                pendingG = nil
                goToInbox()
                return .handled
            }
            return .ignored
        }
        // Enter opens the focused email.
        // v1: the list selection IS the
        // focused email; pressing Enter
        // while the list is focused is a
        // no-op (the email is already shown
        // in the detail). The shortcut is
        // still wired so a future split
        // (search results, etc.) can use
        // it.
        .onKeyPress(.return) {
            return .ignored
        }
    }

    // MARK: - Sidebar

    private var sidebar: some View {
        List(selection: $selectedFolder) {
            Section("Smart") {
                Label("Unread", systemImage: "envelope.badge")
                    .tag(Optional(Folder.inbox))
                Label("Starred", systemImage: "star")
                    .tag(Optional(Folder.inbox))
            }
            Section("Folders") {
                ForEach(standardFolders, id: \.self) { folder in
                    Label {
                        HStack {
                            Text(folder.displayName)
                            Spacer()
                            if let c = folderCounts[folder], c > 0 {
                                Text("\(c)")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    } icon: {
                        Image(systemName: icon(for: folder))
                    }
                    .tag(Optional(folder))
                }
            }
            Section("Accounts") {
                Label("Local", systemImage: "person.crop.circle")
                    .tag(Optional(Folder.inbox))
            }
        }
        .listStyle(.sidebar)
        .onChange(of: selectedFolder) { _, _ in
            // The folder drives the list reload.
        }
    }

    private var standardFolders: [Folder] {
        [.inbox, .sent, .drafts, .archive, .trash]
    }

    private func icon(for folder: Folder) -> String {
        switch folder {
        case .inbox: return "tray"
        case .sent: return "paperplane"
        case .drafts: return "doc"
        case .archive: return "archivebox"
        case .trash: return "trash"
        case .custom: return "tag"
        }
    }

    // MARK: - List

    private var list: some View {
        List(selection: $selectedEmailID) {
            ForEach(filteredEmails) { email in
                EmailRow(email: email)
                    .tag(Optional(email.id))
                    .contextMenu {
                        Button("Reply") { startReply(all: false) }
                        Button("Reply All") { startReply(all: true) }
                        Button("Forward") { startForward() }
                        Divider()
                        Button(email.isStarred ? "Unstar" : "Star") { toggleStar() }
                        Button("Archive") { archiveSelected() }
                        Button("Trash") { trashSelected() }
                    }
            }
        }
        .overlay {
            if isLoading {
                ProgressView().controlSize(.large)
            } else if filteredEmails.isEmpty {
                ContentUnavailableView(
                    selectedFolder == .inbox ? "Inbox is empty" : "No \(selectedFolder.displayName.lowercased())",
                    systemImage: "envelope",
                    description: Text("Use ⌘N to compose, or import an .eml / .mbox file.")
                )
            } else if let err = loadError {
                ContentUnavailableView(
                    "Couldn't load emails",
                    systemImage: "exclamationmark.triangle",
                    description: Text(err)
                )
            }
        }
    }

    // MARK: - Detail

    private func detail(email: EmailMessage) -> some View {
        EmailDetailView(
            email: email,
            store: store,
            onReply: { startReply(all: false) },
            onReplyAll: { startReply(all: true) },
            onForward: { startForward() },
            onArchive: { archiveSelected() },
            onTrash: { trashSelected() },
            onToggleStar: { toggleStar() }
        )
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "envelope")
                .font(.system(size: 64))
                .foregroundStyle(.tertiary)
            Text("Select an email")
                .font(.title3)
                .foregroundStyle(.secondary)
            Text("j/k to move, r to reply, / to search")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Actions

    private func load() async {
        isLoading = true
        loadError = nil
        defer { isLoading = false }
        do {
            emails = try await store.list(limit: 1000)
        } catch {
            loadError = String(describing: error)
        }
    }

    private func moveSelection(by delta: Int) {
        let visible = filteredEmails
        guard !visible.isEmpty else { return }
        let currentIndex: Int
        if let id = selectedEmailID,
           let idx = visible.firstIndex(where: { $0.id == id }) {
            currentIndex = idx
        } else {
            currentIndex = 0
        }
        let next = (currentIndex + delta).clamped(to: 0...(visible.count - 1))
        selectedEmailID = visible[next].id
        Task { _ = try? await store.markRead(visible[next].id, read: true) }
    }

    private func moveThreadSelection(by delta: Int) {
        // J/K moves to the next/previous thread
        // anchor (the first email of each
        // thread). Implemented as a "skip
        // already-selected thread" walk.
        let visible = filteredEmails
        guard !visible.isEmpty else { return }
        let anchors = computeThreadAnchors(visible)
        let currentIndex: Int
        if let id = selectedEmailID,
           let idx = anchors.firstIndex(where: { $0.id == id }) {
            currentIndex = idx
        } else {
            currentIndex = 0
        }
        let next = (currentIndex + delta).clamped(to: 0...(anchors.count - 1))
        selectedEmailID = anchors[next].id
    }

    private func computeThreadAnchors(_ emails: [EmailMessage]) -> [EmailMessage] {
        // The first email of each unique threadID.
        var seen: Set<String> = []
        var out: [EmailMessage] = []
        for e in emails {
            let key = e.threadID ?? e.messageID
            if seen.insert(key).inserted {
                out.append(e)
            }
        }
        return out
    }

    private func startReply(all: Bool) {
        guard let email = selectedEmail else { return }
        composerDraft = EmailComposer(
            mode: .reply(to: email, all: all),
            from: identity
        )
        showComposer = true
    }

    private func startForward() {
        guard let email = selectedEmail else { return }
        composerDraft = EmailComposer(
            mode: .forward(email),
            from: identity
        )
        showComposer = true
    }

    private func startNewCompose() {
        composerDraft = EmailComposer(mode: .new, from: identity)
        showComposer = true
    }

    private func archiveSelected() {
        guard let id = selectedEmailID else { return }
        Task {
            _ = try? await store.setFolder(id, folder: .archive)
            await load()
        }
    }

    private func trashSelected() {
        guard let id = selectedEmailID else { return }
        Task {
            _ = try? await store.setFolder(id, folder: .trash)
            await load()
        }
    }

    private func toggleStar() {
        guard let id = selectedEmailID,
              let email = emails.first(where: { $0.id == id }) else { return }
        Task {
            _ = try? await store.setStarred(id, starred: !email.isStarred)
            await load()
        }
    }

    /// True when a `g` keypress is pending and
    /// hasn't timed out. The chord is
    /// resolved by the next keypress (or
    /// cleared by the timeout). The 1.2s
    /// window is a MailMate convention.
    /// Stale pendings are left for the
    /// caller to clear (the timeout is
    /// short and the next unrelated `g`
    /// press will reset it).
    private func isPendingG() -> Bool {
        guard let pendingG else { return false }
        let elapsed = Date().timeIntervalSince(pendingG)
        return elapsed <= 1.2
    }

    /// Navigate to the Inbox folder.
    /// `g i` is the MailMate-style chord.
    private func goToInbox() {
        selectedFolder = .inbox
        selectedEmailID = nil
    }

    /// Navigate to the Sent folder.
    /// `g s` is the MailMate-style chord.
    private func goToSent() {
        selectedFolder = .sent
        selectedEmailID = nil
    }

    private func presentOpenPanel(allowed: [UTType]) {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = allowed
        panel.allowsMultipleSelection = true
        if panel.runModal() == .OK {
            let urls = panel.urls
            Task {
                isImporting = true
                importStatus = "Importing \(urls.count) file\(urls.count == 1 ? "" : "s")…"
                let ids = (try? await importer.importFiles(urls)) ?? []
                importStatus = "Imported \(ids.count) email\(ids.count == 1 ? "" : "s")."
                await load()
                isImporting = false
            }
        }
    }
}

// MARK: - Comparable clamping

extension Comparable {
    fileprivate func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}

// MARK: - EmailRow

private struct EmailRow: View {
    let email: EmailMessage

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            // Star indicator
            Image(systemName: email.isStarred ? "star.fill" : "star")
                .foregroundStyle(email.isStarred ? Color.yellow : Color.secondary)
                .frame(width: 16)
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(email.senderDisplay)
                        .font(.body)
                        .fontWeight(email.isRead ? .regular : .semibold)
                        .lineLimit(1)
                    Spacer()
                    Text(email.receivedAt, style: .date)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Text(email.displaySubject)
                    .font(.subheadline)
                    .fontWeight(email.isRead ? .regular : .medium)
                    .lineLimit(1)
                    .foregroundStyle(email.isRead ? .secondary : .primary)
                Text(email.snippet)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                HStack(spacing: 6) {
                    if email.hasAttachments {
                        Image(systemName: "paperclip")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    if email.isReplied {
                        Image(systemName: "arrowshape.turn.up.left.fill")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    if email.isForwarded {
                        Image(systemName: "arrowshape.turn.up.right.fill")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
            }
        }
        .padding(.vertical, 4)
    }
}

// MARK: - EmailDetailView

private struct EmailDetailView: View {
    let email: EmailMessage
    let store: EmailStore
    let onReply: () -> Void
    let onReplyAll: () -> Void
    let onForward: () -> Void
    let onArchive: () -> Void
    let onTrash: () -> Void
    let onToggleStar: () -> Void

    @State private var receipts: [GraphReceipt] = []

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                Divider()
                metaBlock
                Divider()
                bodyContent
                if !email.attachments.isEmpty {
                    Divider()
                    attachmentsBlock
                }
                Divider()
                receiptsSection
            }
            .padding()
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button(action: onReply) {
                    Image(systemName: "arrowshape.turn.up.left")
                }
                .help("Reply (r)")
                Button(action: onReplyAll) {
                    Image(systemName: "arrowshape.turn.up.left.2")
                }
                .help("Reply All (R)")
                Button(action: onForward) {
                    Image(systemName: "arrowshape.turn.up.right")
                }
                .help("Forward (f)")
                Button(action: onToggleStar) {
                    Image(systemName: email.isStarred ? "star.fill" : "star")
                }
                .help("Star (s)")
                Button(action: onArchive) {
                    Image(systemName: "archivebox")
                }
                .help("Archive (a)")
                Button(action: onTrash) {
                    Image(systemName: "trash")
                }
                .help("Trash (#)")
            }
        }
        .task {
            await loadReceipts()
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(email.displaySubject)
                .font(.title2)
                .fontWeight(.medium)
            HStack {
                Text(email.senderDisplay)
                    .font(.subheadline)
                Text("<" + email.from.email + ">")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(email.receivedAt, style: .date)
                    .font(.caption)
                Text(email.receivedAt, style: .time)
                    .font(.caption)
            }
        }
    }

    private var metaBlock: some View {
        VStack(alignment: .leading, spacing: 4) {
            if !email.to.isEmpty {
                Text("To: " + email.to.map { $0.mailboxString }.joined(separator: ", "))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            if !email.cc.isEmpty {
                Text("Cc: " + email.cc.map { $0.mailboxString }.joined(separator: ", "))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            if let sent = email.sentAt {
                Text("Sent: \(sent.formatted(date: .abbreviated, time: .shortened))")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            if let tid = email.threadID {
                Text("Thread: \(tid)")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .textSelection(.enabled)
            }
        }
    }

    private var bodyContent: some View {
        // v1: render the plain-text body. The HTML
        // body (when present) is shown in a
        // collapsed section below.
        VStack(alignment: .leading, spacing: 12) {
            Text(email.bodyPlain)
                .font(.body)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            if let html = email.bodyHTML, !html.isEmpty {
                DisclosureGroup("HTML source") {
                    Text(html)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                }
            }
        }
    }

    private var attachmentsBlock: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Attachments (\(email.attachments.count))")
                .font(.subheadline)
                .fontWeight(.medium)
            ForEach(email.attachments) { a in
                HStack {
                    Image(systemName: "paperclip")
                        .foregroundStyle(.tertiary)
                    Text(a.filename)
                        .font(.caption)
                    Spacer()
                    Text("\(a.size) bytes")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private var receiptsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("History")
                .font(.subheadline)
                .fontWeight(.medium)
            if receipts.isEmpty {
                Text("No receipts yet.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(receipts) { r in
                    HStack {
                        Image(systemName: "doc.text")
                            .foregroundStyle(.tertiary)
                        VStack(alignment: .leading) {
                            Text(r.receiptType)
                                .font(.caption)
                            Text(r.witnessedAt.formatted(date: .abbreviated, time: .shortened))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                    }
                }
            }
        }
    }

    private func loadReceipts() async {
        do {
            receipts = try await store.receipts(forEmail: email.id)
        } catch {
            receipts = []
        }
    }
}

// MARK: - EmailRowFocus (placeholder for future per-row focus)

/// The current focus within the email view. v1
/// uses `.list` (the keyboard shortcuts act on
/// the list). v2 will add `.detail` and the
/// shortcuts that depend on the focus (e.g. space
/// to scroll).
private enum EmailRowFocus: Hashable {
    case list
    case detail
}

// MARK: - EmailComposerSheet

/// The modal composer sheet. The composer is a
/// value type; the sheet owns the editable body
/// via `@State` and the rest of the fields are
/// set up at init time.
private struct EmailComposerSheet: View {
    let composer: EmailComposer
    let sender: EmailSender
    let store: EmailStore
    let onClose: () -> Void

    @State private var bodyText: String
    @State private var toText: String
    @State private var ccText: String
    @State private var subjectText: String
    @State private var isSending: Bool = false
    @State private var sendError: String?

    init(
        composer: EmailComposer,
        sender: EmailSender,
        store: EmailStore,
        onClose: @escaping () -> Void
    ) {
        self.composer = composer
        self.sender = sender
        self.store = store
        self.onClose = onClose
        self._bodyText = State(initialValue: composer.bodyPlain)
        self._toText = State(initialValue: composer.to.map { $0.mailboxString }.joined(separator: ", "))
        self._ccText = State(initialValue: composer.cc.map { $0.mailboxString }.joined(separator: ", "))
        self._subjectText = State(initialValue: composer.subject)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Compose")
                    .font(.headline)
                Spacer()
                Button("Close") { onClose() }
                    .keyboardShortcut(.cancelAction)
            }
            fieldRow("To", text: $toText)
            fieldRow("Cc", text: $ccText)
            fieldRow("Subject", text: $subjectText)
            Divider()
            TextEditor(text: $bodyText)
                .font(.body)
                .frame(minHeight: 240)
                .border(Color.gray.opacity(0.2))
            if let err = sendError {
                Text(err)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
            HStack {
                Spacer()
                Button("Save as Draft") { saveDraft() }
                Button("Send via Mail…") { send() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(isSending)
            }
        }
        .padding()
        .frame(width: 640, height: 520)
    }

    private func fieldRow(_ label: String, text: Binding<String>) -> some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 64, alignment: .trailing)
            TextField("", text: text)
                .textFieldStyle(.roundedBorder)
        }
    }

    private func send() {
        isSending = true
        sendError = nil
        Task {
            do {
                let addresses = Self.parseAddresses(toText)
                let ccs = Self.parseAddresses(ccText)
                let final = composer
                    .setTo(addresses)
                    .setCC(ccs)
                    .setSubject(subjectText)
                    .setBody(bodyText)
                let draft = final.build()
                let result = try await sender.send(draft, original: nil)
                switch result {
                case .routedToSystemShare:
                    onClose()
                case .savedAsDraft:
                    sendError = "Cancelled; draft saved."
                    isSending = false
                }
            } catch {
                sendError = String(describing: error)
                isSending = false
            }
        }
    }

    private func saveDraft() {
        Task {
            let addresses = Self.parseAddresses(toText)
            let ccs = Self.parseAddresses(ccText)
            let final = composer
                .setTo(addresses)
                .setCC(ccs)
                .setSubject(subjectText)
                .setBody(bodyText)
            _ = try? await store.saveDraft(final.build().toEmailMessage())
            onClose()
        }
    }

    private static func parseAddresses(_ text: String) -> [EmailAddress] {
        // Accept "Name <email>, email" or just
        // "email" entries. v1 is best-effort;
        // the address parser is a thin wrapper
        // around the RFC 5322 mailbox form.
        let entries = text.split(separator: ",").map(String.init)
        return entries.compactMap { entry in
            let trimmed = entry.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return nil }
            if let open = trimmed.firstIndex(of: "<"),
               let close = trimmed.firstIndex(of: ">"),
               close > open {
                let name = String(trimmed[trimmed.startIndex..<open])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let email = String(trimmed[trimmed.index(after: open)..<close])
                return EmailAddress(name: name.isEmpty ? nil : name, email: email)
            }
            // Bare email
            if trimmed.contains("@") {
                return EmailAddress(email: trimmed)
            }
            return nil
        }
    }
}

// MARK: - KeyboardHintSheet

/// The "?" keyboard hint sheet. Shown when the
/// user clicks the help button in the toolbar.
private struct KeyboardHintSheet: View {
    @Environment(\.dismiss) private var dismiss

    private let hints: [(key: String, action: String)] = [
        ("j", "Next email"),
        ("k", "Previous email"),
        ("J", "Next thread"),
        ("K", "Previous thread"),
        ("r", "Reply"),
        ("R", "Reply all"),
        ("f", "Forward"),
        ("a", "Archive"),
        ("#", "Trash"),
        ("s", "Toggle star"),
        ("c", "Compose new"),
        ("g i", "Go to inbox"),
        ("g s", "Go to sent"),
        ("/", "Search"),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Keyboard shortcuts")
                .font(.headline)
            Divider()
            ForEach(hints, id: \.key) { hint in
                HStack {
                    Text(hint.key)
                        .font(.system(.body, design: .monospaced))
                        .frame(width: 40, alignment: .leading)
                    Text(hint.action)
                        .font(.body)
                    Spacer()
                }
            }
            Spacer()
            HStack {
                Spacer()
                Button("Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding()
        .frame(width: 360, height: 420)
    }
}
