import SwiftUI
import TesseraCore

/// The macOS Contacts surface.
///
/// **Layout:** `NavigationSplitView` with a sidebar (search
/// + filter), a list (the contact rows), and a detail (the
/// focused contact's metadata + receipt chain).
///
/// **Data:** the view reads from ``ContactStore`` (which
/// wraps ``TesseraDataLayer``). Mutations go through the
/// same store so every change is a constitutional receipt.
///
/// **Imports:** the toolbar has an "Import…" menu with one
/// entry per importer (Apple Contacts, VCard file, Google,
/// CardDAV). Each entry opens a sheet / panel that owns its
/// importer actor and calls ``ContactStore/upsert(_:)`` for
/// each parsed contact. The import flow is not v1-critical
/// (the spec lists it as a Phase 4 cross-cutting concern);
/// we ship the menu surface and a working VCard path; the
/// Google + CardDAV + Apple Contacts sheets are stubs that
/// the user can wire to the importer actors in their
/// settings flow.
public struct ContactsView: View {

    public init(store: ContactStore, importer: VCardImporter = VCardImporter()) {
        self.store = store
        self.importer = importer
    }

    private let store: ContactStore
    private let importer: VCardImporter

    @State private var contacts: [Contact] = []
    @State private var searchText: String = ""
    @State private var selectedID: UUID?
    @State private var isLoading: Bool = false
    @State private var loadError: String?
    @State private var showImportSheet: Bool = false
    @State private var importStatus: String = ""

    public var body: some View {
        NavigationSplitView {
            list
                .navigationSplitViewColumnWidth(min: 280, ideal: 320)
        } detail: {
            if let id = selectedID,
               let contact = contacts.first(where: { $0.id == id }) {
                ContactDetailView(contact: contact, store: store)
            } else {
                emptyState
            }
        }
        .navigationTitle("Contacts")
        .searchable(text: $searchText, prompt: "Search contacts")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task { await load() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .help("Reload contacts")
            }
            ToolbarItem(placement: .primaryAction) {
                Menu {
                    Button("VCard file…") {
                        showImportSheet = true
                    }
                    Divider()
                    Button("Apple Contacts (preview)") {
                        // Phase 6: the dev-preview flow is
                        // the VCard path; the entitlement-
                        // gated Apple Contacts flow is wired
                        // in a follow-up.
                    }
                    .disabled(true)
                    Button("Google (opt-in)") {
                        // Wired when the user has
                        // configured their Google OAuth
                        // credentials in Settings.
                    }
                    .disabled(true)
                    Button("CardDAV (opt-in)") {
                        // Wired when the user has
                        // configured their CardDAV
                        // credentials in Settings.
                    }
                    .disabled(true)
                } label: {
                    Label("Import", systemImage: "square.and.arrow.down")
                }
            }
        }
        .onAppear {
            if contacts.isEmpty && !isLoading {
                Task { await load() }
            }
        }
        .onChange(of: searchText) { _, _ in
            // The list is already filtered in `filteredContacts`;
            // no reload needed.
        }
        .sheet(isPresented: $showImportSheet) {
            VCardImportSheet(importer: importer, store: store) { result in
                importStatus = result
                showImportSheet = false
                Task { await load() }
            }
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
    }

    private var filteredContacts: [Contact] {
        guard !searchText.isEmpty else { return contacts }
        let q = searchText.lowercased()
        return contacts.filter { c in
            c.displayName.lowercased().contains(q) ||
            c.organization?.lowercased().contains(q) == true ||
            c.emails.contains { $0.value.lowercased().contains(q) }
        }
    }

    private var list: some View {
        List(selection: $selectedID) {
            ForEach(filteredContacts) { contact in
                ContactRow(contact: contact)
                    .tag(contact.id as UUID?)
            }
        }
        .overlay {
            if isLoading {
                ProgressView().controlSize(.large)
            } else if contacts.isEmpty {
                ContentUnavailableView(
                    "No contacts yet",
                    systemImage: "person.crop.circle.badge.plus",
                    description: Text("Use the Import menu to bring in contacts from Apple, VCard, Google, or CardDAV.")
                )
            } else if let err = loadError {
                ContentUnavailableView(
                    "Couldn't load contacts",
                    systemImage: "exclamationmark.triangle",
                    description: Text(err)
                )
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "person.crop.circle")
                .font(.system(size: 64))
                .foregroundStyle(.tertiary)
            Text("Select a contact")
                .font(.title3)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func load() async {
        isLoading = true
        loadError = nil
        defer { isLoading = false }
        do {
            let rows = try await store.list(limit: 1000)
            contacts = rows.sorted { $0.displayName.localercased() < $1.displayName.localercased() }
        } catch {
            loadError = String(describing: error)
        }
    }
}

// MARK: - Row

private struct ContactRow: View {
    let contact: Contact

    var body: some View {
        HStack(alignment: .center, spacing: 10) {
            Image(systemName: GraphNode.iconName(
                for: "contact",
                subtype: contact.subtype.rawValue
            ))
            .font(.title3)
            .foregroundStyle(GraphNode.color(for: "contact"))
            .frame(width: 28)
            VStack(alignment: .leading, spacing: 2) {
                Text(contact.displayName)
                    .font(.body)
                if let org = contact.organization, !org.isEmpty {
                    Text(org)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else if let email = contact.emails.first {
                    Text(email.value)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Detail

private struct ContactDetailView: View {
    let contact: Contact
    let store: ContactStore

    @State private var receipts: [GraphReceipt] = []
    @State private var showExportError: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                Divider()
                contactFields
                Divider()
                receiptsSection
            }
            .padding()
        }
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    exportVCard()
                } label: {
                    Image(systemName: "square.and.arrow.up")
                }
                .help("Export as VCard")
            }
        }
        .task {
            await loadReceipts()
        }
        .alert("Export failed",
               isPresented: Binding(
                get: { showExportError != nil },
                set: { if !$0 { showExportError = nil } }
               )) {
            Button("OK") { showExportError = nil }
        } message: {
            Text(showExportError ?? "")
        }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 16) {
            Image(systemName: GraphNode.iconName(
                for: "contact",
                subtype: contact.subtype.rawValue
            ))
            .font(.system(size: 48))
            .foregroundStyle(GraphNode.color(for: "contact"))
            VStack(alignment: .leading, spacing: 4) {
                Text(contact.displayName)
                    .font(.title2)
                if let org = contact.organization, !org.isEmpty {
                    Text(org)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Text(contact.subtype.rawValue.capitalized)
                    .font(.caption2)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.quaternary, in: Capsule())
            }
            Spacer()
        }
    }

    private var contactFields: some View {
        VStack(alignment: .leading, spacing: 12) {
            if !contact.emails.isEmpty {
                fieldSection("Email", rows: contact.emails.map { row(label: labelName($0.label), value: $0.value) })
            }
            if !contact.phones.isEmpty {
                fieldSection("Phone", rows: contact.phones.map { row(label: labelName($0.label), value: $0.value) })
            }
            if !contact.addresses.isEmpty {
                fieldSection("Address", rows: contact.addresses.map { row(label: labelName($0.label), value: $0.oneLine) })
            }
            if let title = contact.title, !title.isEmpty {
                Text("Title: \(title)")
                    .font(.caption)
            }
            if let birthday = contact.birthday {
                Text("Birthday: \(birthday.formatted(date: .abbreviated, time: .omitted))")
                    .font(.caption)
            }
            if let notes = contact.notes, !notes.isEmpty {
                Text(notes)
                    .font(.caption)
            }
        }
    }

    private func fieldSection(_ title: String, rows: [(String, String)]) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.subheadline)
                .fontWeight(.medium)
            ForEach(rows, id: \.0) { (label, value) in
                HStack {
                    Text(label)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(width: 80, alignment: .leading)
                    Text(value)
                        .font(.caption)
                        .textSelection(.enabled)
                }
            }
        }
    }

    private func row(label: String, value: String) -> (String, String) {
        (label, value)
    }

    private var receiptsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Receipts")
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
            receipts = try await store.receipts(forContact: contact.id)
        } catch {
            receipts = []
        }
    }

    private func exportVCard() {
        Task {
            do {
                let importer = VCardImporter()
                let data = try await importer.serialize(contacts: [contact])
                _ = try await store.exportVCard(
                    contact,
                    preEncodedVCard: data,
                    provenance: "user_explicit_export"
                )
            } catch {
                showExportError = String(describing: error)
            }
        }
    }

    private func labelName(_ label: LabeledEmail.Label) -> String {
        switch label {
        case .home: return "Home"
        case .work: return "Work"
        case .other: return "Other"
        case .custom(let s): return s
        }
    }

    private func labelName(_ label: LabeledPhone.Label) -> String {
        switch label {
        case .home: return "Home"
        case .work: return "Work"
        case .mobile: return "Mobile"
        case .main: return "Main"
        case .fax: return "Fax"
        case .other: return "Other"
        case .custom(let s): return s
        }
    }

    private func labelName(_ label: LabeledAddress.Label) -> String {
        switch label {
        case .home: return "Home"
        case .work: return "Work"
        case .billing: return "Billing"
        case .other: return "Other"
        case .custom(let s): return s
        }
    }
}

// MARK: - VCard import sheet

private struct VCardImportSheet: View {
    let importer: VCardImporter
    let store: ContactStore
    let onCompletion: (String) -> Void

    @State private var pickedFile: URL?
    @State private var isWorking: Bool = false
    @State private var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Import VCard")
                .font(.headline)
            Text("Pick a .vcf file to import. Each VCard becomes one contact.")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                Button("Choose file…") {
                    let panel = NSOpenPanel()
                    panel.allowedContentTypes = [.vCard]
                    panel.allowsMultipleSelection = false
                    if panel.runModal() == .OK, let url = panel.url {
                        pickedFile = url
                    }
                }
                if let url = pickedFile {
                    Text(url.lastPathComponent)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            if let error {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
            Spacer()
            HStack {
                Button("Cancel") { onCompletion("") }
                Spacer()
                Button("Import") {
                    importFile()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(pickedFile == nil || isWorking)
            }
        }
        .padding()
        .frame(width: 480, height: 200)
    }

    private func importFile() {
        guard let url = pickedFile else { return }
        isWorking = true
        error = nil
        Task {
            do {
                let parsed = try await importer.parse(fileURL: url)
                var imported = 0
                for contact in parsed {
                    _ = try await store.upsert(contact)
                    imported += 1
                }
                onCompletion("Imported \(imported) contact\(imported == 1 ? "" : "s").")
            } catch {
                self.error = String(describing: error)
                isWorking = false
            }
        }
    }
}

extension String {
    fileprivate func localercased() -> String { lowercased() }
}
