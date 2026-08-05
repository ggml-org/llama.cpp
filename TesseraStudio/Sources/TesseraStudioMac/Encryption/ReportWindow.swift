#if canImport(AppKit)
import AppKit
import SwiftUI
import TesseraCore

/// The "Last wipe report..." window. Reads `~/.tessera/last-wipe.json`
/// and shows it in a read-only text view. If no wipe has been
/// recorded, shows a placeholder.
@MainActor
public final class ReportWindow: NSObject, NSWindowDelegate {
    private var window: NSWindow?
    private var hosting: NSHostingController<ReportView>?

    public override init() { super.init() }

    public func present() {
        let view = ReportView(
            load: { [weak self] in
                self?.loadReport() ?? .none
            }
        )
        let hosting = NSHostingController(rootView: view)
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 640, height: 480),
            styleMask: [.titled, .closable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Last Wipe Report"
        window.contentViewController = hosting
        window.center()
        window.delegate = self
        window.makeKeyAndOrderFront(nil)

        self.window = window
        self.hosting = hosting
    }

    public func close() {
        window?.close()
        window = nil
    }

    public func windowWillClose(_ notification: Notification) {
        if let w = self.window, notification.object as? NSWindow === w {
            window = nil
        }
    }

    private func loadReport() -> ReportView.Content {
        let store = WipeReportStore()
        do {
            if let report = try store.loadIfPresent() {
                return .present(report: report)
            } else {
                return .none
            }
        } catch {
            return .error("\(error)")
        }
    }
}

private struct ReportView: View {
    enum Content {
        case none
        case present(report: PleadTheFifthExecutor.WipeReport)
        case error(String)
    }

    let load: () -> Content

    private static let formatter: DateFormatter = {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .medium
        return f
    }()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            switch load() {
            case .none:
                Text("No wipes recorded.")
                    .font(.title3)
                    .foregroundStyle(.secondary)
                Text("~/.tessera/last-wipe.json does not exist. Once you perform a wipe, a record will appear here.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            case .error(let message):
                Text("Could not read the wipe report.")
                    .font(.title3)
                    .foregroundStyle(.red)
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            case .present(let report):
                Text("Last wipe report")
                    .font(.title3)
                HStack(spacing: 18) {
                    LabeledContent("Started",
                                   value: Self.formatter.string(from: report.startedAt))
                    LabeledContent("Completed",
                                   value: Self.formatter.string(from: report.completedAt))
                    LabeledContent("Trigger",
                                   value: report.triggerSource.rawValue)
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                Divider()
                ScrollView {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(Array(report.steps.enumerated()), id: \.offset) { _, step in
                            stepRow(step)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            Spacer()
        }
        .padding(20)
        .frame(minWidth: 560, minHeight: 360)
    }

    @ViewBuilder
    private func stepRow(_ step: PleadTheFifthExecutor.WipeStep) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: step.outcome == .success
                  ? "checkmark.circle.fill"
                  : "exclamationmark.triangle.fill")
                .foregroundStyle(step.outcome == .success ? .green : .orange)
            VStack(alignment: .leading, spacing: 2) {
                Text(step.name).font(.system(.body, design: .monospaced))
                if let reason = step.reason {
                    Text(reason)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Spacer()
            Text("\(step.durationMs) ms")
                .font(.caption)
                .foregroundStyle(.secondary)
                .monospacedDigit()
        }
    }
}
#endif
