import SwiftUI
import AppKit
import TesseraCore

// MARK: - C2PAManifestSheet

/// A sheet that shows the full C2PA manifest as JSON.
/// The manifest is the receipt's `c2paManifest` field
/// (per spec §7.4). The sheet presents a scrollable
/// monospaced text view with the JSON; the user can
/// copy it via Cmd-C.
public struct C2PAManifestSheet: View {

    public let manifest: C2PAManifest
    @Environment(\.dismiss) private var dismiss
    @State private var jsonString: String = ""

    public init(manifest: C2PAManifest) {
        self.manifest = manifest
    }

    public var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("C2PA manifest")
                    .font(.system(size: 14, weight: .semibold))
                Spacer()
                Button("Copy") {
                    let pb = NSPasteboard.general
                    pb.clearContents()
                    pb.setString(jsonString, forType: .string)
                }
                .buttonStyle(.bordered)
                Button("Done") { dismiss() }
                    .keyboardShortcut(.cancelAction)
                    .buttonStyle(.borderedProminent)
            }
            .padding(12)
            Divider()
            ScrollView {
                Text(jsonString)
                    .font(.system(size: 11, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
            }
            .background(Color(NSColor.textBackgroundColor))
        }
        .frame(minWidth: 480, idealWidth: 600, minHeight: 320)
        .onAppear { jsonString = encodeJSON() }
    }

    private func encodeJSON() -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(manifest),
              let str = String(data: data, encoding: .utf8) else {
            return "(failed to encode manifest)"
        }
        return str
    }
}
