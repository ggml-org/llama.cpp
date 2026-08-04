import SwiftUI
#if canImport(AppKit)
import AppKit
#endif

/// Three-page first-run onboarding. Shown once, gated by an @AppStorage
/// flag. Layout adapts: wider on macOS, full screen on iOS (design doc 5.10).
public struct OnboardingView: View {
    @AppStorage(TesseraSettingsKey.onboardingComplete) private var onboardingComplete = false
    @AppStorage(TesseraSettingsKey.modelDirectory) private var modelDirectory = TesseraSettingsDefault.modelDirectory
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var page = 0

    public var onComplete: () -> Void

    public init(onComplete: @escaping () -> Void = {}) {
        self.onComplete = onComplete
    }

    private let pageCount = 3

    public var body: some View {
        VStack(spacing: 24) {
            Spacer(minLength: 12)
            content
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            pageDots
            controls
                .padding(.bottom, 24)
        }
        .padding(.horizontal, 32)
        #if os(macOS)
        .frame(minWidth: 560, minHeight: 460)
        #endif
    }

    @ViewBuilder
    private var content: some View {
        switch page {
        case 0: welcomePage
        case 1: modelPage
        default: agentPage
        }
    }

    private var welcomePage: some View {
        VStack(spacing: 16) {
            Image(systemName: "square.grid.3x3.topleft.filled")
                .font(.system(size: 56))
                .foregroundStyle(.purple)
            Text("Welcome to Tessera Studio")
                .font(.largeTitle.bold())
            Text("Quantize, calibrate, and deploy LLMs for the Apple Neural Engine - from corpus to device.")
                .font(.title3)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            VStack(alignment: .leading, spacing: 8) {
                feature("slider.horizontal.3", "Calibrate", "Per-tensor imatrix activation statistics.")
                feature("point.topleft.down.curvedto.point.bottomright.up", "Evolve", "AWQ genetic search for optimal policies.")
                feature("gauge.with.needle", "Evaluate", "Perplexity, latency, and ANE power.")
                feature("cpu", "Deploy", "CoreML .mlmodelc for on-device inference.")
            }
            .padding(.top, 8)
        }
    }

    private var modelPage: some View {
        VStack(spacing: 16) {
            Image(systemName: "cube.box")
                .font(.system(size: 56))
                .foregroundStyle(.blue)
            Text("Set Up Models")
                .font(.largeTitle.bold())
            Text("Tessera scans a directory for .gguf and .mlmodelc models.")
                .font(.title3)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            VStack(alignment: .leading, spacing: 8) {
                Text("Model directory")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                HStack(spacing: 6) {
                    TextField("~/Models/tessera", text: $modelDirectory)
                        .textFieldStyle(.roundedBorder)
                    #if canImport(AppKit)
                    Button("Browse…") { browseModelDirectory() }
                        .accessibilityLabel("Browse for the model directory")
                    #endif
                }
            }
            .frame(maxWidth: 420)
        }
    }

    #if canImport(AppKit)
    /// Folder-picker companion for the model directory field, same
    /// pattern as Settings' PathField.browse: free typing still works,
    /// but HIG 2.13 also wants a real picker.
    private func browseModelDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        // Start the panel where the current value points, when it
        // names an existing path - saves re-navigating from $HOME.
        if !modelDirectory.isEmpty {
            let expanded = (modelDirectory as NSString).expandingTildeInPath
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: expanded, isDirectory: &isDir) {
                panel.directoryURL = URL(fileURLWithPath: isDir.boolValue
                    ? expanded
                    : (expanded as NSString).deletingLastPathComponent)
            }
        }
        if panel.runModal() == .OK, let url = panel.url {
            modelDirectory = url.path
        }
    }
    #endif

    private var agentPage: some View {
        VStack(spacing: 16) {
            Image(systemName: "bubble.left.and.text.bubble.right")
                .font(.system(size: 56))
                .foregroundStyle(.green)
            Text("Meet the Agent")
                .font(.largeTitle.bold())
            Text("The agent loop calls tools on your behalf. You control how much autonomy it gets.")
                .font(.title3)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            VStack(alignment: .leading, spacing: 8) {
                approvalRow(.auto, "Run without asking.")
                approvalRow(.notify, "Run, then notify you.")
                approvalRow(.prompt, "Ask before running.")
                approvalRow(.denied, "Never run.")
            }
            .padding(.top, 8)
        }
    }

    private func feature(_ icon: String, _ title: String, _ detail: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .frame(width: 24)
                .foregroundStyle(.purple)
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.headline)
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func approvalRow(_ level: ApprovalLevel, _ detail: String) -> some View {
        HStack(spacing: 12) {
            Text(level.rawValue.uppercased())
                .font(.caption.bold().monospaced())
                .frame(width: 72, alignment: .leading)
                .foregroundStyle(.green)
            Text(detail).font(.subheadline)
        }
    }

    private var pageDots: some View {
        HStack(spacing: 8) {
            ForEach(0..<pageCount, id: \.self) { index in
                Circle()
                    .fill(index == page ? Color.primary : Color.secondary.opacity(0.3))
                    .frame(width: 8, height: 8)
            }
        }
    }

    // HIG 2.7 / 3.6: under Reduce Motion, page turns switch instantly
    // instead of animating.
    private var pageTurnAnimation: Animation? {
        reduceMotion ? nil : .default
    }

    private var controls: some View {
        HStack {
            if page > 0 {
                Button("Back") { withAnimation(pageTurnAnimation) { page -= 1 } }
                    .buttonStyle(.bordered)
            }
            Spacer()
            // HIG 14.2: the tutorial is optional - Skip is visible on
            // every page and completes onboarding immediately.
            Button("Skip") { finish() }
                .buttonStyle(.borderless)
            Button(page == pageCount - 1 ? "Get Started" : "Continue") {
                if page == pageCount - 1 {
                    finish()
                } else {
                    withAnimation(pageTurnAnimation) { page += 1 }
                }
            }
            .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: 420)
    }

    private func finish() {
        onboardingComplete = true
        onComplete()
    }
}
