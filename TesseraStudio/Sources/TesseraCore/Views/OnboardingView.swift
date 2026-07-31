import SwiftUI

/// Three-page first-run onboarding. Shown once, gated by an @AppStorage
/// flag. Layout adapts: wider on macOS, full screen on iOS (design doc 5.10).
public struct OnboardingView: View {
    @AppStorage(TesseraSettingsKey.onboardingComplete) private var onboardingComplete = false
    @AppStorage(TesseraSettingsKey.modelDirectory) private var modelDirectory = TesseraSettingsDefault.modelDirectory
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
            Text("Set Up Your Models")
                .font(.largeTitle.bold())
            Text("Tessera scans a directory for .gguf and .mlmodelc models.")
                .font(.title3)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            VStack(alignment: .leading, spacing: 8) {
                Text("Model directory")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("~/Models/tessera", text: $modelDirectory)
                    .textFieldStyle(.roundedBorder)
            }
            .frame(maxWidth: 420)
            Button {
                // Placeholder: starter model download is wired in a later milestone.
            } label: {
                Label("Download a Starter Model", systemImage: "arrow.down.circle")
            }
            .buttonStyle(.bordered)
            .disabled(true)
        }
    }

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

    private var controls: some View {
        HStack {
            if page > 0 {
                Button("Back") { withAnimation { page -= 1 } }
                    .buttonStyle(.bordered)
            }
            Spacer()
            Button(page == pageCount - 1 ? "Get Started" : "Continue") {
                if page == pageCount - 1 {
                    onboardingComplete = true
                    onComplete()
                } else {
                    withAnimation { page += 1 }
                }
            }
            .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: 420)
    }
}
