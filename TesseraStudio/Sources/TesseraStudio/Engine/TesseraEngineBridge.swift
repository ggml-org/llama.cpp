import Foundation

/// Protocol for the Tessera engine backend. The production
/// implementation uses the C FFI (tessera.xcframework); the
/// placeholder shells out to CLI tools via ProcessRunner.
protocol TesseraEngineBridge: Sendable {
    /// Load a model into the engine.
    func loadModel(
        ggufPath: String,
        sidecarPath: String?,
        runtime: TesseraRuntime,
        contextLength: Int
    ) async throws

    /// Generate tokens from a prompt, streaming results.
    func generate(
        prompt: String,
        maxTokens: Int
    ) -> AsyncThrowingStream<GeneratedToken, Error>

    /// Whether a model is currently loaded.
    var isModelLoaded: Bool { get }

    /// The name of the currently loaded model.
    var loadedModelName: String? { get }

    /// Unload the current model and free resources.
    func unloadModel() async
}

/// A single generated token from the engine.
struct GeneratedToken: Sendable {
    let text: String
    let tokenID: Int32
    let latencyMs: Double
}

/// Placeholder engine bridge that shells out to the tessera CLI.
/// Replace with the real C FFI bridge when tessera.xcframework is available.
final class CLIEngineBridge: TesseraEngineBridge, @unchecked Sendable {
    private let runner: ProcessRunner
    private let lock = NSLock()
    private var _loadedModel: String?

    init(runner: ProcessRunner = ProcessRunner()) {
        self.runner = runner
    }

    var isModelLoaded: Bool {
        lock.withLock { _loadedModel != nil }
    }

    var loadedModelName: String? {
        lock.withLock { _loadedModel }
    }

    func loadModel(
        ggufPath: String,
        sidecarPath: String?,
        runtime: TesseraRuntime,
        contextLength: Int
    ) async throws {
        // Validate the model file exists
        let expanded = NSString(string: ggufPath).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: expanded) else {
            throw EngineBridgeError.modelNotFound(expanded)
        }

        // In production: call tessera_context_init via C FFI.
        // Placeholder: just record the path.
        lock.withLock { _loadedModel = expanded }
    }

    func generate(
        prompt: String,
        maxTokens: Int
    ) -> AsyncThrowingStream<GeneratedToken, Error> {
        guard let model = loadedModelName else {
            return AsyncThrowingStream { continuation in
                continuation.finish(throwing: EngineBridgeError.noModelLoaded)
            }
        }

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    let stream = self.runner.runStreaming(
                        executable: "tessera-cli",
                        arguments: [
                            "--model", model,
                            "--prompt", prompt,
                            "--n-predict", String(maxTokens),
                            "--stream",
                        ]
                    )
                    for try await line in stream {
                        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !trimmed.isEmpty else { continue }
                        continuation.yield(GeneratedToken(
                            text: trimmed,
                            tokenID: 0,
                            latencyMs: 0
                        ))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    func unloadModel() async {
        lock.withLock { _loadedModel = nil }
    }
}

enum EngineBridgeError: Error, LocalizedError {
    case modelNotFound(String)
    case noModelLoaded
    case ffiUnavailable

    var errorDescription: String? {
        switch self {
        case .modelNotFound(let path): "Model not found: \(path)"
        case .noModelLoaded: "No model is loaded. Call loadModel first."
        case .ffiUnavailable: "The C FFI bridge is not available. Use CLI mode."
        }
    }
}
