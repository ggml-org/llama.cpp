import Foundation

/// Selects the inference engine bridge. CLlama (runtime-loaded libllama) is
/// the only on-device path; the CLI bridge backs the standalone telemetry
/// view. The dead CTesseraFFI stub was removed - its isAvailable was
/// statically false - so the CLI bridge is always returned here.
public enum TesseraEngineBridgeFactory {
    public static func makeInferenceBridge() -> any TesseraEngineBridge {
        CLIEngineBridge()
    }
}
