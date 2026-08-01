import Foundation

/// Selects the engine bridges. The Tessera engine has three complementary
/// surfaces, deliberately not competing:
///
///   1. TesseraFFIBridge   - the "engine tool" surface (quantize, calibrate,
///                           evolve, evaluate, convert, inspect, list). Live
///                           only when tessera.xcframework is linked. The
///                           tool sites gate on TesseraFFIBridge.isAvailable
///                           and fall back to the CLI subprocess otherwise.
///   2. CLlama             - on-device inference (token generation) via dlopen
///                           of libllama. Consumed by LlamaLLMProvider, the
///                           path consolidated in A3.
///   3. CLIEngineBridge    - shells out to tessera-cli for generation when no
///                           on-device library is loaded; backs the standalone
///                           telemetry view.
///
/// makeInferenceBridge returns the generation bridge (the CLI bridge today;
/// CLlama is reached via LlamaLLMProvider, not this protocol). The engine-tool
/// operations live in the Tools layer, each gated on TesseraFFIBridge.
public enum TesseraEngineBridgeFactory {

    /// The generation bridge. CLlama-driven generation goes through
    /// LlamaLLMProvider; this bridge is the CLI-backed generation path used
    /// by the standalone telemetry view. Kept as the single inference-bridge
    /// implementation so callers compile the same way with or without the
    /// xcframework linked.
    public static func makeInferenceBridge() -> any TesseraEngineBridge {
        CLIEngineBridge()
    }

    /// A snapshot of which engine surfaces are live in this build, for
    /// diagnostics and the Settings view. The xcframework linking decision is
    /// made at Xcode build time; this reflects the result.
    public struct CapabilitySnapshot: Sendable, Equatable {
        public let ffiAvailable: Bool       // tessera.xcframework linked
        public let ffiVersion: String
        public let cllamaAvailable: Bool    // CLlama compiled with headers
        public let cliAvailable: Bool       // tessera-cli on PATH (runtime)

        public init(
            ffiAvailable: Bool = TesseraFFIBridge.isAvailable,
            ffiVersion: String = TesseraFFIBridge.version,
            cllamaAvailable: Bool = true,
            cliAvailable: Bool = true
        ) {
            self.ffiAvailable = ffiAvailable
            self.ffiVersion = ffiVersion
            self.cllamaAvailable = cllamaAvailable
            self.cliAvailable = cliAvailable
        }
    }

    /// Current capability snapshot (used by diagnostics/Settings).
    public static var capabilities: CapabilitySnapshot { CapabilitySnapshot() }
}
