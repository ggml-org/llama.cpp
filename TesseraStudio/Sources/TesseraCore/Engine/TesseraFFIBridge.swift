import Foundation
import CTesseraFFI

/// Swift wrapper for the native Tessera engine C FFI
/// (CTesseraFFI/tessera_ffi.h).
///
/// Layered with CLlama and the CLI bridge:
///   - TesseraFFIBridge handles quantize/calibrate/evolve/evaluate/convert/
///     inspect/list - the "engine tool" surface. It is only live when
///     tessera.xcframework is linked; otherwise isAvailable is false and the
///     tools fall back to the CLI subprocess.
///   - CLlama handles on-device inference (token generation) via dlopen of
///     libllama. It is the complementary inference path and is never used
///     for the engine-tool operations here.
///
/// See Sources/CTesseraFFI/tessera_ffi.c for the unbuilt-default stub and
/// TesseraStudio/ffi/tessera_ffi.cpp for the real implementation.
public enum TesseraFFIBridge {

    /// True only when the real C++ engine is linked (the xcframework).
    /// The SwiftPM stub returns 0, so isAvailable is false in `swift build`
    /// and `swift test`, which is what forces the CLI fallback there.
    public static var isAvailable: Bool {
        tessera_ffi_is_available() != 0
    }

    /// Static version string from the linked engine ("tessera-1.0.0-cpp"
    /// for the real impl, "tessera-ffi-stub" for the SwiftPM stub).
    public static var version: String {
        if let c = tessera_ffi_version() {
            return String(cString: c)
        }
        return "unknown"
    }

    // MARK: - Result type

    /// Outcome of an FFI operation. The int-returning C calls use 0 for
    /// success and a positive code for "valid request, not runnable via FFI"
    /// (so the caller falls back to the CLI); negatives are real errors.
    public enum Outcome: Sendable, Equatable {
        /// The FFI completed the operation.
        case success(output: String)
        /// The request was valid but the FFI cannot run it (no loaded model
        /// context, or the xcframework is not linked). Use the CLI bridge.
        case fallbackToCLI
        /// A hard error (bad arguments, I/O failure). `code` is the C return.
        case error(code: Int32, message: String)
    }

    // MARK: - Operations

    /// Quantize a GGUF model. `config` is merged into the dispatch params.
    public static func quantize(
        model modelPath: String,
        output outputPath: String,
        config: [String: Any] = [:]
    ) -> Outcome {
        let configJSON = serializeConfig(config)
        let code = tessera_quantize(modelPath, outputPath, configJSON)
        return intOutcome(code, success: "Quantization complete via FFI.")
    }

    /// Run imatrix calibration over a corpus or .npz file.
    public static func calibrate(
        model modelPath: String,
        corpus corpusPath: String,
        config: [String: Any] = [:]
    ) -> Outcome {
        let configJSON = serializeConfig(config)
        let code = tessera_calibrate(modelPath, corpusPath, configJSON)
        return intOutcome(code, success: "Calibration complete via FFI.")
    }

    /// Run AWQ-evolve policy search. Returns .fallbackToCLI in the unbuilt
    /// stub and in the real impl (the GA needs a loaded model context).
    public static func evolve(
        model modelPath: String,
        config: [String: Any] = [:]
    ) -> Outcome {
        let configJSON = serializeConfig(config)
        let code = tessera_evolve(modelPath, configJSON)
        return intOutcome(code, success: "Evolution complete via FFI.")
    }

    /// Evaluate a model. Returns the JSON result string on success.
    public static func evaluate(
        model modelPath: String,
        config: [String: Any] = [:]
    ) -> Outcome {
        let configJSON = serializeConfig(config)
        if let s = tessera_evaluate(modelPath, configJSON) {
            let str = String(cString: s)
            tessera_free_string(s)
            // The real impl always needs a forward pass; the JSON it returns
            // is the "use the CLI" marker. Decode the ok flag to decide.
            if let payload = str.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: payload) as? [String: Any],
               let ok = obj["ok"] as? Bool, ok {
                return .success(output: str)
            }
            return .fallbackToCLI
        }
        return .fallbackToCLI
    }

    /// Convert a Tessera GGUF to CoreML. Returns .fallbackToCLI when the FFI
    /// cannot dequantize weight tensors without a loaded model context.
    public static func convert(
        model modelPath: String,
        output outputPath: String,
        format: String
    ) -> Outcome {
        let code = tessera_convert(modelPath, outputPath, format)
        return intOutcome(code, success: "Conversion complete via FFI.")
    }

    /// Inspect a sidecar file. Returns its contents as a JSON string.
    public static func inspectSidecar(path: String) -> Outcome {
        if let s = tessera_inspect_sidecar(path) {
            let str = String(cString: s)
            tessera_free_string(s)
            if let payload = str.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: payload) as? [String: Any],
               let ok = obj["ok"] as? Bool {
                return ok
                    ? .success(output: str)
                    : .error(code: -1, message: (obj["error"] as? String) ?? "inspect failed")
            }
            return .fallbackToCLI
        }
        return .fallbackToCLI
    }

    /// List .gguf models in a directory. Returns a JSON array string.
    public static func listModels(dir: String) -> Outcome {
        if let s = tessera_list_models(dir) {
            let str = String(cString: s)
            tessera_free_string(s)
            // The real impl returns a JSON array; the stub returns the
            // unavailable marker object. Distinguish by decoding.
            if let payload = str.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: payload) as? [String: Any],
               obj["ok"] != nil {
                return .fallbackToCLI
            }
            return .success(output: str)
        }
        return .fallbackToCLI
    }

    // MARK: - Helpers

    /// Map a C int return to an Outcome: 0 -> success, positive -> CLI
    /// fallback, negative -> error. Internal so TesseraEngineContext can
    /// reuse the same mapping for the model-context path.
    static func intOutcome(_ code: Int32, success: String) -> Outcome {
        if code == 0 { return .success(output: success) }
        if code > 0 { return .fallbackToCLI }
        switch code {
        case -2: return .error(code: code, message: "malformed config_json")
        default: return .error(code: code, message: "FFI call failed")
        }
    }

    /// Serialize the Swift config dictionary to a JSON string for the C calls.
    /// Empty/invalid configs become empty strings (the C side treats null and
    /// "" the same: use defaults). Internal so TesseraEngineContext can
    /// reuse the serialiser.
    static func serializeConfig(_ config: [String: Any]) -> String {
        guard !config.isEmpty,
              let data = try? JSONSerialization.data(withJSONObject: config),
              let str = String(data: data, encoding: .utf8) else {
            return ""
        }
        return str
    }

    // MARK: - Model-context path (header added 2026-08)
    //
    // The handle-based variants run in-process against a loaded model
    // context. They delegate to TesseraEngineContext (the actor that owns
    // the underlying C++ objects) so the native side never sees
    // concurrent calls. The *_model() operations return .fallbackToCLI
    // while the engine wiring is incomplete; the real implementation will
    // turn the same calls into .success(output:) responses.

    /// Load a model via the native FFI. Throws TesseraEngineError on
    /// failure (engine not available, GGUF parse error, ...).
    public static func loadModel(path: String) async throws -> TesseraModelHandle {
        try await TesseraEngineContext.shared.loadModel(path: path)
    }

    /// Release a handle. Safe to call with a handle that was already
    /// freed; the actor no-ops in that case.
    public static func freeModel(handle: TesseraModelHandle) async {
        await TesseraEngineContext.shared.free(handle: handle)
    }

    /// AWQ evolve against the loaded model. The call runs inside the
    /// engine context actor so it serialises with concurrent free() and
    /// load() calls on the same model.
    public static func evolveModel(
        handle: TesseraModelHandle,
        config: [String: Any] = [:]
    ) async -> Outcome {
        await TesseraEngineContext.shared.evolve(handle: handle, config: config)
    }

    /// Perplexity / KL forward probe against the loaded model. Returns
    /// the structured JSON on success; the .fallbackToCLI case carries
    /// the engine's TODO note so the UI can show it.
    public static func evaluateModel(
        handle: TesseraModelHandle,
        config: [String: Any] = [:]
    ) async -> Outcome {
        await TesseraEngineContext.shared.evaluate(handle: handle, config: config)
    }

    /// Convert the loaded model to the named format.
    public static func convertModel(
        handle: TesseraModelHandle,
        output outputPath: String,
        format: String
    ) async -> Outcome {
        await TesseraEngineContext.shared.convert(
            handle: handle, outputPath: outputPath, format: format
        )
    }
}
