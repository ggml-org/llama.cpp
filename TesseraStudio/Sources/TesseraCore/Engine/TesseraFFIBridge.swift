import Foundation
import CTesseraFFI

/// Errors thrown by the FFI bridge.
public enum TesseraFFIError: Error, LocalizedError {
    case unavailable
    case callFailed(operation: String, code: Int32)
    case invalidJSON(operation: String)

    public var errorDescription: String? {
        switch self {
        case .unavailable:
            "The Tessera native engine (FFI) is not linked. Falling back to CLI."
        case .callFailed(let op, let code):
            "FFI \(op) failed with code \(code)."
        case .invalidJSON(let op):
            "FFI \(op) returned invalid JSON."
        }
    }
}

/// Swift wrapper over the thin C FFI (tessera_ffi.h). Converts the C
/// JSON results into Swift types. When the native engine is not linked
/// (the stub), `isAvailable` is false and callers fall back to the CLI.
public struct TesseraFFIBridge: Sendable {
    /// Whether a real native engine is linked (false for the stub).
    public static var isAvailable: Bool {
        tessera_ffi_is_available() != 0
    }

    /// The native engine version string.
    public static var version: String {
        guard let cString = tessera_ffi_version() else { return "unknown" }
        return String(cString: cString)
    }

    // MARK: - Operations

    /// Quantize a model. Returns 0 on success; throws on failure.
    public static func quantize(
        modelPath: String,
        outputPath: String,
        config: [String: JSONValue] = [:]
    ) throws -> Int32 {
        let code = tessera_quantize(modelPath, outputPath, encodeConfig(config))
        guard code == 0 else { throw TesseraFFIError.callFailed(operation: "quantize", code: code) }
        return code
    }

    /// Run imatrix calibration. Throws on failure.
    public static func calibrate(
        modelPath: String,
        corpusPath: String,
        config: [String: JSONValue] = [:]
    ) throws -> Int32 {
        let code = tessera_calibrate(modelPath, corpusPath, encodeConfig(config))
        guard code == 0 else { throw TesseraFFIError.callFailed(operation: "calibrate", code: code) }
        return code
    }

    /// Run AWQ-evolve policy search. Throws on failure.
    public static func evolve(
        modelPath: String,
        config: [String: JSONValue] = [:]
    ) throws -> Int32 {
        let code = tessera_evolve(modelPath, encodeConfig(config))
        guard code == 0 else { throw TesseraFFIError.callFailed(operation: "evolve", code: code) }
        return code
    }

    /// Evaluate a model, returning the parsed JSON result.
    public static func evaluate(
        modelPath: String,
        config: [String: JSONValue] = [:]
    ) throws -> [String: JSONValue] {
        guard let json = takeString(tessera_evaluate(modelPath, encodeConfig(config))) else {
            throw TesseraFFIError.invalidJSON(operation: "evaluate")
        }
        guard let object = parseObject(json) else {
            throw TesseraFFIError.invalidJSON(operation: "evaluate")
        }
        return object
    }

    /// Convert a model to the named format (e.g. "coreml"). Throws on failure.
    public static func convert(
        modelPath: String,
        outputPath: String,
        format: String
    ) throws -> Int32 {
        let code = tessera_convert(modelPath, outputPath, format)
        guard code == 0 else { throw TesseraFFIError.callFailed(operation: "convert", code: code) }
        return code
    }

    /// Inspect a sidecar, returning a typed SidecarInfo.
    public static func inspectSidecar(path: String) throws -> SidecarInfo {
        guard let json = takeString(tessera_inspect_sidecar(path)),
              let object = parseObject(json) else {
            throw TesseraFFIError.invalidJSON(operation: "inspect_sidecar")
        }
        return makeSidecarInfo(from: object, path: path)
    }

    /// List models in a directory, returning file names.
    public static func listModels(directory: String) throws -> [String] {
        guard let json = takeString(tessera_list_models(directory)) else {
            throw TesseraFFIError.invalidJSON(operation: "list_models")
        }
        guard let data = json.data(using: .utf8),
              let values = try? JSONDecoder().decode([JSONValue].self, from: data) else {
            throw TesseraFFIError.invalidJSON(operation: "list_models")
        }
        return values.compactMap(\.stringValue)
    }

    // MARK: - Helpers

    private static func encodeConfig(_ config: [String: JSONValue]) -> String {
        guard let data = try? JSONEncoder().encode(config),
              let str = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return str
    }

    /// Take ownership of a C string, copy it, and free the original.
    private static func takeString(_ ptr: UnsafeMutablePointer<CChar>?) -> String? {
        guard let ptr else { return nil }
        defer { tessera_free_string(ptr) }
        return String(cString: ptr)
    }

    private static func parseObject(_ json: String) -> [String: JSONValue]? {
        guard let data = json.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode([String: JSONValue].self, from: data)
    }

    private static func makeSidecarInfo(from object: [String: JSONValue], path: String) -> SidecarInfo {
        var scales: [ModalityScale] = []
        if case .array(let items)? = object["modality_scales"] {
            for item in items {
                guard case .object(let ms) = item else { continue }
                scales.append(ModalityScale(
                    modality: ms["modality"]?.stringValue ?? "?",
                    awqAlpha: ms["awq_alpha"]?.numberValue ?? 0,
                    componentCount: ms["component_count"]?.numberValue.map { Int($0) } ?? 0
                ))
            }
        }

        let dequantRaw = object["dequant_mode"]?.stringValue ?? DequantMode.t640_3d.rawValue
        return SidecarInfo(
            modelPath: path,
            schemaVersion: object["schema_version"]?.numberValue.map { Int($0) } ?? 1,
            tesseraProfile: object["tessera_profile"]?.stringValue ?? "unknown",
            effectiveBits: object["effective_bits"]?.numberValue ?? 0,
            kernelVersion: object["kernel_version"]?.stringValue ?? "unknown",
            dequantMode: DequantMode(rawValue: dequantRaw) ?? .t640_3d,
            modalityScales: scales,
            calibrationCorpus: object["calibration_corpus"]?.stringValue ?? "",
            calibrationTokenCount: object["calibration_token_count"]?.numberValue.map { Int($0) } ?? 0
        )
    }
}

/// Inference bridge backed by the native FFI engine. Only selected by the
/// factory when `TesseraFFIBridge.isAvailable` is true. The inference
/// streaming surface (tessera_stream_response, design doc 14.15) is wired
/// here once the native engine ships; until then load/generate report the
/// FFI state explicitly.
public final class FFIEngineBridge: TesseraEngineBridge, @unchecked Sendable {
    private let lock = NSLock()
    private var _loadedModel: String?

    public init() {}

    public var isModelLoaded: Bool {
        lock.withLock { _loadedModel != nil }
    }

    public var loadedModelName: String? {
        lock.withLock { _loadedModel }
    }

    public func loadModel(
        ggufPath: String,
        sidecarPath: String?,
        runtime: TesseraRuntime,
        contextLength: Int
    ) async throws {
        guard TesseraFFIBridge.isAvailable else { throw EngineBridgeError.ffiUnavailable }
        let expanded = NSString(string: ggufPath).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: expanded) else {
            throw EngineBridgeError.modelNotFound(expanded)
        }
        // Production: tessera_context_init(lctx, sidecarPath). Placeholder records the path.
        lock.withLock { _loadedModel = expanded }
    }

    public func generate(
        prompt: String,
        maxTokens: Int
    ) -> AsyncThrowingStream<GeneratedToken, Error> {
        AsyncThrowingStream { continuation in
            guard TesseraFFIBridge.isAvailable else {
                continuation.finish(throwing: EngineBridgeError.ffiUnavailable)
                return
            }
            // Production: consume tessera_stream_response callbacks here.
            continuation.finish(throwing: EngineBridgeError.ffiUnavailable)
        }
    }

    public func unloadModel() async {
        lock.withLock { _loadedModel = nil }
    }
}

/// Chooses an engine bridge: prefers the native FFI when linked, otherwise
/// falls back to the CLI bridge.
public enum TesseraEngineBridgeFactory {
    /// Make an inference bridge. Set `preferCLI` to force the CLI path.
    public static func makeInferenceBridge(preferCLI: Bool = false) -> any TesseraEngineBridge {
        if !preferCLI, TesseraFFIBridge.isAvailable {
            return FFIEngineBridge()
        }
        return CLIEngineBridge()
    }
}
