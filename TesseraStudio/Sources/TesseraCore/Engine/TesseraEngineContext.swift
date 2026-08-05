import Foundation
import CTesseraFFI

/// Errors surfaced by `TesseraEngineContext`. The native engine reports
/// most failure modes by returning a NULL handle (loadModel) or a negative
/// C return code (the *_model variants). The actor wraps both into a
/// Swift error so callers can `try` instead of checking return values.
public enum TesseraEngineError: Error, LocalizedError {
    /// tessera.xcframework is not linked (SwiftPM stub). The caller should
    /// fall back to tessera-cli subprocess.
    case engineUnavailable
    /// tessera_load_model returned NULL. Common causes: bad path, GGUF
    /// parse error, OOM, missing required fields.
    case modelLoadFailed(String)
    /// The handle was NULL at a *_model() call site - the caller passed a
    /// handle they had already freed, or one from a different library
    /// version. Both are programming errors; not a fallback case.
    case invalidHandle
    /// The native impl returned a negative error code (bad arguments, I/O
    /// failure). The CLI subprocess would have returned this as an exit
    /// code; we surface the code and the engine's note (if any).
    case engineError(code: Int32, note: String?)

    public var errorDescription: String? {
        switch self {
        case .engineUnavailable:
            return "Tessera engine is not available (xcframework not linked). Use the tessera-cli subprocess."
        case .modelLoadFailed(let detail):
            return "Failed to load the model: \(detail)"
        case .invalidHandle:
            return "Tessera model handle is NULL or has already been freed."
        case .engineError(let code, let note):
            if let n = note, !n.isEmpty {
                return "Tessera engine error \(code): \(n)"
            }
            return "Tessera engine error \(code)."
        }
    }
}

/// Sendable handle to a loaded Tessera model context. The handle wraps an
/// `OpaquePointer` whose target is a `tessera_model` C++ struct (which in
/// turn holds a `llama_model*`). The handle is unowned - the actor keeps
/// it alive in its `loaded` set; freeing the handle from anywhere else
/// is a use-after-free.
///
/// `OpaquePointer?` is `Sendable` (it is just a pointer), and the struct
/// itself is `Sendable` because the only stored value is the pointer;
/// all access to the underlying native object goes through
/// `TesseraEngineContext`.
public struct TesseraModelHandle: @unchecked Sendable, Equatable {
    /// Raw C handle. nil when the load failed. Never expose this to
    /// callers - they must go through `TesseraEngineContext` to free it.
    let raw: OpaquePointer?

    public init(raw: OpaquePointer?) {
        self.raw = raw
    }

    public static func == (lhs: TesseraModelHandle, rhs: TesseraModelHandle) -> Bool {
        lhs.raw == rhs.raw
    }
}

/// Actor that owns every live `tessera_model_handle_t` for the process.
///
/// Why an actor: the underlying C++ objects (`tessera_model` and its
/// `llama_model*`) are not thread-safe for concurrent mutating operations
/// such as running a GA and freeing the model. Swift's actor isolation
/// gives us a single serial executor for free, and `Sendable` handles
/// let cross-actor calls hand the pointer through without copying.
///
/// Singleton layout: there is one process-wide `TesseraEngineContext` -
/// the Tessera engine is heavy (multi-GB model resident in memory) so we
/// do not want a fresh context per call. The Studio app holds at most
/// one model loaded at a time in practice; if a future caller needs
/// multiple concurrent models they can spin up a dedicated context.
public actor TesseraEngineContext {

    public static let shared = TesseraEngineContext()

    /// Live handles, keyed by the raw pointer for O(1) contains/remove.
    /// Uses a Box<OpaquePointer?> wrapper because Swift actors can store
    /// class instances but not bare C pointers - we keep the
    /// `OpaquePointer?` in a tiny final class to satisfy the actor's
    /// mutable-state rules. The Box itself is reference-typed so the
    /// stored pointer identity stays stable across mutations.
    private final class Box {
        var pointer: OpaquePointer?
        init(_ p: OpaquePointer?) { self.pointer = p }
    }

    private var live: [Box] = []

    public init() {}

    // MARK: - Public API

    /// Load a model via the native FFI. Returns a Sendable handle on
    /// success; throws on failure. The handle is tracked in `live` so
    /// the actor's `deinit` can free any that are still around when the
    /// process exits (the C++ side is a regular `llama_model_free`, so a
    /// double-free would crash).
    public func loadModel(path: String) throws -> TesseraModelHandle {
        let cPath = (path as NSString).utf8String
        guard let raw = tessera_load_model(cPath, nil) else {
            if !TesseraFFIBridge.isAvailable {
                throw TesseraEngineError.engineUnavailable
            }
            throw TesseraEngineError.modelLoadFailed(path)
        }
        let box = Box(raw)
        live.append(box)
        return TesseraModelHandle(raw: raw)
    }

    /// Free a handle. Safe to call with a handle that was already freed -
    /// we look it up in `live` and skip the FFI call if absent. Returns
    /// silently on success.
    public func free(handle: TesseraModelHandle) {
        guard let raw = handle.raw else { return }
        guard let idx = live.firstIndex(where: { $0.pointer == raw }) else {
            // Already freed (or never owned by us). No-op.
            return
        }
        live.remove(at: idx)
        tessera_free_model(raw)
    }

    // MARK: - Engine operations (run inside the actor for serialisation)
    //
    // The C entry points are synchronous and (for the engine calls) may
    // run for seconds. Running them inside the actor means the actor's
    // executor is the only thread that ever touches a given model, which
    // matches llama.cpp's expectation of single-threaded model access
    // for forward passes. SwiftUI / CLI callers that need progress
    // reporting can use Task.detached wrappers or a separate progress
    // hook on the C++ side; we keep the bridge minimal here.
    //
    // Outcome mapping: 0 -> .success, positive -> .fallbackToCLI
    // (engine not wired yet), negative -> .error. evaluateModel uses the
    // same JSON-decoding path as the no-handle evaluate; if the native
    // side returns a "not yet wired" JSON we surface the engine's note
    // through the fallback path so the UI can show it.

    public func evolve(handle: TesseraModelHandle,
                       config: [String: Any]) -> TesseraFFIBridge.Outcome {
        let configJSON = TesseraFFIBridge.serializeConfig(config)
        let code = tessera_evolve_model(handle.raw, configJSON)
        return TesseraFFIBridge.intOutcome(code, success: "Evolution complete via FFI.")
    }

    public func evaluate(handle: TesseraModelHandle,
                         config: [String: Any]) -> TesseraFFIBridge.Outcome {
        let configJSON = TesseraFFIBridge.serializeConfig(config)
        if let s = tessera_evaluate_model(handle.raw, configJSON) {
            let str = String(cString: s)
            tessera_free_string(s)
            // The real impl returns a JSON envelope (ok, ...). The SwiftPM
            // stub and the "engine not wired" return both carry ok=false;
            // treat those as .fallbackToCLI.
            if let payload = str.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: payload) as? [String: Any],
               let ok = obj["ok"] as? Bool, ok {
                return .success(output: str)
            }
            return .fallbackToCLI
        }
        return .fallbackToCLI
    }

    public func convert(handle: TesseraModelHandle,
                        outputPath: String,
                        format: String) -> TesseraFFIBridge.Outcome {
        let code = tessera_convert_model(handle.raw, outputPath, format)
        return TesseraFFIBridge.intOutcome(code, success: "Conversion complete via FFI.")
    }

    deinit {
        // deinit on an actor runs on the actor's executor; we cannot
        // `await` here, but the FFI call is safe to invoke from any
        // thread (llama_model_free is internally synchronised). We do
        // not need the `live` array once everything is freed.
        for box in live {
            if let raw = box.pointer {
                tessera_free_model(raw)
            }
        }
    }
}
