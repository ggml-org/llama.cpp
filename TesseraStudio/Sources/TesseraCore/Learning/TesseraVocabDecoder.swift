import CLlama
import Foundation

/// Token decoding surface the curation stage needs from a trunk model:
/// vocab size (compatibility check) and id -> UTF-8 detokenization (analysis
/// text). A protocol so the sweep can run against a fake decoder in tests
/// without a native library (runtime-traces spec section 12.2).
public protocol TesseraSessionDecoder: AnyObject {
    /// Size of the trunk vocab; captured token ids must fall inside it.
    var nVocab: Int32 { get }

    /// Detokenize a token sequence to UTF-8. Nil on decode failure.
    func detokenize(_ tokens: [Int32]) -> String?

    /// Decode a single token (per-piece garbage heuristics).
    func piece(for token: Int32) -> String?
}

extension TesseraSessionDecoder {
    public func piece(for token: Int32) -> String? {
        detokenize([token])
    }
}

/// Shim-backed decoder: loads the trunk once per sweep purely for its vocab.
/// The engine is freed on close()/deinit; a failed load yields nil from
/// open(modelPath:) so the stage degrades open.
public final class TesseraVocabDecoder: TesseraSessionDecoder {
    private var engine: OpaquePointer?

    private init(engine: OpaquePointer) {
        self.engine = engine
    }

    /// Load the library (idempotent) and the trunk model. The decoder never
    /// runs inference, so the context stays minimal and CPU-only.
    public static func open(modelPath: String, libraryPath: String = "") -> TesseraVocabDecoder? {
        guard !modelPath.isEmpty,
              FileManager.default.fileExists(atPath: modelPath) else { return nil }
        guard cllama_load_library(libraryPath) != 0 else { return nil }
        guard let engine = cllama_engine_load(modelPath, 16, 0, 0) else { return nil }
        return TesseraVocabDecoder(engine: engine)
    }

    deinit { close() }

    public func close() {
        if let engine = self.engine {
            cllama_engine_free(engine)
            self.engine = nil
        }
    }

    public var nVocab: Int32 {
        guard let engine else { return -1 }
        return cllama_engine_n_vocab(engine)
    }

    public func detokenize(_ tokens: [Int32]) -> String? {
        guard let engine else { return nil }
        guard !tokens.isEmpty else { return "" }

        // The shim returns the negative required size when the buffer is too
        // small; retry with that size (+1 for the NUL). A non-growing
        // negative is an error, not a hint.
        var capacity = tokens.count * 8 + 16
        for _ in 0..<4 {
            var buf = [CChar](repeating: 0, count: capacity)
            let n = tokens.withUnsafeBufferPointer { ptr -> Int32 in
                cllama_detokenize(engine, ptr.baseAddress, Int32(tokens.count), &buf, Int32(capacity))
            }
            if n >= 0 {
                let bytes = buf.prefix(Int(n)).map { UInt8(bitPattern: $0) }
                return String(decoding: bytes, as: UTF8.self)
            }
            let required = Int(-n)
            guard required + 1 > capacity else { return nil }
            capacity = required + 1
        }
        return nil
    }
}
