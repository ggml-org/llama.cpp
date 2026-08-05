import Foundation

/// One captured S2S utterance, schema llama.tessera.s2s.v1 (s2s design
/// section 4.1). Written as NDJSON by TesseraTraceStore.appendS2S into
/// traces-s2s-<date>.jsonl, one record per utterance with one device-local
/// random sid (same semantics as the runtime trace sid: maps to nothing but
/// its own records, stripped on any promotion).
///
/// Tier semantics (s2s design section 4.2): codes are Tier B LOCAL-ONLY and
/// voice-bearing (they resynthesize through Code2Wav), so records carry
/// "provenance":"s2s" and are never eligible for dataset staging. Capture is
/// default-on with no opt-out (mandatory-collection doctrine); the local-only
/// tier means default-on storage creates no egress exposure.
public struct TesseraS2SRecord: Codable, Sendable, Equatable {
    /// Exact text pair, training-ready without re-derivation: the tokens
    /// Gemma produced plus the post-retokenize Qwen ids.
    public struct Text: Codable, Sendable, Equatable {
        public var gemmaTokens: [Int32]
        public var qwenIds: [Int32]
        /// The exact UTF-8 answer Gemma produced, pre-retokenize.
        public var utf8: String

        private enum CodingKeys: String, CodingKey {
            case gemmaTokens = "gemma_tokens"
            case qwenIds = "qwen_ids"
            case utf8
        }

        public init(gemmaTokens: [Int32], qwenIds: [Int32], utf8: String) {
            self.gemmaTokens = gemmaTokens
            self.qwenIds = qwenIds
            self.utf8 = utf8
        }
    }

    /// Codec codes, zlib-compressed base64 (code streams are highly
    /// compressible). Payload layout is fixed by the schema version:
    /// frame-major, each frame = 16 little-endian UInt16 values where index
    /// 0 is codebook 0 (semantic) and indices 1-15 are the acoustic layers.
    public struct Codes: Codable, Sendable, Equatable {
        public var zlibB64: String
        public var frames: Int

        private enum CodingKeys: String, CodingKey {
            case zlibB64 = "zlib_b64"
            case frames
        }

        public init(zlibB64: String, frames: Int) {
            self.zlibB64 = zlibB64
            self.frames = frames
        }
    }

    /// Timing channel. Durations in microseconds, rates in frames/second.
    public struct Timing: Codable, Sendable, Equatable {
        public var retokenizeUs: Int
        public var talkerTtftUs: Int
        public var decodeFramesPerS: Double
        public var code2wavFramesPerS: Double
        public var firstPacketUs: Int

        private enum CodingKeys: String, CodingKey {
            case retokenizeUs = "retokenize_us"
            case talkerTtftUs = "talker_ttft_us"
            case decodeFramesPerS = "decode_frames_per_s"
            case code2wavFramesPerS = "code2wav_frames_per_s"
            case firstPacketUs = "first_packet_us"
        }

        public init(
            retokenizeUs: Int,
            talkerTtftUs: Int,
            decodeFramesPerS: Double,
            code2wavFramesPerS: Double,
            firstPacketUs: Int
        ) {
            self.retokenizeUs = retokenizeUs
            self.talkerTtftUs = talkerTtftUs
            self.decodeFramesPerS = decodeFramesPerS
            self.code2wavFramesPerS = code2wavFramesPerS
            self.firstPacketUs = firstPacketUs
        }
    }

    /// Voice configuration. Presets are the only real producer (cloning is
    /// on indefinite hold), so preset id is the norm; the schema also allows
    /// a reference-audio CONTENT HASH, never raw audio.
    public struct Voice: Codable, Sendable, Equatable {
        public var preset: String
        public var refHash: String?

        private enum CodingKeys: String, CodingKey {
            case preset
            case refHash = "ref_hash"
        }

        public init(preset: String, refHash: String? = nil) {
            self.preset = preset
            self.refHash = refHash
        }
    }

    /// Implicit feedback, as known at record write time.
    public struct Feedback: Codable, Sendable, Equatable {
        public var interrupted: Bool
        public var regenerated: Bool
        public var replayed: Bool

        public init(interrupted: Bool, regenerated: Bool, replayed: Bool) {
            self.interrupted = interrupted
            self.regenerated = regenerated
            self.replayed = replayed
        }
    }

    public static let schemaStamp = "llama.tessera.s2s.v1"
    public static let provenanceValue = "s2s"
    /// One frame = codebook 0 plus acoustic layers 1-15.
    public static let codesPerFrame = 16

    public var schema: String
    public var sid: String
    public var provenance: String
    public var text: Text
    public var codes: Codes
    public var timing: Timing
    public var voice: Voice
    public var feedback: Feedback
    /// Source-manifest lineage: model digests keyed by role (e.g. trunk,
    /// talker, code2wav). Values are digest hex of the producing assets.
    public var models: [String: String]

    public init(
        sid: String,
        text: Text,
        codes: Codes,
        timing: Timing,
        voice: Voice,
        feedback: Feedback,
        models: [String: String]
    ) {
        self.schema = Self.schemaStamp
        self.sid = sid
        self.provenance = Self.provenanceValue
        self.text = text
        self.codes = codes
        self.timing = timing
        self.voice = voice
        self.feedback = feedback
        self.models = models
    }

    /// One NDJSON line, deterministic key order.
    public func jsonLine() throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(self)
        return String(decoding: data, as: UTF8.self)
    }

    /// Decode one NDJSON line. Tolerant of surrounding whitespace, strict on
    /// schema stamp and provenance: anything else reads as nil.
    public static func decode(line: String) -> TesseraS2SRecord? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let data = trimmed.data(using: .utf8) else { return nil }
        guard let record = try? JSONDecoder().decode(TesseraS2SRecord.self, from: data) else { return nil }
        guard record.schema == schemaStamp, record.provenance == provenanceValue else { return nil }
        return record
    }

    /// Strip the sid for promotion (s2s design 4.1: stripped on ANY
    /// promotion, mirroring the runtime trace promotion path where promoted
    /// replay records carry no sid). Fail-closed: only s2s-provenance
    /// records are eligible; any other line returns nil, so this path can
    /// never launder a foreign record. Idempotent: an already-stripped s2s
    /// record passes through unchanged.
    public static func strippingSid(fromLine line: String) -> String? {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              let data = trimmed.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data),
              var dict = obj as? [String: Any] else { return nil }
        guard (dict["schema"] as? String) == schemaStamp,
              (dict["provenance"] as? String) == provenanceValue else { return nil }
        guard dict["sid"] != nil else { return trimmed }
        dict.removeValue(forKey: "sid")
        guard let out = try? JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys]) else { return nil }
        return String(decoding: out, as: UTF8.self)
    }
}

/// Codebook stream codec for TesseraS2SRecord.Codes: frame-major
/// little-endian UInt16 payload, zlib-compressed, base64-encoded.
public enum TesseraS2SCodes {
    /// Encode frames (each exactly TesseraS2SRecord.codesPerFrame codes,
    /// index 0 = codebook 0, indices 1-15 = acoustic layers). Throws when a
    /// frame has the wrong width.
    public static func encode(frames: [[UInt16]]) throws -> TesseraS2SRecord.Codes {
        guard frames.allSatisfy({ $0.count == TesseraS2SRecord.codesPerFrame }) else {
            throw TesseraS2SCodesError.frameWidth
        }
        guard !frames.isEmpty else { return TesseraS2SRecord.Codes(zlibB64: "", frames: 0) }
        var payload = Data(capacity: frames.count * TesseraS2SRecord.codesPerFrame * 2)
        for frame in frames {
            for code in frame {
                payload.append(UInt8(code & 0xFF))
                payload.append(UInt8(code >> 8))
            }
        }
        let compressed = try (payload as NSData).compressed(using: .zlib) as Data
        return TesseraS2SRecord.Codes(zlibB64: compressed.base64EncodedString(), frames: frames.count)
    }

    /// Decode back to frames. Returns nil for anything that is not a valid
    /// base64 zlib stream of an exact multiple of one frame's bytes.
    public static func decode(_ codes: TesseraS2SRecord.Codes) -> [[UInt16]]? {
        if codes.frames == 0 && codes.zlibB64.isEmpty { return [] }
        guard let compressed = Data(base64Encoded: codes.zlibB64) else { return nil }
        guard let payload = try? (compressed as NSData).decompressed(using: .zlib) as Data else { return nil }
        let frameBytes = TesseraS2SRecord.codesPerFrame * 2
        guard !payload.isEmpty, payload.count % frameBytes == 0 else { return nil }
        var frames: [[UInt16]] = []
        var frame: [UInt16] = []
        var i = 0
        while i < payload.count {
            let code = UInt16(payload[i]) | (UInt16(payload[i + 1]) << 8)
            frame.append(code)
            i += 2
            if frame.count == TesseraS2SRecord.codesPerFrame {
                frames.append(frame)
                frame = []
            }
        }
        guard frames.count == codes.frames else { return nil }
        return frames
    }
}

public enum TesseraS2SCodesError: LocalizedError, Equatable {
    case frameWidth

    public var errorDescription: String? {
        switch self {
        case .frameWidth:
            return "every S2S code frame must hold exactly \(TesseraS2SRecord.codesPerFrame) codes (codebook 0 plus acoustic layers 1-15)"
        }
    }
}
