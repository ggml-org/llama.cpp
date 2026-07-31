import Foundation

/// A quantization receipt, decoded from `tessera receipts` JSON.
/// Schema: `llama.tessera.calibration-receipt.v1`
/// (docs/tessera-studio-design.md section 14.10).
public struct QuantizationReceipt: Identifiable, Codable, Sendable {
    public let id: UUID
    public let schemaVersion: String
    public let model: ReceiptModelInfo
    public let tensors: [ReceiptTensorStat]
    public let calibration: ReceiptCalibrationConfig
    public let gaArchive: ReceiptGAArchive?
    public let durationSeconds: Double
    public let dateCreated: Date

    public init(
        schemaVersion: String = "llama.tessera.calibration-receipt.v1",
        model: ReceiptModelInfo,
        tensors: [ReceiptTensorStat] = [],
        calibration: ReceiptCalibrationConfig,
        gaArchive: ReceiptGAArchive? = nil,
        durationSeconds: Double = 0,
        dateCreated: Date = Date()
    ) {
        self.id = UUID()
        self.schemaVersion = schemaVersion
        self.model = model
        self.tensors = tensors
        self.calibration = calibration
        self.gaArchive = gaArchive
        self.durationSeconds = durationSeconds
        self.dateCreated = dateCreated
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case model
        case tensors
        case calibration
        case gaArchive = "ga_archive"
        case durationSeconds = "duration_seconds"
        case dateCreated = "date_created"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = UUID()
        self.schemaVersion = (try? c.decode(String.self, forKey: .schemaVersion)) ?? "llama.tessera.calibration-receipt.v1"
        self.model = try c.decode(ReceiptModelInfo.self, forKey: .model)
        self.tensors = (try? c.decode([ReceiptTensorStat].self, forKey: .tensors)) ?? []
        self.calibration = try c.decode(ReceiptCalibrationConfig.self, forKey: .calibration)
        self.gaArchive = try? c.decode(ReceiptGAArchive.self, forKey: .gaArchive)
        self.durationSeconds = (try? c.decode(Double.self, forKey: .durationSeconds)) ?? 0
        self.dateCreated = (try? c.decode(Date.self, forKey: .dateCreated)) ?? Date()
    }

    /// Mean squared error across all tensors, or nil when there are none.
    public var meanMSE: Double? {
        guard !tensors.isEmpty else { return nil }
        return tensors.map(\.mse).reduce(0, +) / Double(tensors.count)
    }
}

public struct ReceiptModelInfo: Codable, Sendable {
    public let name: String
    public let family: String
    public let parameterCount: String
    public let sourceBits: Double
    public let outputBits: Double
    public let fileSizeBytes: Int64

    enum CodingKeys: String, CodingKey {
        case name, family
        case parameterCount = "parameter_count"
        case sourceBits = "source_bits"
        case outputBits = "output_bits"
        case fileSizeBytes = "file_size_bytes"
    }

    public init(
        name: String,
        family: String,
        parameterCount: String,
        sourceBits: Double,
        outputBits: Double,
        fileSizeBytes: Int64
    ) {
        self.name = name
        self.family = family
        self.parameterCount = parameterCount
        self.sourceBits = sourceBits
        self.outputBits = outputBits
        self.fileSizeBytes = fileSizeBytes
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.name = (try? c.decode(String.self, forKey: .name)) ?? "Unknown"
        self.family = (try? c.decode(String.self, forKey: .family)) ?? "Unknown"
        self.parameterCount = (try? c.decode(String.self, forKey: .parameterCount)) ?? "?"
        self.sourceBits = (try? c.decode(Double.self, forKey: .sourceBits)) ?? 16
        self.outputBits = (try? c.decode(Double.self, forKey: .outputBits)) ?? 4
        self.fileSizeBytes = (try? c.decode(Int64.self, forKey: .fileSizeBytes)) ?? 0
    }
}

public struct ReceiptTensorStat: Codable, Sendable, Identifiable {
    public var id: String { name }
    public let name: String
    public let bits: Double
    public let mse: Double
    public let snrDB: Double

    enum CodingKeys: String, CodingKey {
        case name, bits, mse
        case snrDB = "snr_db"
    }

    public init(name: String, bits: Double, mse: Double, snrDB: Double) {
        self.name = name
        self.bits = bits
        self.mse = mse
        self.snrDB = snrDB
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.name = (try? c.decode(String.self, forKey: .name)) ?? "?"
        self.bits = (try? c.decode(Double.self, forKey: .bits)) ?? 0
        self.mse = (try? c.decode(Double.self, forKey: .mse)) ?? 0
        self.snrDB = (try? c.decode(Double.self, forKey: .snrDB)) ?? 0
    }
}

public struct ReceiptCalibrationConfig: Codable, Sendable {
    public let corpus: String
    public let tokenCount: Int
    public let modality: String
    public let dequantMode: String

    enum CodingKeys: String, CodingKey {
        case corpus
        case tokenCount = "token_count"
        case modality
        case dequantMode = "dequant_mode"
    }

    public init(corpus: String, tokenCount: Int, modality: String, dequantMode: String) {
        self.corpus = corpus
        self.tokenCount = tokenCount
        self.modality = modality
        self.dequantMode = dequantMode
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.corpus = (try? c.decode(String.self, forKey: .corpus)) ?? ""
        self.tokenCount = (try? c.decode(Int.self, forKey: .tokenCount)) ?? 0
        self.modality = (try? c.decode(String.self, forKey: .modality)) ?? "text"
        self.dequantMode = (try? c.decode(String.self, forKey: .dequantMode)) ?? "T640_3D"
    }
}

public struct ReceiptGAArchive: Codable, Sendable {
    public let generations: Int
    public let population: Int
    public let bestFitness: Double
    public let archiveSize: Int

    enum CodingKeys: String, CodingKey {
        case generations, population
        case bestFitness = "best_fitness"
        case archiveSize = "archive_size"
    }

    public init(generations: Int, population: Int, bestFitness: Double, archiveSize: Int) {
        self.generations = generations
        self.population = population
        self.bestFitness = bestFitness
        self.archiveSize = archiveSize
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.generations = (try? c.decode(Int.self, forKey: .generations)) ?? 0
        self.population = (try? c.decode(Int.self, forKey: .population)) ?? 0
        self.bestFitness = (try? c.decode(Double.self, forKey: .bestFitness)) ?? 0
        self.archiveSize = (try? c.decode(Int.self, forKey: .archiveSize)) ?? 0
    }
}
