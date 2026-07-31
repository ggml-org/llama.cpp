import Foundation

/// Metadata describing a model available in the library.
public struct ModelInfo: Identifiable, Codable, Sendable {
    public let id: UUID
    public let name: String
    public let family: String
    public let parameterCount: String
    public let quantization: String
    public let effectiveBits: Double
    public let fileSizeBytes: Int64
    public let runtime: TesseraRuntime
    public let isTesseraQuantized: Bool
    public let hasMLModelC: Bool
    public let hasSidecar: Bool
    public let ggufPath: String?
    public let mlmodelcPath: String?
    public let dateAdded: Date

    public var fileSizeFormatted: String {
        ByteCountFormatter.string(fromByteCount: fileSizeBytes, countStyle: .file)
    }

    public var badges: [ModelBadge] {
        var result: [ModelBadge] = []
        if isTesseraQuantized { result.append(.ane) }
        if hasMLModelC { result.append(.coreml) }
        if runtime == .mlx { result.append(.mlx) }
        return result
    }

    public init(
        name: String,
        family: String,
        parameterCount: String,
        quantization: String,
        effectiveBits: Double,
        fileSizeBytes: Int64,
        runtime: TesseraRuntime,
        isTesseraQuantized: Bool = false,
        hasMLModelC: Bool = false,
        hasSidecar: Bool = false,
        ggufPath: String? = nil,
        mlmodelcPath: String? = nil
    ) {
        self.id = UUID()
        self.name = name
        self.family = family
        self.parameterCount = parameterCount
        self.quantization = quantization
        self.effectiveBits = effectiveBits
        self.fileSizeBytes = fileSizeBytes
        self.runtime = runtime
        self.isTesseraQuantized = isTesseraQuantized
        self.hasMLModelC = hasMLModelC
        self.hasSidecar = hasSidecar
        self.ggufPath = ggufPath
        self.mlmodelcPath = mlmodelcPath
        self.dateAdded = Date()
    }
}

public enum ModelBadge: String, CaseIterable, Sendable {
    case ane = "ANE"
    case coreml = "CoreML"
    case mlx = "MLX"
    case gpu = "GPU"
    case reasoning = "REASONING"

    public var color: String {
        switch self {
        case .ane: "green"
        case .coreml: "blue"
        case .mlx: "purple"
        case .gpu: "orange"
        case .reasoning: "pink"
        }
    }
}
