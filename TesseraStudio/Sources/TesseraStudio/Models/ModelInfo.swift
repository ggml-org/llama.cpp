import Foundation

/// Metadata describing a model available in the library.
struct ModelInfo: Identifiable, Codable, Sendable {
    let id: UUID
    let name: String
    let family: String
    let parameterCount: String
    let quantization: String
    let effectiveBits: Double
    let fileSizeBytes: Int64
    let runtime: TesseraRuntime
    let isTesseraQuantized: Bool
    let hasMLModelC: Bool
    let hasSidecar: Bool
    let ggufPath: String?
    let mlmodelcPath: String?
    let dateAdded: Date

    var fileSizeFormatted: String {
        ByteCountFormatter.string(fromByteCount: fileSizeBytes, countStyle: .file)
    }

    var badges: [ModelBadge] {
        var result: [ModelBadge] = []
        if isTesseraQuantized { result.append(.ane) }
        if hasMLModelC { result.append(.coreml) }
        if runtime == .mlx { result.append(.mlx) }
        return result
    }

    init(
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

enum ModelBadge: String, CaseIterable, Sendable {
    case ane = "ANE"
    case coreml = "CoreML"
    case mlx = "MLX"
    case gpu = "GPU"
    case reasoning = "REASONING"

    var color: String {
        switch self {
        case .ane: "green"
        case .coreml: "blue"
        case .mlx: "purple"
        case .gpu: "orange"
        case .reasoning: "pink"
        }
    }
}
