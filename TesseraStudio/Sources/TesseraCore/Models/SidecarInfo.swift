import Foundation

/// Metadata from a Tessera sidecar file (calibration-policy.v1 + modality_scales).
public struct SidecarInfo: Identifiable, Codable, Sendable {
    public let id: UUID
    public let modelPath: String
    public let schemaVersion: Int
    public let tesseraProfile: String
    public let effectiveBits: Double
    public let kernelVersion: String
    public let dequantMode: DequantMode
    public let modalityScales: [ModalityScale]
    public let calibrationCorpus: String
    public let calibrationTokenCount: Int
    public let dateCreated: Date

    public init(
        modelPath: String,
        schemaVersion: Int = 1,
        tesseraProfile: String,
        effectiveBits: Double,
        kernelVersion: String,
        dequantMode: DequantMode = .t640_3d,
        modalityScales: [ModalityScale] = [],
        calibrationCorpus: String = "",
        calibrationTokenCount: Int = 0
    ) {
        self.id = UUID()
        self.modelPath = modelPath
        self.schemaVersion = schemaVersion
        self.tesseraProfile = tesseraProfile
        self.effectiveBits = effectiveBits
        self.kernelVersion = kernelVersion
        self.dequantMode = dequantMode
        self.modalityScales = modalityScales
        self.calibrationCorpus = calibrationCorpus
        self.calibrationTokenCount = calibrationTokenCount
        self.dateCreated = Date()
    }
}

public enum DequantMode: String, Codable, Sendable {
    case t640_3d = "T640_3D"
    case t640_4d = "T640_4D"
    case stockKQuant = "STOCK_KQUANT"
}

public struct ModalityScale: Codable, Sendable, Identifiable {
    public var id: String { modality }
    public let modality: String
    public let awqAlpha: Double
    public let componentCount: Int

    public init(modality: String, awqAlpha: Double, componentCount: Int) {
        self.modality = modality
        self.awqAlpha = awqAlpha
        self.componentCount = componentCount
    }
}
