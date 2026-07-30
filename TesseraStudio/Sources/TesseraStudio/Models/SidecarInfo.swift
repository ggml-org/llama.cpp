import Foundation

/// Metadata from a Tessera sidecar file (calibration-policy.v1 + modality_scales).
struct SidecarInfo: Identifiable, Codable, Sendable {
    let id: UUID
    let modelPath: String
    let schemaVersion: Int
    let tesseraProfile: String
    let effectiveBits: Double
    let kernelVersion: String
    let dequantMode: DequantMode
    let modalityScales: [ModalityScale]
    let calibrationCorpus: String
    let calibrationTokenCount: Int
    let dateCreated: Date

    init(
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

enum DequantMode: String, Codable, Sendable {
    case t640_3d = "T640_3D"
    case t640_4d = "T640_4D"
    case stockKQuant = "STOCK_KQUANT"
}

struct ModalityScale: Codable, Sendable, Identifiable {
    var id: String { modality }
    let modality: String
    let awqAlpha: Double
    let componentCount: Int
}
