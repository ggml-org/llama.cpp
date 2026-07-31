import Foundation

/// The inference runtime backend for a model.
public enum TesseraRuntime: String, Codable, CaseIterable, Sendable {
    /// CoreML on the Neural Engine (iPhone / iPad / Mac ANE).
    case onDevice = "on_device"
    /// MLX framework (Apple Silicon GPU, Mac-only).
    case mlx = "mlx"
    /// Remote private cloud endpoint.
    case privateCloud = "private_cloud"

    public var displayName: String {
        switch self {
        case .onDevice: "On-Device (ANE)"
        case .mlx: "MLX"
        case .privateCloud: "Private Cloud"
        }
    }

    public var icon: String {
        switch self {
        case .onDevice: "cpu"
        case .mlx: "gpu"
        case .privateCloud: "cloud"
        }
    }

    public var isAvailable: Bool {
        switch self {
        case .onDevice:
            return true
        case .mlx:
            #if os(macOS)
            return true
            #else
            return false
            #endif
        case .privateCloud:
            return true
        }
    }
}
