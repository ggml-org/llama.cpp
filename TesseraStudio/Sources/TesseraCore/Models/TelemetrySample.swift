import Foundation

/// A single telemetry sample reported by the engine during a run.
/// See docs/tessera-studio-design.md section 3.6 / 11.7 (IOReport telemetry).
public struct TelemetrySample: Sendable, Identifiable {
    public let id: UUID
    public let timestamp: Date
    /// Decode throughput in tokens per second.
    public let tokensPerSecond: Double
    /// Resident memory usage in megabytes.
    public let memoryUsageMB: Double
    /// GPU utilization fraction (0...1), nil when unavailable.
    public let gpuUtilization: Double?
    /// Mean kernel dispatch time in milliseconds.
    public let kernelDispatchMs: Double
    /// ANE power draw in milliwatts, nil when unavailable (public-API builds).
    public let anePowerMW: Double?

    public init(
        timestamp: Date = Date(),
        tokensPerSecond: Double,
        memoryUsageMB: Double,
        gpuUtilization: Double? = nil,
        kernelDispatchMs: Double,
        anePowerMW: Double? = nil
    ) {
        self.id = UUID()
        self.timestamp = timestamp
        self.tokensPerSecond = tokensPerSecond
        self.memoryUsageMB = memoryUsageMB
        self.gpuUtilization = gpuUtilization
        self.kernelDispatchMs = kernelDispatchMs
        self.anePowerMW = anePowerMW
    }
}

/// Metric categories used to group the telemetry drawer sections.
public enum TelemetryCategory: String, CaseIterable, Sendable {
    case throughput = "Throughput"
    case memory = "Memory"
    case gpu = "GPU"
    case kernel = "Kernel"

    public var icon: String {
        switch self {
        case .throughput: "gauge.with.needle"
        case .memory: "memorychip"
        case .gpu: "gpu"
        case .kernel: "cpu"
        }
    }
}
