import Foundation

/// Resolves the single llama-quantize binary that carries every Tessera C++
/// harness (anonymizer, capability-eval, adapt). One binary, several
/// subcommands selected by flag. The configured path wins; an empty setting
/// falls back to the installed default. This mirrors TesseraAnonymizerService's
/// resolution so every shell-out in the learning subsystem agrees on location.
enum TesseraHarnessBinary {
    static let defaultPath = "/usr/local/bin/llama-quantize"

    static var path: String {
        let configured = TesseraSettings.learningAnonymizerBinary
        return configured.isEmpty ? defaultPath : configured
    }

    static var isAvailable: Bool {
        FileManager.default.isExecutableFile(atPath: path)
    }
}
