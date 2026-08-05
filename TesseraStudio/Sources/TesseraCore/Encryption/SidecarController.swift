import Foundation

/// The two long-running sidecar processes (Postgres and Valkey) are
/// stopped before the key is destroyed. This protocol is the seam so
/// the executor does not couple to the sidecar implementations.
///
/// The real implementations land with the Postgres / Valkey wiring
/// (later phase). Today, ``NoOpSidecarController`` is what every
/// caller injects; when the sidecars arrive, the production code
/// swaps in a real one and the executor does not change.
public protocol SidecarController: Sendable {
    func stopPostgres() async throws
    func stopValkey() async throws
}

/// Default ``SidecarController`` for builds where the sidecars are not
/// yet wired. `stopPostgres` and `stopValkey` are silent no-ops.
public struct NoOpSidecarController: SidecarController {
    public init() {}

    public func stopPostgres() async throws {
        // TODO(phase-postgres): send SIGTERM to tessera-postgres, wait up
        // to 5s, then SIGKILL. Wired in the Postgres integration phase.
    }

    public func stopValkey() async throws {
        // TODO(phase-valkey): send SIGTERM to tessera-valkey, wait up
        // to 5s, then SIGKILL. Wired in the Valkey integration phase.
    }
}
