import Foundation

// MARK: - Action-class classifier (autonomy spec section 3)

/// Structural identity for a family of actions. Reads toolName and argument
/// STRUCTURE only; never reads natural language. This is the primary defense
/// against approval-gaming: an agent cannot rephrase a dangerous action into
/// an approved pattern because phrasing is not an input.
///
/// Three pattern shapes:
/// 1. Verb-prefix (shell-like tools): `bash:git`, `bash:npm`.
/// 2. Path-glob (file tools): `file_write:src/**`, `file_write:<external>`.
/// 3. Arg-shape (everything else): `quantize#a1b2c3`.
/// Fallback: tool-only (`toolname`).
public enum TesseraActionClass {

    // MARK: Constants

    /// Programs that get a two-token class head (program only, not subcommand).
    static let multiWordPrograms: Set<String> = [
        "git", "npm", "cargo", "docker", "swift", "make", "gh", "uv", "pip",
    ]

    /// Destructive verbs: any class whose head matches is irreversible (section 4).
    static let destructiveVerbs: Set<String> = [
        "rm", "rmdir", "del", "delete", "drop", "purge", "erase", "format",
        "mkfs", "dd", "shred", "sudo", "chmod", "chown", "kill", "shutdown",
        "reboot",
    ]

    /// Tool-name fragments that identify shell-like tools.
    static let shellFragments: Set<String> = ["bash", "shell", "terminal"]

    /// Tool-name fragments that identify file tools.
    static let fileFragments: Set<String> = ["file", "write", "read", "edit"]

    /// Argument keys that may hold a command string (shell tools).
    static let commandKeys: Set<String> = ["command", "cmd", "script", "code", "shell"]

    /// Argument keys that may hold a file path (file tools).
    static let pathKeys: Set<String> = ["path", "file", "file_path", "filepath", "target", "destination"]

    // MARK: Classification

    /// Classify an action into a structural action-class id.
    /// Deterministic and pure: same action -> same class, always.
    public static func classify(_ action: PendingAction, pathGlobDepth: Int = 1) -> String {
        let name = action.toolName.lowercased()

        // 1. Verb-prefix class (shell-like tools).
        if shellFragments.contains(where: { name.contains($0) }) {
            return verbPrefixClass(action, toolName: name)
        }

        // 2. Path-glob class (file tools).
        if fileFragments.contains(where: { name.contains($0) }) {
            return pathGlobClass(action, toolName: name, depth: pathGlobDepth)
        }

        // 3. Arg-shape class (everything else with arguments).
        if !action.arguments.isEmpty {
            return argShapeClass(action, toolName: name)
        }

        // Fallback: tool-only.
        return name
    }

    // MARK: Verb-prefix

    /// Extract the program token from a shell command and return `tool:program`.
    static func verbPrefixClass(_ action: PendingAction, toolName: String) -> String {
        guard let cmd = commandString(from: action.arguments) else {
            return toolName
        }
        let tokens = cmd.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        guard let program = tokens.first?.lowercased() else { return toolName }
        // Strip a leading path (e.g. /usr/bin/git -> git).
        let bare = (program as NSString).lastPathComponent
        return "\(toolName):\(bare)"
    }

    /// Find the command string in the arguments.
    static func commandString(from arguments: [String: JSONValue]) -> String? {
        for key in commandKeys {
            if case .string(let value)? = arguments[key], !value.isEmpty {
                return value
            }
        }
        // Fallback: first string argument.
        for (_, value) in arguments.sorted(by: { $0.key < $1.key }) {
            if case .string(let s) = value, !s.isEmpty { return s }
        }
        return nil
    }

    // MARK: Path-glob

    /// Reduce a file path to a glob prefix and return `tool:glob`.
    /// External paths (absolute, ~, ..) collapse to `tool:<external>`.
    static func pathGlobClass(_ action: PendingAction, toolName: String, depth: Int) -> String {
        guard let path = pathString(from: action.arguments) else {
            return toolName
        }
        // External-path detection: absolute, home-relative, or parent-relative.
        if path.hasPrefix("/") || path.hasPrefix("~") || path.hasPrefix("..") {
            return "\(toolName):<external>"
        }
        let segments = path.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
        guard !segments.isEmpty else { return toolName }
        let kept = segments.prefix(max(1, depth))
        let glob = kept.joined(separator: "/") + "/**"
        return "\(toolName):\(glob)"
    }

    /// Find the path string in the arguments.
    static func pathString(from arguments: [String: JSONValue]) -> String? {
        for key in pathKeys {
            if case .string(let value)? = arguments[key], !value.isEmpty {
                return value
            }
        }
        return nil
    }

    // MARK: Arg-shape

    /// Tool name plus a stable hash of the sorted argument KEYS (structure,
    /// not values). Two calls with the same argument keys land in the same
    /// class regardless of values.
    static func argShapeClass(_ action: PendingAction, toolName: String) -> String {
        let keys = action.arguments.keys.sorted().joined(separator: ",")
        let hash = stableHash(keys)
        return "\(toolName)#\(hash)"
    }

    /// A short, stable, hex hash for class identity. Not cryptographic;
    /// just deterministic and collision-resistant for small key sets.
    static func stableHash(_ input: String) -> String {
        var h: UInt64 = 0xcbf29ce484222325  // FNV-1a offset basis
        for byte in input.utf8 {
            h ^= UInt64(byte)
            h = h &* 0x100000001b3           // FNV-1a prime
        }
        return String(h, radix: 16)
    }

    // MARK: - Irreversible guard (autonomy spec section 4)

    /// The invariant the whole system rests on. RULES, not ML. No learned
    /// component may override this.
    ///
    /// An action class is irreversible if ANY of:
    /// - Its verb head is in the destructive denylist.
    /// - `ruleBasedRisk` rates it `.high` or `.forbidden`.
    /// - It is an external-path file write (`<external>`).
    /// - It is on the user's manual denylist.
    public static func isIrreversible(
        _ actionClass: String,
        risk: TesseraActionRisk,
        denylist: Set<String> = []
    ) -> Bool {
        // Destructive verb head.
        if let head = verbHead(of: actionClass), destructiveVerbs.contains(head) {
            return true
        }
        // High or forbidden risk.
        if risk >= .high { return true }
        // External-path file write.
        if actionClass.hasSuffix(":<external>") { return true }
        // Manual denylist.
        if denylist.contains(actionClass) { return true }
        return false
    }

    /// Extract the verb head from a class id. For `bash:git` returns `git`;
    /// for `file_write:src/**` returns nil (not a verb-prefix class).
    static func verbHead(of actionClass: String) -> String? {
        let parts = actionClass.split(separator: ":", maxSplits: 1)
        guard parts.count == 2 else { return nil }
        let suffix = String(parts[1])
        // Path-glob classes contain `/` or `<external>`; they are not verb heads.
        if suffix.contains("/") || suffix.hasPrefix("<") { return nil }
        return suffix
    }
}
