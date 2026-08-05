import Foundation

// MARK: - Schema sidecar

/// On-disk format for a Python tool's parameter schema sidecar.
///
/// Lives at `tools/tessera/<script>.schema.json` and is the source of truth
/// for the agent's tool catalogue. The Swift wrapper decodes this file at
/// init time and uses the same JSON object as the `TesseraTool.parameters`
/// payload (so the agent loop's system-prompt tools block is built from
/// exactly what the CLI invokes with).
///
/// Schema format:
///   {
///     "name":                       "<tool name, e.g. multimodal_calibrate>",
///     "description":                "<human-readable description for the agent>",
///     "default_approval_level":     "auto" | "prompt" | "deny"   (optional, default "prompt")
///     "script":                     "<.py basename without extension>",
///     "subcommand":                 "<optional first-positional subcommand name>",
///     "positional":   ["<prop>", ...]   (property names to pass as positional args,
///                                        in order; default [])
///     "parameters":   { standard JSON Schema object }
///   }
///
/// Property-to-CLI mapping (what becomes `argv`):
///   * string / integer / number  --<flag> <value>
///   * boolean                    --<flag> when true, omitted when false
///   * array of strings           --<flag> v1 v2 v3 ...  (single flag, multiple values;
/////                                       matches argparse nargs='*' / '+')
///   * property key `vision_tower` -> flag `--vision-tower` (snake_case -> kebab-case)
///
/// The `default_approval_level` and `default` values in property schemas
/// are the only JSON-Schema extensions in use here; both are already
/// present in ``SchemaProperty`` (see Agent/TesseraTool.swift).
public struct PythonSchemaSidecar: Codable, Sendable, Equatable {
    public let name: String
    public let description: String
    public let defaultApprovalLevel: String?
    public let script: String
    public let subcommand: String?
    public let positional: [String]?
    public let parameters: JSONSchema

    enum CodingKeys: String, CodingKey {
        case name
        case description
        case defaultApprovalLevel = "default_approval_level"
        case script
        case subcommand
        case positional
        case parameters
    }

    public init(
        name: String,
        description: String,
        defaultApprovalLevel: String? = "prompt",
        script: String,
        subcommand: String? = nil,
        positional: [String]? = nil,
        parameters: JSONSchema
    ) {
        self.name = name
        self.description = description
        self.defaultApprovalLevel = defaultApprovalLevel
        self.script = script
        self.subcommand = subcommand
        self.positional = positional
        self.parameters = parameters
    }

    /// Decodes a sidecar from a JSON file on disk. Throws if the file is
    /// missing or malformed.
    public static func load(from url: URL) throws -> PythonSchemaSidecar {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(PythonSchemaSidecar.self, from: data)
    }
}

// MARK: - Factory

/// Builds a `TesseraTool` from a `PythonSchemaSidecar`. The tool runs the
/// named Python script via ``PythonEngineBridge``; arguments are translated
/// from the JSON parameter map to the script's argv using the convention
/// documented on ``PythonSchemaSidecar``.
///
/// Usage:
///   let sidecar = try PythonSchemaSidecar.load(from: schemaURL)
///   let tool = PythonTool(sidecar: sidecar)
///   // tool implements TesseraTool
public struct PythonTool: TesseraTool, Sendable {
    public let name: String
    public let description: String
    public let parameters: JSONSchema
    public let defaultApprovalLevel: ApprovalLevel

    private let script: String
    private let subcommand: String?
    private let positional: [String]
    private let scriptDirHint: URL?

    /// Build a tool from an in-memory sidecar.
    public init(sidecar: PythonSchemaSidecar, scriptDirHint: URL? = nil) {
        self.name = sidecar.name
        self.description = sidecar.description
        self.parameters = sidecar.parameters
        self.defaultApprovalLevel = Self.parseApproval(sidecar.defaultApprovalLevel)
        self.script = sidecar.script
        self.subcommand = sidecar.subcommand
        self.positional = sidecar.positional ?? []
        self.scriptDirHint = scriptDirHint
    }

    /// Convenience: load a sidecar from a JSON file on disk and build the tool.
    public init(schemaURL: URL, scriptDirHint: URL? = nil) throws {
        let sidecar = try PythonSchemaSidecar.load(from: schemaURL)
        self.init(sidecar: sidecar, scriptDirHint: scriptDirHint)
    }

    /// Convenience: load the sidecar by script basename. The wrapper looks
    /// for `<scriptName>.schema.json` next to the script (using the same
    /// discovery chain as ``PythonEngineBridge``). Use this in the
    /// registry: `PythonTool(scriptName: "multimodal_calibrate")`.
    public init(scriptName: String, scriptDirHint: URL? = nil) throws {
        let dir = scriptDirHint ?? Self.resolveScriptDir()
        let schemaURL = dir.appendingPathComponent("\(scriptName).schema.json")
        try self.init(schemaURL: schemaURL, scriptDirHint: dir)
    }

    /// Synchronous script-dir lookup. The bridge is an actor; in the
    /// registry (which is initialised at app launch) we do a one-shot
    /// directory probe without going through the actor.
    private static func resolveScriptDir() -> URL {
        let env = ProcessInfo.processInfo.environment
        let fm = FileManager.default
        if let explicit = env["TESSERA_SCRIPT_DIR"], !explicit.isEmpty {
            let url = URL(fileURLWithPath: explicit)
            if fm.fileExists(atPath: url.path) { return url }
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        let homeURL = home.appendingPathComponent("Developer/GitHub/tessera/tools/tessera")
        if fm.fileExists(atPath: homeURL.path) { return homeURL }
        let bundle = Bundle.main.bundleURL
        var dir = bundle.deletingLastPathComponent()
        for _ in 0..<8 {
            let probe = dir.appendingPathComponent("tools/tessera")
            if fm.fileExists(atPath: probe.path) { return probe }
            let parent = dir.deletingLastPathComponent()
            if parent.path == dir.path { break }
            dir = parent
        }
        return URL(fileURLWithPath: "/opt/tessera/tools/tessera")
    }

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        // Build argv from the JSON arg map + the (optional) subcommand.
        let argv = Self.buildArgv(
            parameters: parameters,
            positional: positional,
            subcommand: subcommand,
            arguments: arguments
        )

        let bridge = PythonEngineBridge.shared
        let result: (exitCode: Int32, stdout: String, stderr: String)
        do {
            result = try await bridge.runCollect(script: script, args: argv)
        } catch let err as PythonError {
            return .fail(err.errorDescription ?? "Python bridge error: \(err)")
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            return .fail("Python bridge error: \(error)")
        }

        if result.exitCode != 0 {
            let tail = tailOf(result.stderr, maxBytes: 4096)
            let err = PythonError.nonZeroExit(code: result.exitCode, stderrTail: tail)
            return .fail(err.errorDescription ?? "Python exited \(result.exitCode)")
        }

        let trimmed = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return .ok("Python tool '\(name)' completed with no output.",
                       data: ["script": .string(script),
                              "exit_code": .number(Double(result.exitCode))])
        }

        // Parse as JSON if the stdout looks like JSON.
        if trimmed.hasPrefix("{") || trimmed.hasPrefix("[") {
            if let data = trimmed.data(using: .utf8),
               let parsed = try? JSONSerialization.jsonObject(with: data),
               let json = jsonToJSONValue(parsed) {
                let payload: String
                if let str = try? JSONSerialization.data(
                    withJSONObject: parsed, options: [.prettyPrinted, .sortedKeys]
                ), let pretty = String(data: str, encoding: .utf8) {
                    payload = pretty
                } else {
                    payload = trimmed
                }
                return .ok(payload, data: [
                    "script": .string(script),
                    "exit_code": .number(Double(result.exitCode)),
                    "parsed": json
                ])
            }
        }
        return .ok(trimmed, data: [
            "script": .string(script),
            "exit_code": .number(Double(result.exitCode))
        ])
    }

    // MARK: argv builder

    /// Maps the agent-supplied JSON argument map to a script argv.
    /// Honours `positional` (in order), then the remaining properties in
    /// the order they appear in the schema. The subcommand (if any) is
    /// the first argv element after the script name (which the bridge adds).
    public static func buildArgv(
        parameters: JSONSchema,
        positional: [String],
        subcommand: String?,
        arguments: [String: JSONValue]
    ) -> [String] {
        var argv: [String] = []
        if let sub = subcommand, !sub.isEmpty {
            argv.append(sub)
        }

        // 1. Positional args (in declared order).
        let props = parameters.properties ?? [:]
        for key in positional {
            guard let v = arguments[key] else { continue }
            if let s = stringValue(of: v) {
                argv.append(s)
            }
        }

        // 2. Remaining properties as named flags. We walk the schema's
        //    property dictionary to keep ordering stable for human readers
        //    running the script by hand.
        let positionalSet = Set(positional)
        for (key, prop) in props where !positionalSet.contains(key) {
            guard let v = arguments[key] else { continue }
            // Skip null/empty values; honour `default` only when the
            // caller actually provided the key.
            switch v {
            case .null: continue
            case .string(let s) where s.isEmpty: continue
            default: break
            }
            let flag = "--" + key.replacingOccurrences(of: "_", with: "-")
            switch prop.type {
            case "boolean":
                if v.boolValue == true {
                    argv.append(flag)
                }
            case "array":
                guard let arr = v.arrayValue else { continue }
                let strs = arr.compactMap { $0.stringValue }
                if !strs.isEmpty {
                    argv.append(flag)
                    argv.append(contentsOf: strs)
                }
            case "integer", "number":
                if let n = v.numberValue {
                    argv.append(flag)
                    // Render integer without a trailing ".0" for cleanliness
                    if n.truncatingRemainder(dividingBy: 1) == 0,
                       abs(n) < 1e15 {
                        argv.append(String(Int64(n)))
                    } else {
                        argv.append(String(n))
                    }
                }
            default:
                // string (and unknown): treat as a single string value
                if let s = v.stringValue {
                    argv.append(flag)
                    argv.append(s)
                }
            }
        }
        return argv
    }

    // MARK: helpers

    private static func parseApproval(_ raw: String?) -> ApprovalLevel {
        guard let raw else { return .prompt }
        return ApprovalLevel(rawValue: raw) ?? .prompt
    }

    private static func stringValue(of v: JSONValue) -> String? {
        switch v {
        case .string(let s): return s
        case .number(let n):
            if n.truncatingRemainder(dividingBy: 1) == 0, abs(n) < 1e15 {
                return String(Int64(n))
            }
            return String(n)
        case .bool(let b): return b ? "true" : "false"
        default: return nil
        }
    }
}

// MARK: - JSONValue helpers

extension JSONValue {
    fileprivate var arrayValue: [JSONValue]? {
        if case .array(let v) = self { return v }
        return nil
    }
}

/// Recursively converts a Foundation JSON object into ``JSONValue``.
private func jsonToJSONValue(_ obj: Any) -> JSONValue? {
    if obj is NSNull { return .null }
    if let n = obj as? NSNumber {
        // NSNumber bridges Bool as a special class; distinguish via the
        // underlying Objective-C type so we don't lose booleans.
        let typeStr = String(cString: n.objCType)
        if typeStr == "c" || typeStr == "B" {
            return .bool(n.boolValue)
        }
        return .number(n.doubleValue)
    }
    if let s = obj as? String { return .string(s) }
    if let arr = obj as? [Any] {
        let mapped = arr.compactMap { jsonToJSONValue($0) }
        return .array(mapped)
    }
    if let dict = obj as? [String: Any] {
        var mapped: [String: JSONValue] = [:]
        for (k, v) in dict {
            if let jv = jsonToJSONValue(v) {
                mapped[k] = jv
            }
        }
        return .object(mapped)
    }
    return nil
}

/// Returns up to `maxBytes` of the trailing portion of a string, preserving
/// the tail (which is usually where the error message lives) and trimming
/// to whole UTF-8 boundaries.
private func tailOf(_ s: String, maxBytes: Int) -> String {
    if s.utf8.count <= maxBytes { return s }
    let drop = s.utf8.count - maxBytes
    let idx = s.utf8.index(s.utf8.startIndex, offsetBy: drop)
    return "..." + String(s[idx...])
}
