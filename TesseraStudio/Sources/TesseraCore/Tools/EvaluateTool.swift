import Foundation

/// Evaluates a quantized model: perplexity, latency, and power metrics.
public struct EvaluateTool: TesseraTool {
    public let name = "evaluate"
    public let description = "Evaluate a quantized model by measuring perplexity on a held-out set, token latency, and ANE power draw. Optionally runs the multi-axis capability eval (capability_eval=true)."
    public let defaultApprovalLevel = ApprovalLevel.notify

    public let parameters = JSONSchema(
        type: "object",
        properties: [
            "model_path": SchemaProperty(
                type: "string",
                description: "Path to the quantized GGUF or .mlmodelc to evaluate."
            ),
            "eval_corpus": SchemaProperty(
                type: "string",
                description: "Path to the evaluation corpus for perplexity."
            ),
            "n_tokens": SchemaProperty(
                type: "integer",
                description: "Number of tokens to evaluate. Default 512.",
                defaultValue: "512",
                minimum: 1
            ),
            "runtime": SchemaProperty(
                type: "string",
                description: "Runtime backend for evaluation.",
                enumValues: TesseraRuntime.allCases.map(\.rawValue),
                defaultValue: TesseraRuntime.onDevice.rawValue
            ),
            "measure_power": SchemaProperty(
                type: "boolean",
                description: "Whether to measure IOReport power metrics. Default true.",
                defaultValue: "true"
            ),
            "capability_eval": SchemaProperty(
                type: "boolean",
                description: "Run the multi-axis capability eval instead of perplexity/latency/power. Default false.",
                defaultValue: "false"
            ),
            "eval_instances": SchemaProperty(
                type: "string",
                description: "Optional path to a JSON file of caller-supplied eval instances + results (from a live model run). Defaults to the built-in held-out instance store."
            ),
        ],
        required: ["model_path"]
    )

    private let shell: TesseraProcessShell

    public init() {
        self.shell = ProcessRunner()
    }

    /// Test seam.
    init(shell: TesseraProcessShell) {
        self.shell = shell
    }

    public func execute(arguments: [String: JSONValue]) async throws -> ToolResult {
        // Multi-axis capability eval is a separate path; when off, the
        // perplexity/latency/power behavior below is unchanged.
        let capabilityEval = arguments["capability_eval"].map { $0 == .bool(true) || $0 == .string("true") } ?? false
        if capabilityEval {
            return await runCapabilityEval(arguments: arguments)
        }

        guard let modelPath = arguments["model_path"]?.stringValue, !modelPath.isEmpty else {
            return .fail("model_path is required")
        }

        let evalCorpus = arguments["eval_corpus"]?.stringValue ?? ""
        let nTokens = arguments["n_tokens"]?.numberValue.map { Int($0) } ?? 512
        let runtime = arguments["runtime"]?.stringValue ?? TesseraRuntime.onDevice.rawValue
        let measurePower = arguments["measure_power"].map { $0 != .bool(false) && $0 != .string("false") } ?? true
        let expandedModel = NSString(string: modelPath).expandingTildeInPath

        if TesseraFFIBridge.isAvailable {
            var config: [String: Any] = [
                "n_tokens": nTokens,
                "runtime": runtime,
                "measure_power": measurePower,
            ]
            if !evalCorpus.isEmpty {
                config["eval_corpus"] = NSString(string: evalCorpus).expandingTildeInPath
            }
            switch TesseraFFIBridge.evaluate(model: expandedModel, config: config) {
            case let .success(output):
                return .ok(output, data: [
                    "model_path": .string(modelPath),
                    "runtime": .string(runtime),
                    "n_tokens": .number(Double(nTokens)),
                    "backend": .string("ffi"),
                ])
            case .fallbackToCLI:
                break
            case let .error(code, message):
                return .fail("Evaluation failed via FFI (code \(code)): \(message)")
            }
        }

        // CLI fallback: tessera-cli evaluate <model> --config <json>
        guard let cli = TesseraCLIBinaryResolver.resolve(
            override: TesseraSettings.tesseraCLIPath,
            settingsKey: TesseraSettingsKey.tesseraCLIPath
        ) else {
            return .fail(TesseraCLIBinaryResolver.diagnosticMessage())
        }

        var config: [String: Any] = [
            "n_tokens": nTokens,
            "runtime": runtime,
            "measure_power": measurePower,
        ]
        if !evalCorpus.isEmpty {
            config["eval_corpus"] = NSString(string: evalCorpus).expandingTildeInPath
        }
        let configPath: String
        do {
            configPath = try EngineToolSupport.writeConfigFile(config: config)
        } catch {
            return .fail("Failed to write evaluate config: \(error.localizedDescription)")
        }
        defer { try? FileManager.default.removeItem(atPath: configPath) }

        let args = [
            "evaluate", expandedModel,
            "--config", configPath,
        ]
        let result = try await shell.run(
            executable: cli,
            arguments: args,
            environment: nil,
            workingDirectory: nil
        )

        if result.exitCode == 0 {
            // tessera-cli evaluate prints JSON: surface it as the agent's data
            // payload when it parses, and otherwise fall back to the raw text.
            let payload: [String: JSONValue] = [
                "model_path": .string(modelPath),
                "runtime": .string(runtime),
                "n_tokens": .number(Double(nTokens)),
                "backend": .string("cli"),
            ]
            let outputText: String
            if let parsed = EngineToolSupport.parseJSONObject(stdout: result.stdout) {
                var merged = payload
                for (k, v) in parsed {
                    if let s = v as? String { merged[k] = .string(s) }
                    else if let n = v as? NSNumber {
                        if CFGetTypeID(n) == CFBooleanGetTypeID() {
                            merged[k] = .bool(n.boolValue)
                        } else {
                            merged[k] = .number(n.doubleValue)
                        }
                    }
                }
                return .ok("Evaluation complete.", data: merged)
            }
            outputText = "Evaluation complete.\n\(result.stdout)"
            return .ok(outputText, data: payload)
        } else {
            return .fail("Evaluation failed (exit \(result.exitCode)):\n\(result.stderr)")
        }
    }

    /// Multi-axis capability eval (design 4.7 / 8). Scores caller-supplied
    /// instance results into a TesseraCapabilityScore and returns the five
    /// axis scores plus the guard value. Scoring goes through the C++ harness
    /// (--tessera-capability-eval) when its binary is installed - the source
    /// of truth - and falls back to the in-process Swift reduction when it is
    /// not; both reduce per-axis pass/fail to the same fractions.
    ///
    /// HONESTY: running instances against a live model requires a model +
    /// compute we do not have here; results are supplied by the caller via
    /// `eval_instances` (a future runner executes the instances and records
    /// pass/fail). Without results, every axis scores 0 because zero instances
    /// were scored - NOT because the model failed - and the output says so
    /// plainly. No scores are fabricated.
    private func runCapabilityEval(arguments: [String: JSONValue]) async -> ToolResult {
        let service = TesseraCapabilityEvalService()
        let store = TesseraEvalInstanceStore()
        store.seedDefaultsIfNeeded()

        var instances: [TesseraEvalInstance]
        var results: [TesseraEvalInstanceResult]

        if let path = arguments["eval_instances"]?.stringValue, !path.isEmpty {
            let url = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath)
            guard let data = try? Data(contentsOf: url),
                  let file = try? JSONDecoder().decode(TesseraEvalInstanceFile.self, from: data) else {
                return .fail("eval_instances file is missing or invalid: \(path)")
            }
            instances = file.instances
            results = file.results
        } else {
            instances = store.allInstances()
            // Requires a live model run; results supplied by the caller / a
            // future runner. We do not invent pass/fail here.
            results = []
        }

        let outcome = await service.scoreResults(results)
        let score = outcome.score
        let hasResults = !results.isEmpty

        // Persist the latest scored eval so the adaptation scheduler has a
        // real, honest input (and a baseline for its guard). Only when results
        // were actually supplied - a zero-result eval carries no signal.
        if hasResults {
            let record = TesseraCapabilityEvalRecord(
                tallies: outcome.tallies,
                score: score,
                weightedSum: outcome.weightedSum,
                backend: outcome.backend
            )
            try? TesseraCapabilityEvalStore().recordLatest(record)
        }

        let message: String
        if hasResults {
            message = "Capability eval complete (\(outcome.backend)): scored \(results.count) result(s) across \(TesseraCapabilityScore.axisNames.count) axes."
        } else {
            message = "Capability eval: no results supplied. A live model run is required to execute the "
                + "\(instances.count) held-out instance(s); pass them via eval_instances. Scores are 0 because "
                + "zero instances were scored, not because the model failed."
        }

        var data: [String: JSONValue] = [
            "backend": .string("capability_eval"),
            "scoring_backend": .string(outcome.backend),
            "scoring_note": .string(outcome.note),
            "weighted_sum": .number(outcome.weightedSum),
            "mechanical": .number(score.mechanical),
            "apiCurrency": .number(score.apiCurrency),
            "hardTail": .number(score.hardTail),
            "personalStyle": .number(score.personalStyle),
            "generalCompetence": .number(score.generalCompetence),
            "guard": .number(score.generalCompetence),
            "guard_axis": .string("generalCompetence"),
            "results_supplied": .bool(hasResults),
            "instances_scored": .number(Double(results.count)),
            "instances_available": .number(Double(instances.count)),
        ]
        if let modelPath = arguments["model_path"]?.stringValue, !modelPath.isEmpty {
            data["model_path"] = .string(modelPath)
        }

        return .ok(message, data: data)
    }
}
