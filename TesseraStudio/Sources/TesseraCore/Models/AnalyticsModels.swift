import Foundation

// Analytics models for the runtime-aware pipeline results (S11). Each type
// mirrors a C++ JSON report emitted by tools/quantize/tessera:
//
//   ArchiveReport     tessera.map-elites-archive.v1   (tessera-search.cpp)
//   AcceptanceVerdict llama.tessera.acceptance.v1     (tessera-acceptance.cpp)
//   ABReport          A/B harness receipt             (tessera-ab-harness.cpp)
//   L2Report          llama.tessera.runtime-probe.v1  (tessera-l2-diff.cpp)
//
// Decoding is defensive (missing fields fall back to neutral defaults) to
// match the rest of the Studio models.

// MARK: - MAP-Elites archive

/// The MAP-Elites quality-diversity archive (schema tessera.map-elites-archive.v1).
/// The JSON only persists occupied cells; the full grid size is the product
/// of the bin counts.
public struct ArchiveReport: Identifiable, Codable, Sendable {
    public let id: UUID
    public let schema: String
    public let nKurtosisBins: Int
    public let nRankBins: Int
    public let nFamilyBins: Int
    public let nModalityBins: Int
    public let cells: [ArchiveCell]

    enum CodingKeys: String, CodingKey {
        case schema
        case nKurtosisBins = "n_kurtosis_bins"
        case nRankBins = "n_rank_bins"
        case nFamilyBins = "n_family_bins"
        case nModalityBins = "n_modality_bins"
        case cells
    }

    public init(
        schema: String = "tessera.map-elites-archive.v1",
        nKurtosisBins: Int = 5,
        nRankBins: Int = 5,
        nFamilyBins: Int = 8,
        nModalityBins: Int = 3,
        cells: [ArchiveCell] = []
    ) {
        self.id = UUID()
        self.schema = schema
        self.nKurtosisBins = nKurtosisBins
        self.nRankBins = nRankBins
        self.nFamilyBins = nFamilyBins
        self.nModalityBins = nModalityBins
        self.cells = cells
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = UUID()
        self.schema = (try? c.decode(String.self, forKey: .schema)) ?? "tessera.map-elites-archive.v1"
        self.nKurtosisBins = (try? c.decode(Int.self, forKey: .nKurtosisBins)) ?? 5
        self.nRankBins = (try? c.decode(Int.self, forKey: .nRankBins)) ?? 5
        self.nFamilyBins = (try? c.decode(Int.self, forKey: .nFamilyBins)) ?? 8
        self.nModalityBins = (try? c.decode(Int.self, forKey: .nModalityBins)) ?? 3
        self.cells = (try? c.decode([ArchiveCell].self, forKey: .cells)) ?? []
    }

    /// Total grid cells (product of the bin counts).
    public var totalCells: Int {
        max(nKurtosisBins, 0) * max(nRankBins, 0) * max(nFamilyBins, 0) * max(nModalityBins, 0)
    }

    public var summary: ArchiveSummary {
        ArchiveSummary.compute(from: cells, totalCells: totalCells)
    }
}

/// One occupied MAP-Elites cell: the best policy found for a regime cell.
public struct ArchiveCell: Codable, Sendable, Identifiable {
    public var id: String {
        "\(kurtosisBucket)x\(effRankBucket)x\(familyBucket)x\(modalityBucket)"
    }
    public let kurtosisBucket: Double
    public let effRankBucket: Double
    public let familyBucket: Int
    public let modalityBucket: Int
    public let bestFitness: Double
    public let bestAlpha: Double
    public let bestClip: Double
    public let evalCount: Int64
    public let tensorName: String

    enum CodingKeys: String, CodingKey {
        case kurtosisBucket = "kurtosis_bucket"
        case effRankBucket = "eff_rank_bucket"
        case familyBucket = "family_bucket"
        case modalityBucket = "modality_bucket"
        case bestFitness = "best_fitness"
        case bestAlpha = "best_alpha"
        case bestClip = "best_clip"
        case evalCount = "eval_count"
        case tensorName = "tensor_name"
    }

    public init(
        kurtosisBucket: Double,
        effRankBucket: Double,
        familyBucket: Int,
        modalityBucket: Int,
        bestFitness: Double,
        bestAlpha: Double,
        bestClip: Double,
        evalCount: Int64,
        tensorName: String
    ) {
        self.kurtosisBucket = kurtosisBucket
        self.effRankBucket = effRankBucket
        self.familyBucket = familyBucket
        self.modalityBucket = modalityBucket
        self.bestFitness = bestFitness
        self.bestAlpha = bestAlpha
        self.bestClip = bestClip
        self.evalCount = evalCount
        self.tensorName = tensorName
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.kurtosisBucket = (try? c.decode(Double.self, forKey: .kurtosisBucket)) ?? 0
        self.effRankBucket = (try? c.decode(Double.self, forKey: .effRankBucket)) ?? 0
        self.familyBucket = (try? c.decode(Int.self, forKey: .familyBucket)) ?? 0
        self.modalityBucket = (try? c.decode(Int.self, forKey: .modalityBucket)) ?? 0
        self.bestFitness = (try? c.decode(Double.self, forKey: .bestFitness)) ?? 0
        self.bestAlpha = (try? c.decode(Double.self, forKey: .bestAlpha)) ?? 0
        self.bestClip = (try? c.decode(Double.self, forKey: .bestClip)) ?? 0
        self.evalCount = (try? c.decode(Int64.self, forKey: .evalCount)) ?? 0
        self.tensorName = (try? c.decode(String.self, forKey: .tensorName)) ?? ""
    }

    /// Modality label for modality_bucket (0=text, 1=image, 2=audio).
    public var modalityName: String {
        switch modalityBucket {
        case 0: "text"
        case 1: "image"
        case 2: "audio"
        default: "modality-\(modalityBucket)"
        }
    }
}

/// Aggregate stats over the archive (mirrors ts_archive_summary).
public struct ArchiveSummary: Codable, Sendable, Equatable {
    public let totalCells: Int
    public let occupiedCells: Int
    public let meanFitness: Double
    public let bestFitness: Double
    public let worstFitness: Double

    enum CodingKeys: String, CodingKey {
        case totalCells = "total_cells"
        case occupiedCells = "occupied_cells"
        case meanFitness = "mean_fitness"
        case bestFitness = "best_fitness"
        case worstFitness = "worst_fitness"
    }

    public init(totalCells: Int, occupiedCells: Int, meanFitness: Double, bestFitness: Double, worstFitness: Double) {
        self.totalCells = totalCells
        self.occupiedCells = occupiedCells
        self.meanFitness = meanFitness
        self.bestFitness = bestFitness
        self.worstFitness = worstFitness
    }

    /// Replicates ts_archive_summarize: only cells with eval_count > 0 count
    /// as occupied; best = min fitness (lower is better), worst = max.
    public static func compute(from cells: [ArchiveCell], totalCells: Int) -> ArchiveSummary {
        let occupied = cells.filter { $0.evalCount > 0 }
        guard !occupied.isEmpty else {
            return ArchiveSummary(totalCells: totalCells, occupiedCells: 0, meanFitness: 0, bestFitness: 0, worstFitness: 0)
        }
        var sum = 0.0
        var best = Double.greatestFiniteMagnitude
        var worst = -Double.greatestFiniteMagnitude
        for cell in occupied {
            sum += cell.bestFitness
            best = min(best, cell.bestFitness)
            worst = max(worst, cell.bestFitness)
        }
        return ArchiveSummary(
            totalCells: totalCells,
            occupiedCells: occupied.count,
            meanFitness: sum / Double(occupied.count),
            bestFitness: best,
            worstFitness: worst
        )
    }
}

// MARK: - Acceptance gate

/// G6 acceptance gate result (schema llama.tessera.acceptance.v1).
public struct AcceptanceVerdict: Identifiable, Codable, Sendable {
    public let id: UUID
    public let schema: String
    public let acceptancePassed: Bool
    public let compositeT2: Double
    public let bestSingleT2: Double
    public let improvementPct: Double
    public let compositeWins: Bool
    public let kendallTau: Double
    public let rankingDisagreement: Double
    public let noveltySurvives: Bool
    public let perProxy: AcceptancePerProxy
    public let nTensorsTotal: Int64
    public let nTensorsHeldout: Int64
    public let verdict: String
    public let tensors: [AcceptanceTensor]

    enum CodingKeys: String, CodingKey {
        case schema
        case acceptancePassed = "acceptance_passed"
        case compositeT2 = "composite_t2"
        case bestSingleT2 = "best_single_t2"
        case improvementPct = "improvement_pct"
        case compositeWins = "composite_wins"
        case kendallTau = "kendall_tau"
        case rankingDisagreement = "ranking_disagreement"
        case noveltySurvives = "novelty_survives"
        case perProxy = "per_proxy"
        case nTensorsTotal = "n_tensors_total"
        case nTensorsHeldout = "n_tensors_heldout"
        case verdict
        case tensors
    }

    public init(
        schema: String = "llama.tessera.acceptance.v1",
        acceptancePassed: Bool,
        compositeT2: Double,
        bestSingleT2: Double,
        improvementPct: Double,
        compositeWins: Bool,
        kendallTau: Double,
        rankingDisagreement: Double,
        noveltySurvives: Bool,
        perProxy: AcceptancePerProxy = AcceptancePerProxy(),
        nTensorsTotal: Int64 = 0,
        nTensorsHeldout: Int64 = 0,
        verdict: String = "",
        tensors: [AcceptanceTensor] = []
    ) {
        self.id = UUID()
        self.schema = schema
        self.acceptancePassed = acceptancePassed
        self.compositeT2 = compositeT2
        self.bestSingleT2 = bestSingleT2
        self.improvementPct = improvementPct
        self.compositeWins = compositeWins
        self.kendallTau = kendallTau
        self.rankingDisagreement = rankingDisagreement
        self.noveltySurvives = noveltySurvives
        self.perProxy = perProxy
        self.nTensorsTotal = nTensorsTotal
        self.nTensorsHeldout = nTensorsHeldout
        self.verdict = verdict
        self.tensors = tensors
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = UUID()
        self.schema = (try? c.decode(String.self, forKey: .schema)) ?? "llama.tessera.acceptance.v1"
        self.acceptancePassed = (try? c.decode(Bool.self, forKey: .acceptancePassed)) ?? false
        self.compositeT2 = (try? c.decode(Double.self, forKey: .compositeT2)) ?? 0
        self.bestSingleT2 = (try? c.decode(Double.self, forKey: .bestSingleT2)) ?? 0
        self.improvementPct = (try? c.decode(Double.self, forKey: .improvementPct)) ?? 0
        self.compositeWins = (try? c.decode(Bool.self, forKey: .compositeWins)) ?? false
        self.kendallTau = (try? c.decode(Double.self, forKey: .kendallTau)) ?? 0
        self.rankingDisagreement = (try? c.decode(Double.self, forKey: .rankingDisagreement)) ?? 0
        self.noveltySurvives = (try? c.decode(Bool.self, forKey: .noveltySurvives)) ?? false
        self.perProxy = (try? c.decode(AcceptancePerProxy.self, forKey: .perProxy)) ?? AcceptancePerProxy()
        self.nTensorsTotal = (try? c.decode(Int64.self, forKey: .nTensorsTotal)) ?? 0
        self.nTensorsHeldout = (try? c.decode(Int64.self, forKey: .nTensorsHeldout)) ?? 0
        self.verdict = (try? c.decode(String.self, forKey: .verdict)) ?? ""
        self.tensors = (try? c.decode([AcceptanceTensor].self, forKey: .tensors)) ?? []
    }
}

/// Per-proxy mean t_l^2 over held-out tensors. The JSON keys are the C++
/// field names; the display labels map rotation->DartQuant, lowrank->FLRQ,
/// hessian->SEPTQ.
public struct AcceptancePerProxy: Codable, Sendable, Equatable {
    public let awq: Double
    public let rotation: Double
    public let lowrank: Double
    public let hessian: Double

    public init(awq: Double = 0, rotation: Double = 0, lowrank: Double = 0, hessian: Double = 0) {
        self.awq = awq
        self.rotation = rotation
        self.lowrank = lowrank
        self.hessian = hessian
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.awq = (try? c.decode(Double.self, forKey: .awq)) ?? 0
        self.rotation = (try? c.decode(Double.self, forKey: .rotation)) ?? 0
        self.lowrank = (try? c.decode(Double.self, forKey: .lowrank)) ?? 0
        self.hessian = (try? c.decode(Double.self, forKey: .hessian)) ?? 0
    }

    /// (display label, value) pairs for the breakdown panel.
    public var labeled: [(label: String, value: Double)] {
        [("AWQ", awq), ("DartQuant", rotation), ("FLRQ", lowrank), ("SEPTQ", hessian)]
    }
}

/// Per-tensor acceptance scores.
public struct AcceptanceTensor: Codable, Sendable, Identifiable {
    public var id: String { name }
    public let name: String
    public let compositeT2: Double
    public let awqT2: Double
    public let rotationT2: Double
    public let lowrankT2: Double
    public let hessianT2: Double
    public let offlineProxyMSE: Double
    public let kernelDirectT2: Double
    public let heldOut: Bool

    enum CodingKeys: String, CodingKey {
        case name
        case compositeT2 = "composite_t2"
        case awqT2 = "awq_t2"
        case rotationT2 = "rotation_t2"
        case lowrankT2 = "lowrank_t2"
        case hessianT2 = "hessian_t2"
        case offlineProxyMSE = "offline_proxy_mse"
        case kernelDirectT2 = "kernel_direct_t2"
        case heldOut = "held_out"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.name = (try? c.decode(String.self, forKey: .name)) ?? ""
        self.compositeT2 = (try? c.decode(Double.self, forKey: .compositeT2)) ?? 0
        self.awqT2 = (try? c.decode(Double.self, forKey: .awqT2)) ?? 0
        self.rotationT2 = (try? c.decode(Double.self, forKey: .rotationT2)) ?? 0
        self.lowrankT2 = (try? c.decode(Double.self, forKey: .lowrankT2)) ?? 0
        self.hessianT2 = (try? c.decode(Double.self, forKey: .hessianT2)) ?? 0
        self.offlineProxyMSE = (try? c.decode(Double.self, forKey: .offlineProxyMSE)) ?? 0
        self.kernelDirectT2 = (try? c.decode(Double.self, forKey: .kernelDirectT2)) ?? 0
        self.heldOut = (try? c.decode(Bool.self, forKey: .heldOut)) ?? false
    }
}

// MARK: - A/B harness receipt

/// A/B harness receipt comparing the offline proxy against kernel-direct
/// fitness (ts_ab_receipt_json; no schema field in the JSON).
public struct ABReport: Identifiable, Codable, Sendable {
    public let id: UUID
    public let nTensors: Int
    public let compositeOffline: Double
    public let compositeKernel: Double
    public let kendallTau: Double
    public let rankingDisagreement: Double
    public let compositeBeatsSingle: Bool
    public let scores: [ABTensorScore]

    enum CodingKeys: String, CodingKey {
        case nTensors = "n_tensors"
        case compositeOffline = "composite_offline"
        case compositeKernel = "composite_kernel"
        case kendallTau = "kendall_tau"
        case rankingDisagreement = "ranking_disagreement"
        case compositeBeatsSingle = "composite_beats_single"
        case scores
    }

    public init(
        nTensors: Int = 0,
        compositeOffline: Double,
        compositeKernel: Double,
        kendallTau: Double,
        rankingDisagreement: Double,
        compositeBeatsSingle: Bool,
        scores: [ABTensorScore] = []
    ) {
        self.id = UUID()
        self.nTensors = nTensors
        self.compositeOffline = compositeOffline
        self.compositeKernel = compositeKernel
        self.kendallTau = kendallTau
        self.rankingDisagreement = rankingDisagreement
        self.compositeBeatsSingle = compositeBeatsSingle
        self.scores = scores
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = UUID()
        self.compositeOffline = (try? c.decode(Double.self, forKey: .compositeOffline)) ?? 0
        self.compositeKernel = (try? c.decode(Double.self, forKey: .compositeKernel)) ?? 0
        self.kendallTau = (try? c.decode(Double.self, forKey: .kendallTau)) ?? 0
        self.rankingDisagreement = (try? c.decode(Double.self, forKey: .rankingDisagreement)) ?? 0
        self.compositeBeatsSingle = (try? c.decode(Bool.self, forKey: .compositeBeatsSingle)) ?? false
        self.scores = (try? c.decode([ABTensorScore].self, forKey: .scores)) ?? []
        self.nTensors = (try? c.decode(Int.self, forKey: .nTensors)) ?? self.scores.count
    }

    /// Names of the tensors whose rank differs most between the offline and
    /// kernel-direct orderings (the largest proxy/kernel ranking disagreements).
    public var mostDisagreedTensors: Set<String> {
        guard scores.count > 1 else { return [] }
        let offlineRank = Self.ranks(scores.map(\.offlineProxyMSE))
        let kernelRank = Self.ranks(scores.map(\.kernelDirectT2))
        let deltas = scores.indices.map { i in
            (name: scores[i].name, delta: abs(offlineRank[i] - kernelRank[i]))
        }
        let threshold = deltas.map(\.delta).max() ?? 0
        guard threshold > 0 else { return [] }
        return Set(deltas.filter { $0.delta == threshold }.map(\.name))
    }

    /// Ordinal ranks (0-based, dense) of the given values, ascending.
    private static func ranks(_ values: [Double]) -> [Int] {
        let order = values.indices.sorted { values[$0] < values[$1] }
        var result = [Int](repeating: 0, count: values.count)
        for (rank, index) in order.enumerated() {
            result[index] = rank
        }
        return result
    }
}

/// Per-tensor A/B scores.
public struct ABTensorScore: Codable, Sendable, Identifiable {
    public var id: String { name }
    public let name: String
    public let offlineProxyMSE: Double
    public let kernelDirectT2: Double
    public let alphaL: Double

    enum CodingKeys: String, CodingKey {
        case name
        case offlineProxyMSE = "offline_proxy_mse"
        case kernelDirectT2 = "kernel_direct_t2"
        case alphaL = "alpha_l"
    }

    public init(name: String, offlineProxyMSE: Double, kernelDirectT2: Double, alphaL: Double = 1.0) {
        self.name = name
        self.offlineProxyMSE = offlineProxyMSE
        self.kernelDirectT2 = kernelDirectT2
        self.alphaL = alphaL
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.name = (try? c.decode(String.self, forKey: .name)) ?? ""
        self.offlineProxyMSE = (try? c.decode(Double.self, forKey: .offlineProxyMSE)) ?? 0
        self.kernelDirectT2 = (try? c.decode(Double.self, forKey: .kernelDirectT2)) ?? 0
        self.alphaL = (try? c.decode(Double.self, forKey: .alphaL)) ?? 1.0
    }
}

// MARK: - L2 divergence report

/// L2 BF16-vs-quantized divergence report (schema llama.tessera.runtime-probe.v1).
public struct L2Report: Identifiable, Codable, Sendable {
    public let id: UUID
    public let schema: String
    public let layer: String
    public let bf16Model: String
    public let quantModel: String
    public let corpus: String
    public let flagMultiplier: Double
    public let nTensors: Int
    public let nFlagged: Int
    public let tensors: [L2TensorResult]

    enum CodingKeys: String, CodingKey {
        case schema, layer, corpus, tensors
        case bf16Model = "bf16_model"
        case quantModel = "quant_model"
        case flagMultiplier = "flag_multiplier"
        case nTensors = "n_tensors"
        case nFlagged = "n_flagged"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = UUID()
        self.schema = (try? c.decode(String.self, forKey: .schema)) ?? "llama.tessera.runtime-probe.v1"
        self.layer = (try? c.decode(String.self, forKey: .layer)) ?? "L2"
        self.bf16Model = (try? c.decode(String.self, forKey: .bf16Model)) ?? ""
        self.quantModel = (try? c.decode(String.self, forKey: .quantModel)) ?? ""
        self.corpus = (try? c.decode(String.self, forKey: .corpus)) ?? ""
        self.flagMultiplier = (try? c.decode(Double.self, forKey: .flagMultiplier)) ?? 1.5
        self.tensors = (try? c.decode([L2TensorResult].self, forKey: .tensors)) ?? []
        self.nTensors = (try? c.decode(Int.self, forKey: .nTensors)) ?? self.tensors.count
        self.nFlagged = (try? c.decode(Int.self, forKey: .nFlagged)) ?? self.tensors.filter(\.flagged).count
    }
}

/// Per-tensor L2 divergence result with the type-aware flag decision.
public struct L2TensorResult: Codable, Sendable, Identifiable {
    public var id: String { tensor }
    public let tensor: String
    public let qtype: String
    public let shape: [Int64]
    public let divergence: L2Divergence
    public let expectedFrob: Double
    public let flagThreshold: Double
    public let flagged: Bool

    enum CodingKeys: String, CodingKey {
        case tensor, qtype, shape, divergence, flagged
        case expectedFrob = "expected_frob"
        case flagThreshold = "flag_threshold"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.tensor = (try? c.decode(String.self, forKey: .tensor)) ?? ""
        self.qtype = (try? c.decode(String.self, forKey: .qtype)) ?? ""
        self.shape = (try? c.decode([Int64].self, forKey: .shape)) ?? []
        self.divergence = (try? c.decode(L2Divergence.self, forKey: .divergence)) ?? L2Divergence()
        self.expectedFrob = (try? c.decode(Double.self, forKey: .expectedFrob)) ?? 0
        self.flagThreshold = (try? c.decode(Double.self, forKey: .flagThreshold)) ?? 0
        self.flagged = (try? c.decode(Bool.self, forKey: .flagged)) ?? false
    }
}

/// Divergence metrics between BF16 source and dequantized weights.
public struct L2Divergence: Codable, Sendable, Equatable {
    public let maxAbs: Double
    public let meanAbs: Double
    public let relativeFrobenius: Double
    public let perLayerNorm: Double

    enum CodingKeys: String, CodingKey {
        case maxAbs = "max_abs"
        case meanAbs = "mean_abs"
        case relativeFrobenius = "relative_frobenius"
        case perLayerNorm = "per_layer_norm"
    }

    public init(maxAbs: Double = 0, meanAbs: Double = 0, relativeFrobenius: Double = 0, perLayerNorm: Double = 0) {
        self.maxAbs = maxAbs
        self.meanAbs = meanAbs
        self.relativeFrobenius = relativeFrobenius
        self.perLayerNorm = perLayerNorm
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.maxAbs = (try? c.decode(Double.self, forKey: .maxAbs)) ?? 0
        self.meanAbs = (try? c.decode(Double.self, forKey: .meanAbs)) ?? 0
        self.relativeFrobenius = (try? c.decode(Double.self, forKey: .relativeFrobenius)) ?? 0
        self.perLayerNorm = (try? c.decode(Double.self, forKey: .perLayerNorm)) ?? 0
    }
}

// MARK: - Discriminated loader

/// Any analytics report, discriminated by the top-level JSON keys so the
/// dashboard's "Load Analytics..." importer can route a file to its view.
public enum AnalyticsReport: Sendable {
    case archive(ArchiveReport)
    case acceptance(AcceptanceVerdict)
    case ab(ABReport)
    case l2(L2Report)

    public enum DecodeError: Error {
        case unknownSchema
    }

    public static func decode(_ data: Data) throws -> AnalyticsReport {
        let probe = (try? JSONDecoder().decode([String: JSONValue].self, from: data)) ?? [:]
        if probe["cells"] != nil {
            return .archive(try JSONDecoder().decode(ArchiveReport.self, from: data))
        }
        if probe["acceptance_passed"] != nil || probe["per_proxy"] != nil {
            return .acceptance(try JSONDecoder().decode(AcceptanceVerdict.self, from: data))
        }
        if probe["composite_offline"] != nil || probe["composite_kernel"] != nil {
            return .ab(try JSONDecoder().decode(ABReport.self, from: data))
        }
        if probe["flag_multiplier"] != nil || probe["n_flagged"] != nil {
            return .l2(try JSONDecoder().decode(L2Report.self, from: data))
        }
        throw DecodeError.unknownSchema
    }

    /// Load and decode a report from a file URL, handling security-scoped
    /// access for file-picker / sandboxed sources.
    public static func load(from url: URL) throws -> AnalyticsReport {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        return try decode(try Data(contentsOf: url))
    }
}

// MARK: - Run attachment

public extension RunRecord {
    /// Best-effort decode of an analytics report carried in the run's metrics
    /// JSON. Returns nil for receipt-shaped (or empty) metrics, so the
    /// analytics UI only surfaces for runs that actually carry a report.
    var analyticsReport: AnalyticsReport? {
        guard let data = metricsJSON.data(using: .utf8) else { return nil }
        return try? AnalyticsReport.decode(data)
    }

    var hasAnalytics: Bool { analyticsReport != nil }

    var acceptanceVerdict: AcceptanceVerdict? {
        if case .acceptance(let verdict)? = analyticsReport { return verdict }
        return nil
    }

    var archiveReport: ArchiveReport? {
        if case .archive(let archive)? = analyticsReport { return archive }
        return nil
    }
}
