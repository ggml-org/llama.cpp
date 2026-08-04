import XCTest
@testable import TesseraCore

final class ArchiveReportTests: XCTestCase {
    private let sampleJSON = """
    {
      "schema": "tessera.map-elites-archive.v1",
      "n_kurtosis_bins": 5,
      "n_rank_bins": 5,
      "n_family_bins": 8,
      "n_modality_bins": 3,
      "cells": [
        {"kurtosis_bucket": 0, "eff_rank_bucket": 1, "family_bucket": 3, "modality_bucket": 0,
         "best_fitness": 0.05, "best_alpha": 0.5, "best_clip": 1.0, "eval_count": 42,
         "tensor_name": "blk.0.attn_q.weight"},
        {"kurtosis_bucket": 2, "eff_rank_bucket": 3, "family_bucket": 5, "modality_bucket": 1,
         "best_fitness": 0.12, "best_alpha": 0.3, "best_clip": 0.8, "eval_count": 17,
         "tensor_name": "blk.4.ffn_gate.weight"}
      ]
    }
    """

    func testDecodesArchive() throws {
        let archive = try JSONDecoder().decode(ArchiveReport.self, from: Data(sampleJSON.utf8))
        XCTAssertEqual(archive.nKurtosisBins, 5)
        XCTAssertEqual(archive.nModalityBins, 3)
        XCTAssertEqual(archive.cells.count, 2)
        XCTAssertEqual(archive.cells[0].tensorName, "blk.0.attn_q.weight")
        XCTAssertEqual(archive.cells[0].evalCount, 42)
        XCTAssertEqual(archive.cells[0].modalityName, "text")
        XCTAssertEqual(archive.cells[1].modalityName, "image")
    }

    func testTotalCellsIsProductOfBins() throws {
        let archive = try JSONDecoder().decode(ArchiveReport.self, from: Data(sampleJSON.utf8))
        XCTAssertEqual(archive.totalCells, 5 * 5 * 8 * 3)
    }

    func testSummaryFromDecodedArchive() throws {
        let archive = try JSONDecoder().decode(ArchiveReport.self, from: Data(sampleJSON.utf8))
        let summary = archive.summary
        XCTAssertEqual(summary.totalCells, 600)
        XCTAssertEqual(summary.occupiedCells, 2)
        XCTAssertEqual(summary.bestFitness, 0.05, accuracy: 1e-9)
        XCTAssertEqual(summary.worstFitness, 0.12, accuracy: 1e-9)
        XCTAssertEqual(summary.meanFitness, 0.085, accuracy: 1e-9)
    }

    func testModalityNameFallback() {
        let cell = ArchiveCell(
            kurtosisBucket: 0, effRankBucket: 0, familyBucket: 0, modalityBucket: 7,
            bestFitness: 0.1, bestAlpha: 0.5, bestClip: 1.0, evalCount: 1, tensorName: "x"
        )
        XCTAssertEqual(cell.modalityName, "modality-7")
    }
}

final class ArchiveSummaryTests: XCTestCase {
    private func cell(fitness: Double, evalCount: Int64 = 1) -> ArchiveCell {
        ArchiveCell(
            kurtosisBucket: 0, effRankBucket: 0, familyBucket: 0, modalityBucket: 0,
            bestFitness: fitness, bestAlpha: 0.5, bestClip: 1.0, evalCount: evalCount, tensorName: "t"
        )
    }

    func testBestIsMinAndWorstIsMax() {
        let summary = ArchiveSummary.compute(from: [cell(fitness: 0.3), cell(fitness: 0.1), cell(fitness: 0.2)], totalCells: 100)
        XCTAssertEqual(summary.occupiedCells, 3)
        XCTAssertEqual(summary.bestFitness, 0.1, accuracy: 1e-9)
        XCTAssertEqual(summary.worstFitness, 0.3, accuracy: 1e-9)
        XCTAssertEqual(summary.meanFitness, 0.2, accuracy: 1e-9)
    }

    func testSkipsUnoccupiedCells() {
        let summary = ArchiveSummary.compute(from: [cell(fitness: 0.1), cell(fitness: 0.9, evalCount: 0)], totalCells: 10)
        XCTAssertEqual(summary.occupiedCells, 1)
        XCTAssertEqual(summary.bestFitness, 0.1, accuracy: 1e-9)
        XCTAssertEqual(summary.worstFitness, 0.1, accuracy: 1e-9)
    }

    func testEmptyArchiveIsAllZero() {
        let summary = ArchiveSummary.compute(from: [], totalCells: 600)
        XCTAssertEqual(summary, ArchiveSummary(totalCells: 600, occupiedCells: 0, meanFitness: 0, bestFitness: 0, worstFitness: 0))
    }
}

final class AcceptanceVerdictTests: XCTestCase {
    private let sampleJSON = """
    {
      "schema": "llama.tessera.acceptance.v1",
      "acceptance_passed": true,
      "composite_t2": 0.045,
      "best_single_t2": 0.063,
      "improvement_pct": 28.57,
      "composite_wins": true,
      "kendall_tau": 0.62,
      "ranking_disagreement": 0.38,
      "novelty_survives": true,
      "per_proxy": {"awq": 0.063, "rotation": 0.071, "lowrank": 0.080, "hessian": 0.095},
      "n_tensors_total": 120,
      "n_tensors_heldout": 24,
      "verdict": "Composite beats best single proxy.",
      "tensors": [
        {"name": "blk.0.attn_q.weight", "composite_t2": 0.04, "awq_t2": 0.06, "rotation_t2": 0.07,
         "lowrank_t2": 0.08, "hessian_t2": 0.09, "offline_proxy_mse": 0.05, "kernel_direct_t2": 0.045,
         "held_out": true}
      ]
    }
    """

    func testDecodesVerdict() throws {
        let verdict = try JSONDecoder().decode(AcceptanceVerdict.self, from: Data(sampleJSON.utf8))
        XCTAssertTrue(verdict.acceptancePassed)
        XCTAssertTrue(verdict.compositeWins)
        XCTAssertTrue(verdict.noveltySurvives)
        XCTAssertEqual(verdict.compositeT2, 0.045, accuracy: 1e-9)
        XCTAssertEqual(verdict.bestSingleT2, 0.063, accuracy: 1e-9)
        XCTAssertEqual(verdict.improvementPct, 28.57, accuracy: 1e-9)
        XCTAssertEqual(verdict.kendallTau, 0.62, accuracy: 1e-9)
        XCTAssertEqual(verdict.rankingDisagreement, 0.38, accuracy: 1e-9)
        XCTAssertEqual(verdict.nTensorsTotal, 120)
        XCTAssertEqual(verdict.nTensorsHeldout, 24)
    }

    func testPerProxyBreakdown() throws {
        let verdict = try JSONDecoder().decode(AcceptanceVerdict.self, from: Data(sampleJSON.utf8))
        XCTAssertEqual(verdict.perProxy.awq, 0.063, accuracy: 1e-9)
        XCTAssertEqual(verdict.perProxy.hessian, 0.095, accuracy: 1e-9)
        let labels = verdict.perProxy.labeled.map(\.label)
        XCTAssertEqual(labels, ["AWQ", "DartQuant", "FLRQ", "SEPTQ"])
    }

    func testPerTensorScores() throws {
        let verdict = try JSONDecoder().decode(AcceptanceVerdict.self, from: Data(sampleJSON.utf8))
        XCTAssertEqual(verdict.tensors.count, 1)
        XCTAssertEqual(verdict.tensors[0].offlineProxyMSE, 0.05, accuracy: 1e-9)
        XCTAssertTrue(verdict.tensors[0].heldOut)
    }

    func testFailingVerdictDecodes() throws {
        let json = """
        {"acceptance_passed": false, "composite_wins": false, "novelty_survives": false,
         "composite_t2": 0.09, "best_single_t2": 0.06, "ranking_disagreement": 0.02}
        """
        let verdict = try JSONDecoder().decode(AcceptanceVerdict.self, from: Data(json.utf8))
        XCTAssertFalse(verdict.acceptancePassed)
        XCTAssertFalse(verdict.compositeWins)
        XCTAssertFalse(verdict.noveltySurvives)
        XCTAssertEqual(verdict.rankingDisagreement, 0.02, accuracy: 1e-9)
    }
}

final class ABReportTests: XCTestCase {
    private let sampleJSON = """
    {
      "n_tensors": 3,
      "composite_offline": 0.15,
      "composite_kernel": 0.18,
      "kendall_tau": 0.33,
      "ranking_disagreement": 0.33,
      "composite_beats_single": true,
      "scores": [
        {"name": "a", "offline_proxy_mse": 0.01, "kernel_direct_t2": 0.09, "alpha_l": 1.0},
        {"name": "b", "offline_proxy_mse": 0.05, "kernel_direct_t2": 0.05, "alpha_l": 1.0},
        {"name": "c", "offline_proxy_mse": 0.09, "kernel_direct_t2": 0.01, "alpha_l": 1.0}
      ]
    }
    """

    func testDecodesReport() throws {
        let report = try JSONDecoder().decode(ABReport.self, from: Data(sampleJSON.utf8))
        XCTAssertEqual(report.nTensors, 3)
        XCTAssertEqual(report.compositeOffline, 0.15, accuracy: 1e-9)
        XCTAssertEqual(report.compositeKernel, 0.18, accuracy: 1e-9)
        XCTAssertEqual(report.kendallTau, 0.33, accuracy: 1e-9)
        XCTAssertTrue(report.compositeBeatsSingle)
        XCTAssertEqual(report.scores.count, 3)
    }

    func testMostDisagreedTensors() throws {
        let report = try JSONDecoder().decode(ABReport.self, from: Data(sampleJSON.utf8))
        // "a" and "c" flip rank entirely between the two orderings; "b" stays put.
        XCTAssertEqual(report.mostDisagreedTensors, ["a", "c"])
    }

    func testNTensorsFallsBackToScoreCount() throws {
        let json = """
        {"composite_offline": 0.1, "composite_kernel": 0.2, "kendall_tau": 1.0,
         "ranking_disagreement": 0.0, "composite_beats_single": false,
         "scores": [{"name": "x", "offline_proxy_mse": 0.1, "kernel_direct_t2": 0.2, "alpha_l": 1.0}]}
        """
        let report = try JSONDecoder().decode(ABReport.self, from: Data(json.utf8))
        XCTAssertEqual(report.nTensors, 1)
    }
}

final class L2ReportTests: XCTestCase {
    private let sampleJSON = """
    {
      "schema": "llama.tessera.runtime-probe.v1",
      "layer": "L2",
      "bf16_model": "model-bf16.gguf",
      "quant_model": "model-tessera.gguf",
      "corpus": "wiki",
      "flag_multiplier": 1.5,
      "n_tensors": 2,
      "n_flagged": 1,
      "tensors": [
        {"tensor": "blk.0.attn_q.weight", "qtype": "tessera_t640", "shape": [4096, 4096],
         "divergence": {"max_abs": 0.1, "mean_abs": 0.01, "relative_frobenius": 0.04, "per_layer_norm": 0.02},
         "expected_frob": 0.05, "flag_threshold": 0.075, "flagged": false},
        {"tensor": "blk.1.ffn_up.weight", "qtype": "tessera_t320", "shape": [8192, 4096],
         "divergence": {"max_abs": 0.5, "mean_abs": 0.08, "relative_frobenius": 0.20, "per_layer_norm": 0.1},
         "expected_frob": 0.08, "flag_threshold": 0.12, "flagged": true}
      ]
    }
    """

    func testDecodesReport() throws {
        let report = try JSONDecoder().decode(L2Report.self, from: Data(sampleJSON.utf8))
        XCTAssertEqual(report.layer, "L2")
        XCTAssertEqual(report.flagMultiplier, 1.5, accuracy: 1e-9)
        XCTAssertEqual(report.nTensors, 2)
        XCTAssertEqual(report.nFlagged, 1)
        XCTAssertEqual(report.tensors.count, 2)
    }

    func testTensorDivergenceAndFlagging() throws {
        let report = try JSONDecoder().decode(L2Report.self, from: Data(sampleJSON.utf8))
        let clean = report.tensors[0]
        XCTAssertEqual(clean.divergence.relativeFrobenius, 0.04, accuracy: 1e-9)
        XCTAssertFalse(clean.flagged)
        XCTAssertEqual(clean.shape, [4096, 4096])

        let flagged = report.tensors[1]
        XCTAssertTrue(flagged.flagged)
        XCTAssertEqual(flagged.divergence.relativeFrobenius, 0.20, accuracy: 1e-9)
        XCTAssertEqual(flagged.flagThreshold, 0.12, accuracy: 1e-9)
    }
}

final class AnalyticsReportRoutingTests: XCTestCase {
    func testRoutesArchive() throws {
        let json = #"{"schema":"tessera.map-elites-archive.v1","cells":[]}"#
        guard case .archive = try AnalyticsReport.decode(Data(json.utf8)) else {
            return XCTFail("expected archive")
        }
    }

    func testRoutesAcceptance() throws {
        let json = #"{"acceptance_passed":true,"composite_t2":0.05}"#
        guard case .acceptance = try AnalyticsReport.decode(Data(json.utf8)) else {
            return XCTFail("expected acceptance")
        }
    }

    func testRoutesAB() throws {
        let json = #"{"composite_offline":0.1,"composite_kernel":0.2,"scores":[]}"#
        guard case .ab = try AnalyticsReport.decode(Data(json.utf8)) else {
            return XCTFail("expected ab")
        }
    }

    func testRoutesL2() throws {
        let json = #"{"flag_multiplier":1.5,"n_flagged":0,"tensors":[]}"#
        guard case .l2 = try AnalyticsReport.decode(Data(json.utf8)) else {
            return XCTFail("expected l2")
        }
    }

    func testUnknownSchemaThrows() {
        let json = #"{"model":{"name":"x"},"calibration":{"corpus":"c"}}"#
        XCTAssertThrowsError(try AnalyticsReport.decode(Data(json.utf8))) { error in
            guard case AnalyticsReport.DecodeError.unknownSchema = error else {
                return XCTFail("expected unknownSchema, got \(error)")
            }
        }
    }
}

// MARK: - Fitness heatmap text contrast (HIG 1.6 / 1.7)

/// The archive heatmap fills tiles across a red (worst) -> green
/// (best) scale at fixed brightness 0.85. A single fixed text
/// color fails contrast at one end or the other (white on green,
/// black on red). These tests pin the WCAG-based picker: light
/// fills get dark text, dark fills get light text, and the text
/// color only ever flips between the two extremes.
final class FitnessScaleContrastTests: XCTestCase {
    private func cell(_ fitness: Double) -> ArchiveCell {
        ArchiveCell(
            kurtosisBucket: 0, effRankBucket: 0, familyBucket: 0,
            modalityBucket: 0, bestFitness: fitness, bestAlpha: 0,
            bestClip: 0, evalCount: 1, tensorName: "t"
        )
    }

    private func scale() -> FitnessScale {
        FitnessScale(cells: [cell(0.05), cell(0.12)])
    }

    func testBestIsGreenWorstIsRed() {
        let s = scale()
        // best = min fitness (0.05), worst = max (0.12).
        XCTAssertEqual(s.best, 0.05)
        XCTAssertEqual(s.worst, 0.12)
    }

    func testGreenBestTileGetsDarkText() {
        // The best cell maps to hue 0.33 (green, high luminance),
        // which needs dark text for contrast.
        XCTAssertEqual(scale().textColor(for: 0.05), .black)
    }

    func testRedWorstTileGetsLightText() {
        // The worst cell maps to hue 0 (red, low luminance), which
        // needs light text for contrast.
        XCTAssertEqual(scale().textColor(for: 0.12), .white)
    }

    func testDegenerateSingleFitnessGetsDarkText() {
        // All-equal fitness collapses to the green branch of
        // color(for:), so the text must be dark to match.
        let s = FitnessScale(cells: [cell(0.07), cell(0.07)])
        XCTAssertEqual(s.textColor(for: 0.07), .black)
    }

    func testTextColorIsOnlyBlackOrWhite() {
        // The picker must stay binary so the tile never renders a
        // mid-gray that fails contrast against both fill ends.
        let s = scale()
        for f in stride(from: 0.05, through: 0.12, by: 0.01) {
            let c = s.textColor(for: f)
            XCTAssertTrue(c == .black || c == .white,
                "unexpected text color \(c) at fitness \(f)")
        }
    }
}
