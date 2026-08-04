import XCTest
@testable import TesseraCore

/// Tests for the wire-compatibility decision behind the live
/// drag feedback (``WorkflowGeometry/isWireCompatible(source:target:)``).
final class WorkflowGeometryWireTests: XCTestCase {
    func testSameTypeIsAlwaysCompatible() {
        for t in WorkflowPortType.allCases {
            XCTAssertTrue(
                WorkflowGeometry.isWireCompatible(source: t, target: t),
                "\(t.rawValue) should accept its own type"
            )
        }
    }

    func testPathWidensIntoGgufAndJson() {
        XCTAssertTrue(WorkflowGeometry.isWireCompatible(source: .path, target: .gguf))
        XCTAssertTrue(WorkflowGeometry.isWireCompatible(source: .path, target: .json))
    }

    func testWideningDoesNotReverse() {
        XCTAssertFalse(WorkflowGeometry.isWireCompatible(source: .gguf, target: .path))
        XCTAssertFalse(WorkflowGeometry.isWireCompatible(source: .json, target: .path))
    }

    func testDistinctTypesAreIncompatible() {
        XCTAssertFalse(WorkflowGeometry.isWireCompatible(source: .string, target: .number))
        XCTAssertFalse(WorkflowGeometry.isWireCompatible(source: .boolean, target: .string))
        XCTAssertFalse(WorkflowGeometry.isWireCompatible(source: .toolResult, target: .bag))
        XCTAssertFalse(WorkflowGeometry.isWireCompatible(source: .gguf, target: .json))
    }

    /// The live feedback must agree with the drop-time
    /// validation for every type pair, otherwise the editor
    /// would highlight ports the drop then rejects (or the
    /// other way around).
    func testMatchesDropTimeValidation() {
        for source in WorkflowPortType.allCases {
            for target in WorkflowPortType.allCases {
                XCTAssertEqual(
                    WorkflowGeometry.isWireCompatible(source: source, target: target),
                    source.canFlowInto(target)
                )
            }
        }
    }
}
