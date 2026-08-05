import XCTest
import Foundation
@testable import TesseraCore

/// Tests for the `IntTextLocation` concrete NSTextLocation
/// implementation. The class is the only NSTextLocation the
/// editor produces or consumes; it backs NSTextRange for
/// `TesseraTextElement.elementRange` and
/// `TesseraTextContentManager.documentRange`.
final class IntTextLocationTests: XCTestCase {

    func testCompareAscending() {
        let a = IntTextLocation(intValue: 0)
        let b = IntTextLocation(intValue: 5)
        XCTAssertEqual(a.compare(b), .orderedAscending)
    }

    func testCompareDescending() {
        let a = IntTextLocation(intValue: 5)
        let b = IntTextLocation(intValue: 0)
        XCTAssertEqual(a.compare(b), .orderedDescending)
    }

    func testCompareSame() {
        let a = IntTextLocation(intValue: 3)
        let b = IntTextLocation(intValue: 3)
        XCTAssertEqual(a.compare(b), .orderedSame)
    }

    func testEqualityByIntValue() {
        let a = IntTextLocation(intValue: 3)
        let b = IntTextLocation(intValue: 3)
        XCTAssertEqual(a, b)
        XCTAssertEqual(a.hash, b.hash)
    }

    func testInequalityByIntValue() {
        let a = IntTextLocation(intValue: 3)
        let b = IntTextLocation(intValue: 4)
        XCTAssertNotEqual(a, b)
    }

    func testMakeIntTextRange() {
        let range = makeIntTextRange(start: 0, end: 10)
        XCTAssertNotNil(range)
        let loc = range?.location as? IntTextLocation
        let end = range?.endLocation as? IntTextLocation
        XCTAssertEqual(loc?.intValue, 0)
        XCTAssertEqual(end?.intValue, 10)
    }

    func testMakeIntTextRangeReversedReturnsNil() {
        // When start > end, NSTextRange returns nil.
        let range = makeIntTextRange(start: 5, end: 0)
        XCTAssertNil(range)
    }
}
