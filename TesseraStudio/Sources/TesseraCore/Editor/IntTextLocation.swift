import Foundation
#if canImport(AppKit)
import AppKit
#elseif canImport(UIKit)
import UIKit
#endif

// MARK: - IntTextLocation

/// A concrete `NSTextLocation` implementation that wraps a
/// single integer. The TextKit 2 platform APIs take
/// `NSTextLocation`-conforming objects; the platform's
/// built-in `NSTextContentStorage` uses `NSNumber`-based
/// locations under the hood. We use a real subclass so the
/// Swift type system is happy (`NSNumber` doesn't conform
/// to `NSTextLocation` in Swift, even though the
/// Objective-C runtime accepts it).
///
/// The class is the only `NSTextLocation` the editor
/// produces or consumes. `NSTextRange` init takes
/// `NSTextLocation`, so we wrap our integer offsets in
/// `IntTextLocation` instances when building ranges and
/// unwrap them with `intValue` when reading them back.
///
/// **NSTextLocation requirements.** NSTextLocation in
/// TextKit 2 requires a `compare(_:) -> ComparisonResult`
/// method (the platform's internal Swift module declares
/// it; the public headers only forward-declare the
/// protocol). We implement it via the integer value.
///
/// **Equality.** The class implements `isEqual:` so two
/// locations with the same integer compare equal. This is
/// what `NSTextRange` and `NSTextContentManager` rely on
/// for ordering and containment checks.
@objc(IntTextLocation)
public final class IntTextLocation: NSObject, NSTextLocation {
    public let intValue: Int

    public init(intValue: Int) {
        self.intValue = intValue
        super.init()
    }

    public func compare(_ location: NSTextLocation) -> ComparisonResult {
        guard let other = location as? IntTextLocation else { return .orderedSame }
        if intValue < other.intValue { return .orderedAscending }
        if intValue > other.intValue { return .orderedDescending }
        return .orderedSame
    }

    public override func isEqual(_ object: Any?) -> Bool {
        if let other = object as? IntTextLocation {
            return other.intValue == intValue
        }
        if let n = object as? NSNumber {
            return n.intValue == intValue
        }
        return false
    }

    public override var hash: Int {
        intValue.hashValue
    }

    public override var description: String {
        "\(intValue)"
    }
}

// MARK: - IntTextRange helpers

/// Convenience: build an `NSTextRange` from integer offsets.
/// The platform's `NSTextRange.init(location:endLocation:)`
/// is the canonical path; this helper wraps the
/// `IntTextLocation` boilerplate. Returns nil only when
/// the start exceeds the end.
public func makeIntTextRange(start: Int, end: Int) -> NSTextRange? {
    NSTextRange(
        location: IntTextLocation(intValue: start),
        end: IntTextLocation(intValue: end)
    )
}
