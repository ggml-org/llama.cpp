import Foundation

/// Shared logic for numeric parameter fields (HIG 10.15). The
/// panel's display stays text-based, but the STORED value must be
/// a `.number` JSONValue whenever the text parses as a number -
/// tools read parameters through `numberValue` and would reject
/// or mis-parse a string like `"100"`.
///
/// Kept here (below the view layer) so the write-through rules
/// are unit-testable without instantiating SwiftUI.
public enum WorkflowNumericInput {
    /// The outcome of one text-field edit.
    public enum Edit: Equatable {
        /// Field was cleared: remove the key from the parameters.
        case clear
        /// Text parses as a number: store it.
        case store(Double)
        /// Mid-typing state (e.g. "-", "12."): keep the previous
        /// stored value and the user's in-progress text.
        case reject
    }

    /// The write-through rule the parameter panel applies on every
    /// keystroke of a numeric field.
    public static func edit(text: String, integer: Bool) -> Edit {
        if text.isEmpty {
            return .clear
        }
        guard let number = parse(text, integer: integer) else {
            return .reject
        }
        return .store(number)
    }

    /// Parse a field's text into a number. Locale-independent
    /// (`.` decimal separator) to match JSON round-tripping.
    public static func parse(_ text: String, integer: Bool) -> Double? {
        let trimmed = text.trimmingCharacters(in: .whitespaces)
        guard let number = Double(trimmed) else {
            return nil
        }
        return integer ? number.rounded() : number
    }

    /// Display text for a stored value. A legacy `.string` in a
    /// numeric field is shown as-is (it heals to `.number` on the
    /// next successful edit); anything non-numeric shows empty.
    public static func displayText(for value: JSONValue?) -> String {
        guard let value else { return "" }
        if let text = value.stringValue { return text }
        if let number = value.numberValue { return format(number) }
        return ""
    }

    /// Format a stored number for display: integral values render
    /// without a trailing ".0" (samples = 100, not 100.0).
    public static func format(_ number: Double) -> String {
        if number == number.rounded(), abs(number) < 1e15 {
            return String(Int(number))
        }
        return String(number)
    }

    /// The text a field should show after the stored value changed
    /// externally (stepper, undo, document load). Returns nil while
    /// the user is typing so in-progress text is never clobbered,
    /// and nil when nothing changed.
    public static func syncedText(
        current: String, value: JSONValue?, isEditing: Bool
    ) -> String? {
        guard !isEditing else { return nil }
        let display = displayText(for: value)
        return display == current ? nil : display
    }

    /// HIG 13.16: bounded integers pair a Stepper with the text
    /// field. A property is stepper-worthy only when BOTH ends of
    /// the range are known; an open or half-known range (sample
    /// counts, context lengths) stays a plain numeric field.
    public static func usesStepper(for prop: SchemaProperty) -> Bool {
        stepperBounds(for: prop) != nil
    }

    /// The stepper's clamped range, or nil when the property is
    /// not a fully-bounded integer (or its bounds are inverted).
    public static func stepperBounds(for prop: SchemaProperty) -> ClosedRange<Double>? {
        guard prop.type == "integer",
              let lower = prop.minimum, let upper = prop.maximum,
              lower <= upper else {
            return nil
        }
        return lower...upper
    }
}
