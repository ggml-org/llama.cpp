import SwiftUI
import TesseraCore

/// The parameter side panel. Renders the selected node's
/// ``parameterSchema`` as a form (text field, picker, toggle
/// per property), bound to the node's `parameters` dict. Lives
/// on the right side of the canvas; hidden when no node is
/// selected.
///
/// Phase 2.4 ships a one-row-per-property form. Recursive
/// schemas (object properties, array of objects) are out of
/// scope for v1; the panel renders those as a single
/// "unsupported" placeholder so the editor never silently
/// drops a node's parameters.
struct WorkflowParameterPanelView: View {
    let node: WorkflowNode
    let type: any WorkflowNodeType.Type
    @Binding var parameters: [String: JSONValue]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            if properties.isEmpty {
                ContentUnavailableView(
                    "No parameters",
                    systemImage: "slider.horizontal.3",
                    description: Text("This node has no editable parameters; only wired inputs.")
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 12) {
                        ForEach(sortedProperties, id: \.0) { (key, prop) in
                            fieldRow(key: key, prop: prop)
                        }
                    }
                    .padding(12)
                }
                // The numeric fields keep field-local text state;
                // recreate them when the selection changes so that
                // state can't leak between nodes.
                .id(node.id)
            }
        }
        .frame(minWidth: 240)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(type.displayName)
                .font(.headline)
            Text(type.typeId)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(12)
    }

    private var properties: [String: SchemaProperty] {
        type.parameterSchema.properties ?? [:]
    }

    private var sortedProperties: [(String, SchemaProperty)] {
        properties.sorted { $0.key < $1.key }
    }

    @ViewBuilder
    private func fieldRow(key: String, prop: SchemaProperty) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(propLabel(key))
                .font(.subheadline.weight(.medium))
            // The fields below render with .labelsHidden() for
            // layout, which also strips the VoiceOver label.
            // Re-attach it here, plus the schema description as
            // the accessibility hint when one exists.
            field(key: key, prop: prop)
                .accessibilityLabel(propLabel(key))
                .accessibilityHint(prop.description ?? "")
            if let desc = prop.description {
                Text(desc)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private func field(key: String, prop: SchemaProperty) -> some View {
        if let enumValues = prop.enumValues, !enumValues.isEmpty {
            Picker("", selection: bindingForEnum(key: key, values: enumValues)) {
                ForEach(enumValues, id: \.self) { v in
                    Text(v).tag(v)
                }
            }
            .labelsHidden()
        } else {
            switch prop.type {
            case "boolean":
                Toggle("", isOn: bindingForBool(key: key))
                    .labelsHidden()
            case "integer", "number":
                NumericParameterField(
                    isInteger: prop.type == "integer",
                    value: bindingForValue(key: key),
                    bounds: WorkflowNumericInput.stepperBounds(for: prop)
                )
            case "string":
                TextField(
                    "",
                    text: bindingForString(key: key)
                )
                .textFieldStyle(.roundedBorder)
            case "array", "object":
                Text("(unsupported: \(prop.type))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            default:
                TextField("", text: bindingForString(key: key))
                    .textFieldStyle(.roundedBorder)
            }
        }
    }

    // MARK: - Bindings

    private func propLabel(_ snake: String) -> String {
        snake.split(separator: "_")
            .map { $0.prefix(1).uppercased() + $0.dropFirst() }
            .joined(separator: " ")
    }

    private func bindingForString(key: String) -> Binding<String> {
        Binding(
            get: { parameters[key]?.stringValue ?? "" },
            set: { newValue in
                if newValue.isEmpty {
                    parameters.removeValue(forKey: key)
                } else {
                    parameters[key] = .string(newValue)
                }
            }
        )
    }

    private func bindingForValue(key: String) -> Binding<JSONValue?> {
        Binding(
            get: { parameters[key] },
            set: { newValue in
                if let newValue {
                    parameters[key] = newValue
                } else {
                    parameters.removeValue(forKey: key)
                }
            }
        )
    }

    private func bindingForEnum(
        key: String, values: [String]
    ) -> Binding<String> {
        Binding(
            get: {
                let current = parameters[key]?.stringValue ?? values.first ?? ""
                return values.contains(current) ? current : (values.first ?? "")
            },
            set: { newValue in
                parameters[key] = .string(newValue)
            }
        )
    }

    private func bindingForBool(key: String) -> Binding<Bool> {
        Binding(
            get: { parameters[key]?.boolValue ?? false },
            set: { newValue in
                parameters[key] = .bool(newValue)
            }
        )
    }
}

/// A numeric parameter field. The display is a text field, but
/// valid parses write through to the stored value as a `.number`
/// JSONValue (tools read `numberValue`). The text is kept in
/// field-local state so a mid-typing entry ("12.", "-") survives
/// the parse; the write-through rule itself lives in
/// ``WorkflowNumericInput`` so the tests can drive it directly.
/// When the schema gives a full min/max range, a Stepper is
/// paired with the text field (HIG 13.16).
private struct NumericParameterField: View {
    let isInteger: Bool
    let bounds: ClosedRange<Double>?
    @Binding var value: JSONValue?
    @State private var text: String
    @FocusState private var isEditing: Bool

    init(isInteger: Bool, value: Binding<JSONValue?>, bounds: ClosedRange<Double>? = nil) {
        self.isInteger = isInteger
        self.bounds = bounds
        self._value = value
        self._text = State(
            initialValue: WorkflowNumericInput.displayText(for: value.wrappedValue))
    }

    var body: some View {
        if let bounds {
            HStack(spacing: 4) {
                textField
                Stepper("", value: stepperValue(in: bounds), step: 1)
                    .labelsHidden()
            }
        } else {
            textField
        }
    }

    private var textField: some View {
        TextField("", text: $text)
            .textFieldStyle(.roundedBorder)
            .focused($isEditing)
            .onChange(of: text) { _, newValue in
                switch WorkflowNumericInput.edit(text: newValue, integer: isInteger) {
                case .clear:
                    value = nil
                case .store(let number):
                    value = .number(number)
                case .reject:
                    break
                }
            }
            .onChange(of: value) { _, newValue in
                // Stepper, undo, or a document load changed the
                // stored value; follow it unless we are typing.
                if let synced = WorkflowNumericInput.syncedText(
                    current: text, value: newValue, isEditing: isEditing)
                {
                    text = synced
                }
            }
    }

    /// The stepper's clamped view of the stored value. Writes are
    /// `.number` by construction, so the text field follows via
    /// the value-change sync above.
    private func stepperValue(in bounds: ClosedRange<Double>) -> Binding<Double> {
        Binding(
            get: {
                let number = value?.numberValue ?? bounds.lowerBound
                return Swift.min(Swift.max(number, bounds.lowerBound), bounds.upperBound)
            },
            set: { newValue in
                value = .number(newValue)
            }
        )
    }
}
