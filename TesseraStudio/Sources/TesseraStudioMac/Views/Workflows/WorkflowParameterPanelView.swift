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
            field(key: key, prop: prop)
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
                TextField(
                    "",
                    text: bindingForString(key: key)
                )
                .textFieldStyle(.roundedBorder)
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
