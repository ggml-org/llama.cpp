import Foundation

// MARK: - CalendarGraphConnector

/// Wires the graph view's "open entity" hook to the
/// calendar surface. The graph view (Phase 6) already
/// renders `calendar_event` nodes (purple, calendar icon —
/// see `GraphNode.color(for:)` / `iconName(for:)`); this
/// connector supplies the missing click-to-open path: an
/// event node's Open button refocuses the calendar onto
/// the event's day and selects it.
///
/// Usage at construction time:
///
/// ```swift
/// CalendarGraphConnector.wire(graphViewModel, to: calendarViewModel)
/// ```
public enum CalendarGraphConnector {

    /// Install the open handler. Non-calendar nodes fall
    /// through (the handler is shared; a future connector
    /// for another surface can chain by wrapping).
    @MainActor
    public static func wire(
        _ graph: GraphViewModel,
        to calendar: CalendarViewModel
    ) {
        graph.openEntityHandler = { node in
            guard node.entityType == CalendarEvent.entityType else { return }
            Task { @MainActor in
                await calendar.openEvent(id: node.id)
            }
        }
    }
}
