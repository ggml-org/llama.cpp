import Foundation

// MARK: - ProductivityTaskGraphIntegration

/// The bridge between the Tasks surface and the graph
/// view (Phase 6). The graph view's
/// ``GraphNode`` type already maps `entity_type = "task"`
/// to the green icon (see
/// `Sources/TesseraCore/Productivity/Graph/GraphModel.swift`'s
/// `color(for:)` and `iconName(for:)`); this file
/// provides the tasks-specific adapter that turns a
/// task into a graph node.
///
/// **Why a separate file:** the graph view is generic
/// over entity types; the per-material bridges
/// (contacts, tasks, etc.) live alongside the material's
/// own files so a future worker adding a new material
/// has a clear template.
public struct ProductivityTaskGraphIntegration: Sendable {

    private let store: ProductivityTaskStore

    public init(store: ProductivityTaskStore) {
        self.store = store
    }

    /// Load every task and return the corresponding
    /// graph nodes. The graph view calls this when the
    /// user toggles the "tasks" filter in the sidebar.
    /// Returns an empty array when the table is empty.
    public func loadAllNodes(limit: Int = 10_000) async throws -> [GraphNode] {
        let tasks = try await store.list(limit: limit)
        return tasks.map(graphNode(from:))
    }

    /// Build a single ``GraphNode`` from a
    /// ``ProductivityTask``. The label is the task's
    /// title (capped at 30 chars by the ``GraphNode``
    /// constructor). The importance score is 1.0 for
    /// pinned/active tasks; the graph view's
    /// progressive-disclosure policy handles the rest.
    public func graphNode(from task: ProductivityTask) -> GraphNode {
        GraphNode(
            id: task.id,
            entityType: ProductivityTask.entityType,
            subtype: task.subtypeString,
            label: task.title.isEmpty ? "(untitled)" : task.title,
            importance: task.isCompleted ? 0.3 : 1.0,
            updatedAt: task.updatedAt,
            isPinned: false
        )
    }
}
