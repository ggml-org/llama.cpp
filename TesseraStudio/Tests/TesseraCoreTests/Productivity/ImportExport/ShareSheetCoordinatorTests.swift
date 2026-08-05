import XCTest
@testable import TesseraCore

/// Tests for the share-sheet coordinator. The coordinator's
/// pure pieces (target list, no UI) are exercised here; the
/// ``presentShareSheet`` method (which presents
/// ``NSSharingServicePicker``) is not unit-tested because it
/// requires a live ``NSView`` and a running event loop.
final class ShareSheetCoordinatorTests: XCTestCase {

    /// The system share sheet target is always present on
    /// macOS; the slack + custom targets are appended.
    @MainActor
    func testAvailableTargetsIncludesSystemPicker() async {
        let coord = ShareSheetCoordinator(
            slackTargets: [],
            customTargets: []
        )
        let targets = await coord.availableShareTargets()
        #if canImport(AppKit)
        XCTAssertFalse(targets.isEmpty, "system share sheet should be present on macOS")
        XCTAssertTrue(
            targets.contains { $0.id == "system.sharing-service-picker" },
            "the system picker should be in the target list"
        )
        #else
        XCTAssertTrue(targets.isEmpty, "no system picker on non-macOS")
        #endif
    }

    /// Slack targets are appended to the system picker's
    /// list.
    @MainActor
    func testAvailableTargetsIncludesSlack() async {
        let slack = SlackExportTarget(
            webhookURL: URL(string: "https://hooks.slack.com/services/T0/B0/XXX")!,
            channel: "general"
        )
        let coord = ShareSheetCoordinator(
            slackTargets: [slack],
            customTargets: []
        )
        let targets = await coord.availableShareTargets()
        XCTAssertTrue(
            targets.contains { $0.id.hasPrefix("slack.") },
            "slack target should be in the list"
        )
    }

    /// Custom targets are appended in the order they were
    /// passed in.
    @MainActor
    func testAvailableTargetsIncludesCustom() async {
        let custom = ShareTarget(
            id: "custom.test",
            name: "Test target",
            accepts: [.md],
            handler: { _ in }
        )
        let coord = ShareSheetCoordinator(
            slackTargets: [],
            customTargets: [custom]
        )
        let targets = await coord.availableShareTargets()
        XCTAssertTrue(
            targets.contains { $0.id == "custom.test" },
            "custom target should be in the list"
        )
    }

    /// The Slack target's handler runs without throwing
    /// when the webhook URL is malformed; the HTTP call
    /// is exercised against the system URLSession which
    /// returns an error for an unreachable host. The
    /// handler propagates the error.
    func testSlackTargetPostPropagatesError() async throws {
        let slack = SlackExportTarget(
            webhookURL: URL(string: "https://invalid-host-for-slack-test.example/hook")!,
            channel: "general"
        )
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("slack-test-\(UUID().uuidString).md")
        try "Hello, **world**".write(to: tmp, atomically: true, encoding: String.Encoding.utf8)
        defer { try? FileManager.default.removeItem(at: tmp) }
        do {
            try await slack.post(document: tmp)
            XCTFail("expected an error from the unreachable webhook")
        } catch {
            // Expected
        }
    }
}
