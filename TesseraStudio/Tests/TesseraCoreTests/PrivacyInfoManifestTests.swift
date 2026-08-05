import XCTest

/// Verifies the macOS privacy manifest at
/// `TesseraStudio/Support/Mac/PrivacyInfo.xcprivacy` exists, parses, and
/// declares exactly the API categories the app actually touches
/// (UserDefaults + file timestamps). Acts as a regression guard so the
/// `sandbox + user-selected files + network` notarization submission
/// never ships without the manifest or with extra unused entries that
/// would draw reviewer questions.
final class PrivacyInfoManifestTests: XCTestCase {

    private static let manifestURL: URL = {
        let here = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // TesseraCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // TesseraStudio
        return here
            .appendingPathComponent("Support")
            .appendingPathComponent("Mac")
            .appendingPathComponent("PrivacyInfo.xcprivacy")
    }()

    private func loadManifest() throws -> [String: Any] {
        let data = try Data(contentsOf: Self.manifestURL)
        let plist = try PropertyListSerialization.propertyList(
            from: data,
            options: [],
            format: nil
        )
        guard let dict = plist as? [String: Any] else {
            XCTFail("PrivacyInfo.xcprivacy did not parse to a dictionary")
            throw NSError(domain: "PrivacyInfoManifestTests", code: 1)
        }
        return dict
    }

    func testPrivacyInfoManifestExists() throws {
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: Self.manifestURL.path),
            "PrivacyInfo.xcprivacy not found at \(Self.manifestURL.path)"
        )
        let plist = try loadManifest()
        XCTAssertFalse(plist.isEmpty, "manifest parsed to an empty dictionary")
    }

    func testPrivacyInfoDoesNotTrack() throws {
        let plist = try loadManifest()
        XCTAssertEqual(
            plist["NSPrivacyTracking"] as? Bool,
            false,
            "NSPrivacyTracking must be false (the app does no tracking)"
        )
        let domains = plist["NSPrivacyTrackingDomains"] as? [String]
        XCTAssertNotNil(domains, "NSPrivacyTrackingDomains must be an array")
        XCTAssertEqual(
            domains,
            [],
            "NSPrivacyTrackingDomains must be empty"
        )
        // No collected data types either.
        let collected = plist["NSPrivacyCollectedDataTypes"] as? [Any]
        XCTAssertNotNil(collected, "NSPrivacyCollectedDataTypes must be an array")
        XCTAssertEqual(
            collected?.count,
            0,
            "NSPrivacyCollectedDataTypes must be empty"
        )
    }

    func testPrivacyInfoDeclaresUsedAPITypes() throws {
        let plist = try loadManifest()
        guard let apis = plist["NSPrivacyAccessedAPITypes"] as? [[String: Any]] else {
            XCTFail("NSPrivacyAccessedAPITypes must be an array of dicts")
            return
        }
        let declaredTypes = Set(apis.compactMap { $0["NSPrivacyAccessedAPIType"] as? String })

        // Exactly the two categories Tessera Studio touches.
        XCTAssertTrue(
            declaredTypes.contains("NSPrivacyAccessedAPICategoryUserDefaults"),
            "Missing NSPrivacyAccessedAPICategoryUserDefaults declaration"
        )
        XCTAssertTrue(
            declaredTypes.contains("NSPrivacyAccessedAPICategoryFileTimestamp"),
            "Missing NSPrivacyAccessedAPICategoryFileTimestamp declaration"
        )
        XCTAssertEqual(
            declaredTypes.count,
            2,
            "Only UserDefaults and FileTimestamp should be declared; got \(declaredTypes)"
        )

        // Make sure we did not preemptively add categories the app does not use.
        for forbidden in [
            "NSPrivacyAccessedAPICategoryDiskSpace",
            "NSPrivacyAccessedAPICategorySystemBootTime",
            "NSPrivacyAccessedAPICategoryActiveKeyboards",
            "NSPrivacyAccessedAPICategoryFileUserInfo",
        ] {
            XCTAssertFalse(
                declaredTypes.contains(forbidden),
                "\(forbidden) is declared but the app does not use it"
            )
        }

        // Reasons must match the codes the app actually relies on.
        for entry in apis {
            guard let type = entry["NSPrivacyAccessedAPIType"] as? String,
                  let reasons = entry["NSPrivacyAccessedAPITypeReasons"] as? [String] else {
                XCTFail("Malformed API entry: \(entry)")
                continue
            }
            switch type {
            case "NSPrivacyAccessedAPICategoryUserDefaults":
                XCTAssertEqual(
                    reasons,
                    ["CA92.1"],
                    "UserDefaults must declare CA92.1 only"
                )
            case "NSPrivacyAccessedAPICategoryFileTimestamp":
                XCTAssertEqual(
                    reasons,
                    ["C617.1"],
                    "FileTimestamp must declare C617.1 only"
                )
            default:
                XCTFail("Unexpected API type \(type) in manifest")
            }
        }
    }
}
