import XCTest

/// Verifies the macOS app icon and accent color are present in
/// `TesseraStudio/Shared/Assets.xcassets` and have the entries required
/// for the build settings in `TesseraStudio/project.yml`
/// (`ASSETCATALOG_COMPILER_APPICON_NAME: AppIcon`,
/// `ASSETCATALOG_COMPILER_GLOBAL_ACCENT_COLOR_NAME: AccentColor`) to
/// actually produce `Assets.car` and `AppIcon.icns` in the .app bundle.
///
/// Guards against the latent xcodegen 2.45.4 bug where a top-level
/// `resources:` block in `project.yml` is silently dropped, leaving
/// the catalog out of the build and the .app without an AppIcon.
final class AssetCatalogTests: XCTestCase {

    private static let catalogURL: URL = {
        let here = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // TesseraCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // TesseraStudio
        return here
            .appendingPathComponent("Shared")
            .appendingPathComponent("Assets.xcassets")
    }()

    private static let appIconContentsURL: URL = {
        AssetCatalogTests.catalogURL
            .appendingPathComponent("AppIcon.appiconset")
            .appendingPathComponent("Contents.json")
    }()

    private static let accentColorContentsURL: URL = {
        AssetCatalogTests.catalogURL
            .appendingPathComponent("AccentColor.colorset")
            .appendingPathComponent("Contents.json")
    }()

    private func loadJSON(_ url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        let parsed = try JSONSerialization.jsonObject(
            with: data,
            options: []
        )
        guard let dict = parsed as? [String: Any] else {
            XCTFail("\(url.lastPathComponent) did not parse to a dictionary")
            throw NSError(domain: "AssetCatalogTests", code: 1)
        }
        return dict
    }

    func testAssetCatalogExists() throws {
        let fm = FileManager.default
        let contents = Self.catalogURL.appendingPathComponent("Contents.json")
        XCTAssertTrue(
            fm.fileExists(atPath: Self.catalogURL.path),
            "Asset catalog not found at \(Self.catalogURL.path)"
        )
        XCTAssertTrue(
            fm.fileExists(atPath: contents.path),
            "Asset catalog Contents.json missing"
        )
        XCTAssertTrue(
            fm.fileExists(atPath: Self.appIconContentsURL.path),
            "AppIcon.appiconset/Contents.json missing"
        )
        XCTAssertTrue(
            fm.fileExists(atPath: Self.accentColorContentsURL.path),
            "AccentColor.colorset/Contents.json missing"
        )

        XCTAssertNotNil(try? loadJSON(contents))
        XCTAssertNotNil(try? loadJSON(Self.appIconContentsURL))
        XCTAssertNotNil(try? loadJSON(Self.accentColorContentsURL))
    }

    func testAppIconHasMacIconSet() throws {
        let plist = try loadJSON(Self.appIconContentsURL)
        guard let images = plist["images"] as? [[String: Any]] else {
            XCTFail("AppIcon Contents.json missing 'images' array")
            return
        }
        let macIcons = images.filter { ($0["idiom"] as? String) == "mac" }
        XCTAssertFalse(
            macIcons.isEmpty,
            "AppIcon has no 'mac' idiom entries; actool will not generate AppIcon.icns"
        )

        // Minimum macOS set: 16x16 at @1x and @2x.
        let sizes = Set(macIcons.compactMap { $0["size"] as? String })
        let scales = macIcons.compactMap { icon -> (String, String)? in
            guard let size = icon["size"] as? String,
                  let scale = icon["scale"] as? String else { return nil }
            return (size, scale)
        }
        XCTAssertTrue(
            sizes.contains("16x16"),
            "AppIcon missing mac 16x16; got sizes \(sizes.sorted())"
        )
        XCTAssertTrue(
            scales.contains(where: { $0.0 == "16x16" && $0.1 == "1x" }),
            "AppIcon missing mac 16x16@1x"
        )
        XCTAssertTrue(
            scales.contains(where: { $0.0 == "16x16" && $0.1 == "2x" }),
            "AppIcon missing mac 16x16@2x"
        )
    }

    func testAccentColorIsDefined() throws {
        let plist = try loadJSON(Self.accentColorContentsURL)
        guard let colors = plist["colors"] as? [[String: Any]] else {
            XCTFail("AccentColor Contents.json missing 'colors' array")
            return
        }
        XCTAssertFalse(
            colors.isEmpty,
            "AccentColor.colorset has no color entries; actool will not set NSAccentColorName"
        )
        let universal = colors.filter { ($0["idiom"] as? String) == "universal" }
        XCTAssertFalse(
            universal.isEmpty,
            "AccentColor.colorset has no universal color; light/dark resolution will fall back"
        )
    }

    func testAppIconMacSize512x512() throws {
        let plist = try loadJSON(Self.appIconContentsURL)
        guard let images = plist["images"] as? [[String: Any]] else {
            XCTFail("AppIcon Contents.json missing 'images' array")
            return
        }
        // App Store / Finder large-icon requirement: 512x512 @2x
        // (= 1024x1024 pixel canvas).
        let has512x2 = images.contains { icon in
            (icon["idiom"] as? String) == "mac"
                && (icon["size"] as? String) == "512x512"
                && (icon["scale"] as? String) == "2x"
        }
        XCTAssertTrue(
            has512x2,
            "AppIcon missing mac 512x512@2x entry (required for Finder / App Store)"
        )
    }
}
