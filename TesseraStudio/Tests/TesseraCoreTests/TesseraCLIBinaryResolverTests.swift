import XCTest
@testable import TesseraCore

final class TesseraCLIBinaryResolverTests: XCTestCase {

    override func setUp() {
        super.setUp()
        UserDefaults.standard.removeObject(forKey: TesseraSettingsKey.tesseraCLIPath)
    }

    override func tearDown() {
        UserDefaults.standard.removeObject(forKey: TesseraSettingsKey.tesseraCLIPath)
        super.tearDown()
    }

    func testOverrideWins() throws {
        let tmp = try makeExecutableTempFile()
        let resolved = TesseraCLIBinaryResolver.resolve(override: tmp)
        XCTAssertEqual(resolved, tmp)
    }

    func testSettingsKeyWinsWhenOverrideEmpty() throws {
        let tmp = try makeExecutableTempFile()
        UserDefaults.standard.set(tmp, forKey: TesseraSettingsKey.tesseraCLIPath)
        let resolved = TesseraCLIBinaryResolver.resolve(
            override: nil,
            settingsKey: TesseraSettingsKey.tesseraCLIPath
        )
        XCTAssertEqual(resolved, tmp)
    }

    func testOverrideBeatsSettingsKey() throws {
        let overrideBin = try makeExecutableTempFile()
        let settingsBin = try makeExecutableTempFile()
        UserDefaults.standard.set(settingsBin, forKey: TesseraSettingsKey.tesseraCLIPath)
        let resolved = TesseraCLIBinaryResolver.resolve(
            override: overrideBin,
            settingsKey: TesseraSettingsKey.tesseraCLIPath
        )
        XCTAssertEqual(resolved, overrideBin)
    }

    func testIgnoresNonExecutableOverride() throws {
        // A file that exists but is not marked executable must be skipped
        // (mirrors the resolver's `isExecutable` contract) so a stale
        // override cannot make the resolver return a non-runnable path.
        let tmpDir = NSTemporaryDirectory()
        let path = (tmpDir as NSString).appendingPathComponent("tessera-not-exec-\(UUID().uuidString)")
        try "stub".write(toFile: path, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(atPath: path) }
        let resolved = TesseraCLIBinaryResolver.resolve(override: path)
        XCTAssertNotEqual(resolved, path)
    }

    func testKnownLocationsPickedUp() throws {
        let tmp = try makeExecutableTempFile()
        // Swap the known locations list to a single fake path so the test
        // does not depend on the real ~/Developer/GitHub/tessera layout.
        let original = TesseraCLIBinaryResolver.knownLocations
        TesseraCLIBinaryResolver.knownLocations = [tmp]
        defer { TesseraCLIBinaryResolver.knownLocations = original }
        let resolved = TesseraCLIBinaryResolver.resolve(
            override: nil,
            settingsKey: nil
        )
        XCTAssertEqual(resolved, tmp)
    }

    func testPathLookupWinsLast() throws {
        // When override + settings + known locations are all empty, the
        // resolver asks `pathLookup`. Supply one that returns a fake path;
        // the resolver should return that path when the caller reports it
        // as executable.
        let stubPath = "/tmp/tessera-cli-fake-\(UUID().uuidString)"
        let result = TesseraCLIBinaryResolver.resolvedPathOrDiagnostic(
            override: nil,
            settingsKey: nil,
            isExecutable: { $0 == stubPath },
            pathLookup: { stubPath }
        )
        XCTAssertEqual(result, .found(stubPath))
    }

    func testNotFoundReportsSearched() {
        let result = TesseraCLIBinaryResolver.resolvedPathOrDiagnostic(
            override: "/no/such/binary",
            settingsKey: nil,
            isExecutable: { _ in false },
            pathLookup: { nil }
        )
        if case let .notFound(searched) = result {
            XCTAssertTrue(searched.contains("/no/such/binary"))
        } else {
            XCTFail("expected .notFound, got \(result)")
        }
    }

    func testPathSearchFindsTesseraCLIInPATH() throws {
        // Stage a directory with a tessera-cli binary and prepend it to PATH.
        let dir = try makeExecutableTempDir()
        let pathBefore = ProcessInfo.processInfo.environment["PATH"] ?? ""
        setenv("PATH", "\(dir):\(pathBefore)", 1)
        defer { setenv("PATH", pathBefore, 1) }
        let resolved = TesseraCLIBinaryResolver.pathSearch()
        XCTAssertEqual(resolved, (dir as NSString).appendingPathComponent("tessera-cli"))
    }

    // MARK: helpers

    private func makeExecutableTempFile() throws -> String {
        let dir = NSTemporaryDirectory()
        let path = (dir as NSString).appendingPathComponent("tessera-cli-test-\(UUID().uuidString)")
        try "#!/bin/sh\nexit 0\n".write(toFile: path, atomically: true, encoding: .utf8)
        var attrs = try FileManager.default.attributesOfItem(atPath: path)
        attrs[.posixPermissions] = 0o755
        try FileManager.default.setAttributes(attrs, ofItemAtPath: path)
        // Best-effort cleanup; some test methods need the file to remain
        // for the duration, so they are responsible for removing it.
        addTeardownBlock { try? FileManager.default.removeItem(atPath: path) }
        return path
    }

    private func makeExecutableTempDir() throws -> String {
        let dir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("tessera-cli-pa-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let path = (dir as NSString).appendingPathComponent("tessera-cli")
        try "#!/bin/sh\nexit 0\n".write(toFile: path, atomically: true, encoding: .utf8)
        var attrs = try FileManager.default.attributesOfItem(atPath: path)
        attrs[.posixPermissions] = 0o755
        try FileManager.default.setAttributes(attrs, ofItemAtPath: path)
        addTeardownBlock { try? FileManager.default.removeItem(atPath: dir) }
        return dir
    }

    private func makeIsExecutableIncluding(_ paths: [String]) -> (String) -> Bool {
        let set = Set(paths)
        return { path in
            if set.contains(path) { return true }
            return FileManager.default.isExecutableFile(atPath: path)
        }
    }
}
