import XCTest
@testable import TesseraCore

// W4 (s2s design section 4): the S2S trace store.
//
// Covers the Swift-side instrumentation contract: the
// llama.tessera.s2s.v1 record schema (round-trip identity for codes,
// tokens, timing), sid semantics (device-local, stripped on promotion),
// the store's s2s share (rolling cap, quarantine exemption, retention),
// the egress guard's fail-closed refusal of s2s provenance, and the
// default-on capture semantics (no opt-out surface exists).

// MARK: - Codes codec

final class TesseraS2SCodesTests: XCTestCase {
    func testRoundTripIdentity() throws {
        let frames: [[UInt16]] = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            [2047, 3071, 1234, 0, 512, 1024, 2048, 4095, 65535, 7, 8, 9, 10, 11, 12, 13],
            [11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333, 444, 555, 666, 777],
        ]
        let codes = try TesseraS2SCodes.encode(frames: frames)
        XCTAssertEqual(codes.frames, frames.count)
        XCTAssertFalse(codes.zlibB64.isEmpty)
        XCTAssertEqual(TesseraS2SCodes.decode(codes), frames)
    }

    func testRoundTripEmpty() throws {
        let codes = try TesseraS2SCodes.encode(frames: [])
        XCTAssertEqual(codes.frames, 0)
        XCTAssertEqual(codes.zlibB64, "")
        XCTAssertEqual(TesseraS2SCodes.decode(codes), [])
    }

    func testEncodeRejectsRaggedFrame() {
        XCTAssertThrowsError(try TesseraS2SCodes.encode(frames: [[1, 2, 3]]))
        XCTAssertThrowsError(try TesseraS2SCodes.encode(
            frames: [Array(repeating: UInt16(1), count: 17)]))
    }

    func testDecodeRejectsRaggedPayload() throws {
        // A valid zlib stream whose decompressed length is not a whole
        // number of frames must be refused, not truncated.
        let payload = Data((0..<30).map { UInt8($0) })
        let compressed = try (payload as NSData).compressed(using: .zlib) as Data
        let codes = TesseraS2SRecord.Codes(
            zlibB64: compressed.base64EncodedString(), frames: 1)
        XCTAssertNil(TesseraS2SCodes.decode(codes))
    }

    func testDecodeRejectsInvalidBase64() {
        XCTAssertNil(TesseraS2SCodes.decode(
            TesseraS2SRecord.Codes(zlibB64: "not base64 !!", frames: 1)))
    }

    func testDecodeRejectsMismatchedFrameCount() throws {
        let frames = [Array(repeating: UInt16(7), count: TesseraS2SRecord.codesPerFrame)]
        var codes = try TesseraS2SCodes.encode(frames: frames)
        codes.frames = 5  // lies about the frame count
        XCTAssertNil(TesseraS2SCodes.decode(codes))
    }

    func testCompressionShrinksRepetitiveStreams() throws {
        // Code streams are highly compressible; sanity-check that zlib ran.
        let frames = (0..<200).map { _ in Array(repeating: UInt16(42), count: 16) }
        let codes = try TesseraS2SCodes.encode(frames: frames)
        let rawBytes = frames.count * 16 * 2
        guard let compressed = Data(base64Encoded: codes.zlibB64) else {
            return XCTFail("codes did not round-trip through base64")
        }
        XCTAssertLessThan(compressed.count, rawBytes / 2)
    }
}

// MARK: - Record schema (round-trip identity)

final class TesseraS2SRecordTests: XCTestCase {
    static func sampleRecord(sid: String = "6F9619FF-8B86-D011-B42D-00C04FC964FF", pad: Int = 0) -> TesseraS2SRecord {
        let frames: [[UInt16]] = (0..<4).map { f in
            (0..<16).map { c in UInt16((f * 16 + c) * 37 % 3072) }
        }
        let codes = try! TesseraS2SCodes.encode(frames: frames)
        let padding = pad > 0 ? String(repeating: "x", count: pad) : ""
        return TesseraS2SRecord(
            sid: sid,
            text: TesseraS2SRecord.Text(
                gemmaTokens: [101, 202, 303, 262_000],
                qwenIds: [7, 88, 999],
                utf8: "Hello, world.\(padding)"),
            codes: codes,
            timing: TesseraS2SRecord.Timing(
                retokenizeUs: 42,
                talkerTtftUs: 8_100,
                decodeFramesPerS: 61.25,
                code2wavFramesPerS: 410.5,
                firstPacketUs: 96_500),
            voice: TesseraS2SRecord.Voice(preset: "aria"),
            feedback: TesseraS2SRecord.Feedback(
                interrupted: false, regenerated: false, replayed: true),
            models: [
                "trunk": "sha256:0123456789abcdef",
                "talker": "sha256:fedcba9876543210",
                "code2wav": "sha256:1111222233334444",
            ])
    }

    func testJsonLineRoundTripIdentity() throws {
        let record = Self.sampleRecord()
        let line = try record.jsonLine()
        XCTAssertFalse(line.contains("\n"))
        XCTAssertEqual(TesseraS2SRecord.decode(line: line), record)
    }

    func testJsonLineRoundTripWithReferenceAudioHash() throws {
        var record = Self.sampleRecord()
        record.voice.refHash = "sha256:abcd"
        let line = try record.jsonLine()
        let decoded = TesseraS2SRecord.decode(line: line)
        XCTAssertEqual(decoded, record)
        XCTAssertEqual(decoded?.voice.refHash, "sha256:abcd")
    }

    func testSchemaStampAndProvenance() throws {
        let line = try Self.sampleRecord().jsonLine()
        XCTAssertTrue(line.contains("\"schema\":\"llama.tessera.s2s.v1\""))
        XCTAssertTrue(line.contains("\"provenance\":\"s2s\""))
    }

    func testDecodeRejectsForeignSchema() {
        let foreign = "{\"schema\":\"llama.tessera.spec.v1\",\"provenance\":\"s2s\",\"sid\":\"x\"}"
        XCTAssertNil(TesseraS2SRecord.decode(line: foreign))
    }

    func testDecodeRejectsForeignProvenance() throws {
        // Same schema stamp but a tampered provenance is not an s2s record.
        var record = Self.sampleRecord()
        record.provenance = "runtime"
        let line = try record.jsonLine()
        XCTAssertNil(TesseraS2SRecord.decode(line: line))
    }

    func testDecodeToleratesWhitespaceButNotGarbage() {
        XCTAssertNil(TesseraS2SRecord.decode(line: ""))
        XCTAssertNil(TesseraS2SRecord.decode(line: "   \n"))
        XCTAssertNil(TesseraS2SRecord.decode(line: "not json"))
    }

    /// NO-KEY invariant (consent lane condition C3): no device, account, or
    /// contributor identifier anywhere in the record besides the
    /// device-local sid, which is stripped on promotion.
    func testRecordCarriesNoIdentifiersBeyondSid() throws {
        let line = try Self.sampleRecord().jsonLine()
        guard let data = line.data(using: .utf8),
              let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return XCTFail("record line is not a JSON object")
        }
        XCTAssertEqual(
            Set(obj.keys),
            ["schema", "sid", "provenance", "text", "codes", "timing",
             "voice", "feedback", "models"])

        var keys = Set<String>()
        collectKeys(obj, into: &keys)
        let forbidden = ["device", "account", "contributor", "user", "imei",
                         "serial", "hostname", "email"]
        for key in keys {
            for word in forbidden where key.contains(word) {
                XCTFail("record key '\(key)' smells like an identifier")
            }
        }
    }

    private func collectKeys(_ value: Any, into keys: inout Set<String>) {
        if let dict = value as? [String: Any] {
            for (key, child) in dict {
                keys.insert(key.lowercased())
                collectKeys(child, into: &keys)
            }
        } else if let array = value as? [Any] {
            for child in array { collectKeys(child, into: &keys) }
        }
    }
}

// MARK: - Sid stripping on promotion

final class TesseraS2SPromotionTests: XCTestCase {
    func testStrippingSidRemovesSidAndKeepsEverythingElse() throws {
        let line = try TesseraS2SRecordTests.sampleRecord().jsonLine()
        guard let stripped = TesseraS2SRecord.strippingSid(fromLine: line) else {
            return XCTFail("a valid s2s record must promote")
        }
        XCTAssertFalse(stripped.contains("\"sid\""))

        guard let lineData = line.data(using: .utf8),
              let strippedData = stripped.data(using: .utf8),
              var original = try JSONSerialization.jsonObject(with: lineData) as? [String: Any],
              let promoted = try JSONSerialization.jsonObject(with: strippedData) as? [String: Any] else {
            return XCTFail("promotion produced unparseable JSON")
        }
        original.removeValue(forKey: "sid")
        XCTAssertEqual(original as NSDictionary, promoted as NSDictionary)
    }

    func testStrippingSidIsIdempotent() throws {
        let line = try TesseraS2SRecordTests.sampleRecord().jsonLine()
        guard let once = TesseraS2SRecord.strippingSid(fromLine: line) else {
            return XCTFail("a valid s2s record must promote")
        }
        XCTAssertEqual(TesseraS2SRecord.strippingSid(fromLine: once), once)
    }

    func testStrippingSidRefusesRuntimeProvenance() {
        // Fail-closed: this promotion path can never launder a foreign
        // record (e.g. a runtime trace) into an s2s corpus.
        let runtime = "{\"schema\":\"llama.tessera.spec.v1\",\"provenance\":\"runtime\",\"sid\":\"abc\",\"drafted\":3,\"accepted\":2}"
        XCTAssertNil(TesseraS2SRecord.strippingSid(fromLine: runtime))
    }

    func testStrippingSidRefusesForeignSchema() {
        let foreign = "{\"schema\":\"llama.tessera.spec.v1\",\"provenance\":\"s2s\",\"sid\":\"abc\"}"
        XCTAssertNil(TesseraS2SRecord.strippingSid(fromLine: foreign))
    }

    func testStrippingSidRefusesUnparseable() {
        XCTAssertNil(TesseraS2SRecord.strippingSid(fromLine: "not json"))
        XCTAssertNil(TesseraS2SRecord.strippingSid(fromLine: ""))
    }
}

// MARK: - Trace store: s2s share (s2s design section 4.3)

final class TesseraTraceStoreS2STests: XCTestCase {
    private var dirs: [URL] = []

    override func tearDown() {
        for dir in dirs { try? FileManager.default.removeItem(at: dir) }
        dirs.removeAll()
        super.tearDown()
    }

    private func makeStore() throws -> TesseraTraceStore {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-s2s-store-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        dirs.append(dir)
        return TesseraTraceStore(directory: dir)
    }

    private func s2sLine(sid: String, pad: Int = 0) throws -> String {
        try TesseraS2SRecordTests.sampleRecord(sid: sid, pad: pad).jsonLine()
    }

    /// Write an s2s file under a hand-built dated name (as the runtime tests
    /// do for replay files), so oldest-first ordering needs no sleeps.
    private func writeS2SFile(_ store: TesseraTraceStore, name: String, lines: [String]) throws -> URL {
        let url = store.directoryURL.appendingPathComponent(name)
        try (lines.joined(separator: "\n") + "\n")
            .write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    // Naming + verbatim content.

    func testAppendS2SWritesDatedS2SFile() throws {
        let store = try makeStore()
        let records = [try s2sLine(sid: "u1"), try s2sLine(sid: "u2")]
        let url = try store.appendS2S(records: records)
        XCTAssertNotNil(url)
        guard let url else { return }
        XCTAssertTrue(url.lastPathComponent.hasPrefix(TesseraTraceStore.s2sFilePrefix))
        XCTAssertTrue(url.lastPathComponent.hasSuffix(".jsonl"))
        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertEqual(text, records.joined(separator: "\n") + "\n")
    }

    func testAppendS2SEmptyIsNoop() throws {
        let store = try makeStore()
        XCTAssertNil(try store.appendS2S(records: []))
        XCTAssertTrue(store.s2sFiles().isEmpty)
        XCTAssertEqual(store.totalRecords(), 0)
    }

    // Combined counting: s2s files keep the traces- prefix, so
    // totalRecords() sees them alongside every other provenance.

    func testS2SRecordsCountedInTotal() throws {
        let store = try makeStore()
        try store.appendS2S(records: [try s2sLine(sid: "u1"), try s2sLine(sid: "u2")])
        try store.appendRuntime(records: [
            "{\"schema\":\"llama.tessera.spec.v1\",\"drafted\":3,\"accepted\":2,\"provenance\":\"runtime\",\"sid\":\"r1\"}",
        ])
        XCTAssertEqual(store.totalRecords(), 3)
        XCTAssertEqual(store.s2sFiles().count, 1)
        XCTAssertEqual(store.runtimeFiles().count, 1)
    }

    // Rolling cap: the s2s trimmer removes OLDEST s2s files first and never
    // touches calibration, runtime, or replay files.

    func testS2STrimmingSparesOtherProvenances() throws {
        let store = try makeStore()
        let dir = FileManager.default.temporaryDirectory

        // Calibration file (appendRun) with one record.
        let calibrationSource = dir.appendingPathComponent("tessera-calib-\(UUID().uuidString).jsonl")
        dirs.append(calibrationSource)
        try ("{\"schema\":\"llama.tessera.spec.v1\",\"step\":0,\"drafted\":3,\"accepted\":2}\n")
            .write(to: calibrationSource, atomically: true, encoding: .utf8)
        let calibrationStored = try store.appendRun(jsonlPath: calibrationSource)

        // Runtime and replay files, written as their producers would.
        try store.appendRuntime(records: [
            "{\"schema\":\"llama.tessera.spec.v1\",\"drafted\":3,\"accepted\":2,\"provenance\":\"runtime\",\"sid\":\"r1\"}",
        ])
        let replay = store.directoryURL
            .appendingPathComponent("traces-replay-20260101-000000.jsonl")
        try ("{\"schema\":\"llama.tessera.spec.v1\",\"step\":0,\"provenance\":\"replay\",\"replayed_from\":\"runtime\"}\n")
            .write(to: replay, atomically: true, encoding: .utf8)

        // Three ~1 KB s2s files via the store API (same-second appends get
        // numeric suffixes; the identical sizes make the survivor choice
        // irrelevant, as in the runtime cap test).
        var sizes: [Int] = []
        for i in 0..<3 {
            let url = try store.appendS2S(records: [try s2sLine(sid: "s\(i)", pad: 900)])
            sizes.append((try String(contentsOf: url!, encoding: .utf8)).utf8.count)
        }

        // Budget keeps only one s2s file.
        let removed = try store.trimS2SToBudget(budgetBytes: sizes[2])
        XCTAssertEqual(removed, 2)

        XCTAssertEqual(store.s2sFiles().count, 1)
        let storedNames = store.traceFiles().map { $0.lastPathComponent }
        XCTAssertTrue(storedNames.contains(calibrationStored.lastPathComponent))
        XCTAssertTrue(storedNames.contains(replay.lastPathComponent))
        XCTAssertEqual(store.runtimeFiles().count, 1)
        XCTAssertEqual(store.totalRecords(), 4)  // calibration + runtime + replay + 1 s2s
    }

    // Quarantine exemption: a quarantined sid survives BOTH automatic
    // retention paths even when it is the oldest file.

    func testQuarantineExemptionFromS2SRollingCap() throws {
        let store = try makeStore()
        try writeS2SFile(store, name: "traces-s2s-20260101-000000.jsonl",
                         lines: [try s2sLine(sid: "Q", pad: 900)])
        try writeS2SFile(store, name: "traces-s2s-20260102-000000.jsonl",
                         lines: [try s2sLine(sid: "K", pad: 900)])

        // Budget fits one file; Q is oldest but quarantined, so K goes.
        let removed = try store.trimS2SToBudget(budgetBytes: 1000, exemptSids: ["Q"])
        XCTAssertEqual(removed, 1)

        let remaining = store.s2sFiles()
        XCTAssertEqual(remaining.count, 1)
        let text = try String(contentsOf: remaining[0], encoding: .utf8)
        XCTAssertTrue(text.contains("\"sid\":\"Q\""))
    }

    func testQuarantineExemptionFromRetentionCoversS2S() throws {
        let store = try makeStore()
        let old = Date().addingTimeInterval(-100 * 86_400)
        let q = try writeS2SFile(store, name: "traces-s2s-20260101-000000.jsonl",
                                 lines: [try s2sLine(sid: "Q")])
        let k = try writeS2SFile(store, name: "traces-s2s-20260102-000000.jsonl",
                                 lines: [try s2sLine(sid: "K")])
        for file in [q, k] {
            try FileManager.default.setAttributes(
                [.creationDate: old], ofItemAtPath: file.path)
        }

        let removed = try store.trimExpired(retentionDays: 30, exemptSids: ["Q"])
        XCTAssertEqual(removed, 1)  // K; Q exempt

        let survivors = store.s2sFiles()
        XCTAssertEqual(survivors.count, 1)
        let text = try String(contentsOf: survivors[0], encoding: .utf8)
        XCTAssertTrue(text.contains("\"sid\":\"Q\""))
    }

    func testTrimS2SKeepsFreshFilesUnderBudget() throws {
        let store = try makeStore()
        try store.appendS2S(records: [try s2sLine(sid: "N")])
        let removed = try store.trimS2SToBudget(budgetBytes: TesseraTraceStore.s2sBudgetBytesDefault)
        XCTAssertEqual(removed, 0)
        XCTAssertEqual(store.s2sFiles().count, 1)
    }

    // User-initiated purge is the only path that removes a quarantined
    // session, and it covers s2s records too.

    func testPurgeSessionRemovesS2SRecords() throws {
        let store = try makeStore()
        try store.appendRuntime(records: [
            "{\"schema\":\"llama.tessera.spec.v1\",\"drafted\":3,\"accepted\":2,\"provenance\":\"runtime\",\"sid\":\"X\"}",
        ])
        try store.appendS2S(records: [
            try s2sLine(sid: "X"),
            try s2sLine(sid: "Y"),
        ])

        let removed = try store.purgeSession(sid: "X")
        XCTAssertEqual(removed, 2)  // the runtime record plus the s2s record

        let s2sText = store.s2sFiles()
            .compactMap { try? String(contentsOf: $0, encoding: .utf8) }
            .joined()
        XCTAssertFalse(s2sText.contains("\"sid\":\"X\""))
        XCTAssertTrue(s2sText.contains("\"sid\":\"Y\""))
        XCTAssertEqual(store.runtimeFiles().count, 0)
    }
}

// MARK: - Default-on capture semantics (mandatory-collection doctrine)

final class TesseraS2SDefaultOnTests: XCTestCase {
    func testNoS2SOptOutSurfaceInRegisteredSettings() {
        // Code capture is default-on with NO opt-out. The registered settings
        // surface must not grow any s2s capture toggle; a future legitimate
        // s2s setting must revisit this pin deliberately.
        for key in TesseraSettings.registeredDefaults.keys {
            XCTAssertFalse(
                key.lowercased().contains("s2s"),
                "mandatory-collection doctrine forbids an s2s settings surface, found \(key)")
        }
    }

    func testAppendS2SCapturesEvenWhenRuntimeCaptureIsOff() throws {
        // The runtime capture toggle is the runtime traces' opt-out; s2s
        // capture is independent of it and always writes.
        let key = TesseraSettingsKey.learningRuntimeCapture
        let saved = UserDefaults.standard.object(forKey: key)
        UserDefaults.standard.set(false, forKey: key)
        defer {
            if let saved { UserDefaults.standard.set(saved, forKey: key) }
            else { UserDefaults.standard.removeObject(forKey: key) }
        }

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("tessera-s2s-default-on-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = TesseraTraceStore(directory: dir)

        let line = try TesseraS2SRecordTests.sampleRecord().jsonLine()
        let url = try store.appendS2S(records: [line])
        XCTAssertNotNil(url, "s2s capture is default-on: no gate may suppress it")
    }
}
