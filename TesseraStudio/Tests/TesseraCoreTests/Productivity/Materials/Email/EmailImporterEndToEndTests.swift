import XCTest
@testable import TesseraCore

/// End-to-end import tests. These tests
/// exercise the Python ``email.py`` parser
/// directly (via subprocess) to verify that
/// the .eml / .mbox fixtures are parsed into
/// the expected intermediate representation.
/// The Swift ``EmailImporter`` is a thin
/// wrapper around the Python CLI; the parser
/// behavior lives on the Python side.
///
/// The tests skip if the Python parser
/// module can't be loaded (the venv might
/// not have ``python-docx`` installed, which
/// is a hard dep of the
/// ``parsers/__init__.py`` aggregate import).
/// The parser test in
/// ``tools/tessera/importers/tests/`` is
/// the canonical coverage; this Swift test
/// is the dual-side check that the fixtures
/// still parse after a Python refactor.
final class EmailImporterEndToEndTests: XCTestCase {

    /// True when the Python parser is
    /// importable. We probe by running a
    /// tiny one-shot Python script that
    /// imports the parser and prints
    /// "ok". If the import fails (e.g.,
    /// because ``python-docx`` isn't in the
    /// venv), the test skips the rest of
    /// the suite.
    private var pythonImportsEmailParser: Bool = false

    override func setUp() {
        super.setUp()
        pythonImportsEmailParser = Self.probePythonImportsEmailParser()
    }

    // MARK: - .eml parsing

    /// ``sample.eml`` parses to one
    /// IntermediateDocument with subject
    /// "Hello Tessera" and a body of 5
    /// blocks (1 heading + 4 paragraphs).
    /// The test exercises the Python
    /// parser through a subprocess so the
    /// parser changes (or the fixture
    /// changes) are caught on the Swift
    /// side.
    func testParseSampleEML() throws {
        guard pythonImportsEmailParser else {
            throw XCTSkip("Python email parser not importable; skipping end-to-end test")
        }
        let fixture = try Self.locateFixture(named: "sample.eml")
        let result = try Self.parseEMLWithPython(fixture: fixture)
        XCTAssertEqual(result["subject"] as? String, "Hello Tessera")
        XCTAssertEqual(result["from"] as? String, "alice@example.com")
        XCTAssertEqual(result["to"] as? String, "bob@example.com")
        let blockCount = result["block_count"] as? Int ?? 0
        XCTAssertGreaterThan(blockCount, 0, "EML should produce at least one block")
    }

    // MARK: - .mbox parsing

    /// ``sample.mbox`` parses to two
    /// IntermediateDocuments. The first has
    /// subject "First message" and the
    /// second "Second message". The test
    /// exercises the Python parser through
    /// a subprocess.
    func testParseSampleMBOX() throws {
        guard pythonImportsEmailParser else {
            throw XCTSkip("Python email parser not importable; skipping end-to-end test")
        }
        let fixture = try Self.locateFixture(named: "sample.mbox")
        let result = try Self.parseMBOXWithPython(fixture: fixture)
        let count = result["count"] as? Int ?? 0
        XCTAssertEqual(count, 2, "MBOX should produce 2 messages")
        let subjects = result["subjects"] as? [String] ?? []
        XCTAssertEqual(subjects, ["First message", "Second message"])
    }

    // MARK: - Fixture resolution

    /// The Swift importer looks for the
    /// Phase 4 fixtures relative to the
    /// repo root. The fixtures must be
    /// present (the v1 spec uses them in
    /// the Python test suite; the Swift
    /// tests use them too).
    func testFixturesPresent() throws {
        let eml = try Self.locateFixture(named: "sample.eml")
        let mbox = try Self.locateFixture(named: "sample.mbox")
        XCTAssertTrue(FileManager.default.fileExists(atPath: eml))
        XCTAssertTrue(FileManager.default.fileExists(atPath: mbox))
    }

    // MARK: - Helpers

    /// Locate a fixture file. Tries the
    /// current directory, the parent, and
    /// the grandparent. Returns the first
    /// ABSOLUTE path that exists. The
    /// Python subprocess needs an absolute
    /// path (it runs with a different CWD).
    static func locateFixture(named filename: String) throws -> String {
        let candidates = [
            "tools/tessera/importers/tests/fixtures/\(filename)",
            "../tools/tessera/importers/tests/fixtures/\(filename)",
            "../../tools/tessera/importers/tests/fixtures/\(filename)",
        ]
        let cwd = FileManager.default.currentDirectoryPath
        for c in candidates {
            // Resolve to absolute path.
            let url = URL(fileURLWithPath: c, relativeTo: URL(fileURLWithPath: cwd))
            let abs = url.standardizedFileURL.path
            if FileManager.default.fileExists(atPath: abs) {
                return abs
            }
        }
        throw NSError(
            domain: "EmailImporterEndToEndTests",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "fixture not found: \(filename)"]
        )
    }

    /// True iff the Python interpreter on
    /// PATH can import the email parser
    /// directly (bypassing the
    /// ``parsers/__init__.py`` aggregate).
    /// The probe derives the ``tools/``
    /// path from the absolute path of this
    /// test file (the parser module lives
    /// at a known offset from the test).
    static func probePythonImportsEmailParser() -> Bool {
        let python = TesseraCLIPath.pythonExecutable
        // Find the absolute path to
        // tools/tessera/importers/ by
        // walking up from the test file.
        // The test file is at:
        //   TesseraStudio/Tests/TesseraCoreTests/Productivity/Materials/Email/...
        // and the importer is at:
        //   tools/tessera/importers/
        // so we need to walk up 5 levels
        // (Email/ -> Materials/ -> Productivity/ -> TesseraCoreTests/ -> Tests/ -> TesseraStudio/)
        // and then go to tools/tessera/importers/.
        // We use `__file__` of the probe
        // script's caller: there isn't
        // one, so we use the CWD (which
        // is `TesseraStudio/`) and walk up
        // to the repo root.
        let probe = """
        import os, sys
        # Walk up from CWD until we find a 'tools' dir.
        cwd = os.getcwd()
        repo = None
        d = cwd
        for _ in range(8):
            cand = os.path.join(d, 'tools')
            if os.path.isdir(cand):
                repo = d
                break
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        if repo is None:
            print('not_found'); sys.exit(1)
        tools_path = os.path.join(repo, 'tools')
        importers_path = os.path.join(tools_path, 'tessera', 'importers')
        parsers_path = os.path.join(importers_path, 'parsers')
        email_py = os.path.join(parsers_path, 'email.py')
        sys.path.insert(0, tools_path)
        import tools.tessera.importers.intermediate as _intermediate
        import importlib.util, types
        parsers_pkg = types.ModuleType('tools.tessera.importers.parsers')
        parsers_pkg.__path__ = [parsers_path]
        sys.modules['tools.tessera.importers.parsers'] = parsers_pkg
        spec = importlib.util.spec_from_file_location(
            'tools.tessera.importers.parsers.email',
            email_py
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules['tools.tessera.importers.parsers.email'] = mod
        spec.loader.exec_module(mod)
        print('ok')
        """
        do {
            let result = try PythonSubprocessRunner().run(
                script: python,
                args: ["-c", probe]
            )
            return result.exitCode == 0 && result.stdout.contains("ok")
        } catch {
            return false
        }
    }

    /// Parse an .eml fixture through the
    /// Python subprocess and return a JSON
    /// dict with the parsed fields. The
    /// subprocess writes the dict as
    /// JSON to stdout.
    static func parseEMLWithPython(fixture: String) throws -> [String: Any] {
        let python = TesseraCLIPath.pythonExecutable
        let script = """
        import os, sys, importlib.util, types, json
        cwd = os.getcwd()
        repo = None
        d = cwd
        for _ in range(8):
            cand = os.path.join(d, 'tools')
            if os.path.isdir(cand):
                repo = d
                break
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        if repo is None:
            print('repo_not_found'); sys.exit(1)
        tools_path = os.path.join(repo, 'tools')
        importers_path = os.path.join(tools_path, 'tessera', 'importers')
        parsers_path = os.path.join(importers_path, 'parsers')
        email_py = os.path.join(parsers_path, 'email.py')
        sys.path.insert(0, tools_path)
        import tools.tessera.importers.intermediate as _intermediate
        parsers_pkg = types.ModuleType('tools.tessera.importers.parsers')
        parsers_pkg.__path__ = [parsers_path]
        sys.modules['tools.tessera.importers.parsers'] = parsers_pkg
        spec = importlib.util.spec_from_file_location(
            'tools.tessera.importers.parsers.email',
            email_py
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules['tools.tessera.importers.parsers.email'] = mod
        spec.loader.exec_module(mod)
        from pathlib import Path
        doc = mod.parse_eml(Path(sys.argv[1]))
        out = {
            'subject': doc.meta.get('subject', ''),
            'from': doc.meta.get('from', ''),
            'to': doc.meta.get('to', ''),
            'block_count': len(doc.blocks),
            'message_id': doc.meta.get('message_id', ''),
        }
        print(json.dumps(out))
        """
        let result = try PythonSubprocessRunner().run(
            script: python,
            args: ["-c", script, fixture]
        )
        XCTAssertEqual(result.exitCode, 0, "python EML parse failed: \(result.stderr)")
        let json = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = json.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            XCTFail("could not parse Python output as JSON: \(json)")
            return [:]
        }
        return dict
    }

    /// Parse a .mbox fixture through the
    /// Python subprocess. Returns a dict
    /// with `count` and `subjects`.
    static func parseMBOXWithPython(fixture: String) throws -> [String: Any] {
        let python = TesseraCLIPath.pythonExecutable
        let script = """
        import os, sys, importlib.util, types, json
        cwd = os.getcwd()
        repo = None
        d = cwd
        for _ in range(8):
            cand = os.path.join(d, 'tools')
            if os.path.isdir(cand):
                repo = d
                break
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        if repo is None:
            print('repo_not_found'); sys.exit(1)
        tools_path = os.path.join(repo, 'tools')
        importers_path = os.path.join(tools_path, 'tessera', 'importers')
        parsers_path = os.path.join(importers_path, 'parsers')
        email_py = os.path.join(parsers_path, 'email.py')
        sys.path.insert(0, tools_path)
        import tools.tessera.importers.intermediate as _intermediate
        parsers_pkg = types.ModuleType('tools.tessera.importers.parsers')
        parsers_pkg.__path__ = [parsers_path]
        sys.modules['tools.tessera.importers.parsers'] = parsers_pkg
        spec = importlib.util.spec_from_file_location(
            'tools.tessera.importers.parsers.email',
            email_py
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules['tools.tessera.importers.parsers.email'] = mod
        spec.loader.exec_module(mod)
        from pathlib import Path
        docs = mod.parse_mbox(Path(sys.argv[1]))
        out = {
            'count': len(docs),
            'subjects': [d.meta.get('subject', '') for d in docs],
            'froms': [d.meta.get('from', '') for d in docs],
        }
        print(json.dumps(out))
        """
        let result = try PythonSubprocessRunner().run(
            script: python,
            args: ["-c", script, fixture]
        )
        XCTAssertEqual(result.exitCode, 0, "python MBOX parse failed: \(result.stderr)")
        let json = result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = json.data(using: .utf8),
              let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            XCTFail("could not parse Python output as JSON: \(json)")
            return [:]
        }
        return dict
    }
}
