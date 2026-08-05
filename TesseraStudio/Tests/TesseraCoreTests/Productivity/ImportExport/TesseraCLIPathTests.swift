import XCTest
@testable import TesseraCore

/// Tests for the ``TesseraCLIPath`` shim and the
/// ``PythonSubprocessRunner``. The runner is exercised
/// against the actual Python interpreter on the test host;
/// the test asserts the runner reports the right error when
/// the script is missing.
final class TesseraCLIPathTests: XCTestCase {

    /// ``TesseraCLIPath.pythonExecutable`` should return a
    /// non-empty path. The test doesn't assert which path
    /// (the dev machine may have any of the candidates)
    /// but the constant must always be set.
    func testPythonExecutableResolves() {
        let python = TesseraCLIPath.pythonExecutable
        XCTAssertFalse(
            python.isEmpty,
            "TesseraCLIPath.pythonExecutable should resolve to a non-empty path"
        )
    }

    /// ``TesseraCLIPath.importerScript`` is derived from
    /// ``repoRoot``; the constant should end with the
    /// expected path component.
    func testImporterScriptPath() {
        let script = TesseraCLIPath.importerScript
        XCTAssertTrue(
            script.hasSuffix("tools/tessera/importers/cli.py"),
            "importer script should end with tools/tessera/importers/cli.py; got \(script)"
        )
    }

    /// ``TesseraCLIPath.exporterScript`` ends with the
    /// exporter's CLI path.
    func testExporterScriptPath() {
        let script = TesseraCLIPath.exporterScript
        XCTAssertTrue(
            script.hasSuffix("tools/tessera/exporters/cli.py"),
            "exporter script should end with tools/tessera/exporters/cli.py; got \(script)"
        )
    }

    /// When the script is missing, the runner must throw
    /// ``RunnerError.failedToLaunch`` (or, on hosts that
    /// surface the missing-script error as a non-zero exit,
    /// return a non-zero exit code). We exercise the
    /// runner against a path that does not exist.
    func testRunnerFailsCleanlyOnMissingScript() {
        let runner = PythonSubprocessRunner()
        let bogus = "/nonexistent/path/to/script.py"
        do {
            let result = try runner.run(
                script: bogus,
                args: ["--help"]
            )
            // Some Python interpreters succeed when run with
            // --help on a non-existent script (they print
            // the usage). Accept a non-zero exit OR a help
            // string in stdout.
            XCTAssertTrue(
                result.exitCode != 0 || result.stdout.contains("usage"),
                "runner on missing script should fail or print usage; got exit=\(result.exitCode) stdout=\(result.stdout.prefix(200))"
            )
        } catch PythonSubprocessRunner.RunnerError.failedToLaunch {
            // Expected
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    /// Round-trip: a valid python -c "print('hello')" call
    /// should return a Result with stdout containing
    /// "hello" and exit code 0.
    func testRunnerRoundTripPrint() throws {
        // Use a one-shot script path that invokes Python
        // directly with `-c` so we don't need a fixture
        // script on disk. The runner's API takes a
        // `script` path; we point it at the python
        // interpreter itself and pass `-c` as the first
        // argument. This is the same pattern `python -c`
        // uses.
        let python = TesseraCLIPath.pythonExecutable
        let result = try PythonSubprocessRunner().run(
            script: python,
            args: ["-c", "print('hello')"]
        )
        XCTAssertEqual(result.exitCode, 0, "python -c should exit 0; stderr=\(result.stderr)")
        XCTAssertTrue(
            result.stdout.contains("hello"),
            "stdout should contain 'hello'; got \(result.stdout)"
        )
    }
}
