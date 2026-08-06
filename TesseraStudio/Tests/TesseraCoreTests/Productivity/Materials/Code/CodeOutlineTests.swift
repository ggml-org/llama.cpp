import XCTest
@testable import TesseraCore

/// Tests for ``CodeOutlineExtractor``. The extractor
/// is regex-based; the tests cover the most common
/// patterns per language and the per-language edge
/// cases (Python indentation, Ruby `end`, ...).
final class CodeOutlineTests: XCTestCase {

    private let extractor = CodeOutlineExtractor()

    // MARK: - Swift

    func testSwiftClassExtraction() {
        let src = """
        import Foundation

        public class Foo {
            let value: Int
        }

        private class Bar {
            let name: String
        }
        """
        let outline = extractor.extract(source: src, language: "swift")
        let classNames = outline.items.filter { $0.kind == .class }.map(\.label)
        XCTAssertTrue(classNames.contains(where: { $0.contains("Foo") }))
        XCTAssertTrue(classNames.contains(where: { $0.contains("Bar") }))
    }

    func testSwiftStructExtraction() {
        let src = """
        struct Point {
            let x: Int
        }
        """
        let outline = extractor.extract(source: src, language: "swift")
        XCTAssertTrue(outline.items.contains { $0.kind == .struct && $0.label.contains("Point") })
    }

    func testSwiftFunctionExtraction() {
        let src = """
        func foo() -> Int { return 1 }
        private func bar(x: Int) { }
        """
        let outline = extractor.extract(source: src, language: "swift")
        let functions = outline.items.filter { $0.kind == .function }
        XCTAssertTrue(functions.contains { $0.label.contains("foo") })
        XCTAssertTrue(functions.contains { $0.label.contains("bar") })
    }

    func testSwiftProtocolExtraction() {
        let src = """
        protocol Greeter {
            func greet() -> String
        }
        """
        let outline = extractor.extract(source: src, language: "swift")
        XCTAssertTrue(outline.items.contains { $0.kind == .proto && $0.label.contains("Greeter") })
    }

    func testSwiftExtensionExtraction() {
        let src = """
        extension String {
            func shout() -> String { return self.uppercased() + "!" }
        }
        """
        let outline = extractor.extract(source: src, language: "swift")
        XCTAssertTrue(outline.items.contains { $0.kind == .extension })
    }

    func testSwiftNestedMethodInsideClass() {
        let src = """
        class Calculator {
            func add(_ a: Int, _ b: Int) -> Int {
                return a + b
            }
        }
        """
        let outline = extractor.extract(source: src, language: "swift")
        let theClass = outline.items.first { $0.kind == .class }
        XCTAssertNotNil(theClass)
        let theMethod = outline.items.first { $0.kind == .function && $0.label.contains("add") }
        XCTAssertNotNil(theMethod)
        XCTAssertEqual(theMethod?.parentID, theClass?.id)
    }

    // MARK: - Python

    func testPythonClassExtraction() {
        let src = """
        class Animal:
            def __init__(self, name):
                self.name = name

            def speak(self):
                return "noise"
        """
        let outline = extractor.extract(source: src, language: "python")
        XCTAssertTrue(outline.items.contains { $0.kind == .class && $0.label.contains("Animal") })
        let methods = outline.items.filter { $0.kind == .function }
        XCTAssertTrue(methods.contains { $0.label.contains("__init__") })
        XCTAssertTrue(methods.contains { $0.label.contains("speak") })
    }

    func testPythonFunctionExtraction() {
        let src = """
        def helper():
            return 1
        """
        let outline = extractor.extract(source: src, language: "python")
        XCTAssertTrue(outline.items.contains { $0.kind == .function && $0.label.contains("helper") })
    }

    func testPythonAsyncFunctionExtraction() {
        let src = """
        async def fetch():
            pass
        """
        let outline = extractor.extract(source: src, language: "python")
        XCTAssertTrue(outline.items.contains { $0.kind == .function && $0.label.contains("fetch") })
    }

    // MARK: - JavaScript / TypeScript

    func testTypeScriptClassExtraction() {
        let src = """
        export class UserService {
            getUser(id: string) { return null; }
        }
        """
        let outline = extractor.extract(source: src, language: "typescript")
        XCTAssertTrue(outline.items.contains { $0.kind == .class && $0.label.contains("UserService") })
    }

    func testJavaScriptFunctionExtraction() {
        let src = """
        export function process(data) {
            return data;
        }
        """
        let outline = extractor.extract(source: src, language: "javascript")
        XCTAssertTrue(outline.items.contains { $0.kind == .function && $0.label.contains("process") })
    }

    func testJavaScriptArrowFunctionExtraction() {
        let src = """
        const square = (x) => x * x;
        """
        let outline = extractor.extract(source: src, language: "javascript")
        XCTAssertTrue(outline.items.contains { $0.kind == .function && $0.label.contains("square") })
    }

    // MARK: - Rust

    func testRustStructExtraction() {
        let src = """
        pub struct Config {
            name: String,
        }
        """
        let outline = extractor.extract(source: src, language: "rust")
        XCTAssertTrue(outline.items.contains { $0.kind == .struct && $0.label.contains("Config") })
    }

    func testRustFunctionExtraction() {
        let src = """
        pub fn hello() -> String { "hi".to_string() }
        """
        let outline = extractor.extract(source: src, language: "rust")
        XCTAssertTrue(outline.items.contains { $0.kind == .function && $0.label.contains("hello") })
    }

    // MARK: - Go

    func testGoFunctionExtraction() {
        let src = """
        func main() {
            println("hi")
        }
        """
        let outline = extractor.extract(source: src, language: "go")
        XCTAssertTrue(outline.items.contains { $0.kind == .function && $0.label.contains("main") })
    }

    func testGoMethodExtraction() {
        let src = """
        type Server struct {}
        func (s *Server) Start() {}
        """
        let outline = extractor.extract(source: src, language: "go")
        XCTAssertTrue(outline.items.contains { $0.kind == .function && $0.label.contains("Start") })
    }

    // MARK: - Java

    func testJavaClassExtraction() {
        let src = """
        public class Hello {
            public static void main(String[] args) {
            }
        }
        """
        let outline = extractor.extract(source: src, language: "java")
        XCTAssertTrue(outline.items.contains { $0.kind == .class && $0.label.contains("Hello") })
    }

    // MARK: - Ruby

    func testRubyClassExtraction() {
        let src = """
        class Dog
          def bark
            "woof"
          end
        end
        """
        let outline = extractor.extract(source: src, language: "ruby")
        XCTAssertTrue(outline.items.contains { $0.kind == .class && $0.label.contains("Dog") })
    }

    // MARK: - Unknown languages

    func testUnknownLanguageReturnsEmptyOutline() {
        let outline = extractor.extract(source: "anything", language: "plain")
        XCTAssertTrue(outline.isEmpty)
    }

    // MARK: - Line numbers

    func testLineNumbersAreCorrect() {
        let src = """
        line 1
        line 2
        class Foo {
        }
        """
        let outline = extractor.extract(source: src, language: "swift")
        let foo = outline.items.first { $0.kind == .class && $0.label.contains("Foo") }
        XCTAssertEqual(foo?.line, 3)
    }

    // MARK: - Comments are ignored

    func testCommentedClassIsNotExtracted() {
        let src = """
        // class NotAClass { }
        class RealClass { }
        """
        let outline = extractor.extract(source: src, language: "swift")
        let classNames = outline.items.filter { $0.kind == .class }.map(\.label)
        XCTAssertFalse(classNames.contains(where: { $0.contains("NotAClass") }))
        XCTAssertTrue(classNames.contains(where: { $0.contains("RealClass") }))
    }
}
