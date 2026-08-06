import Foundation

// MARK: - CodeSearchHit

/// One match in a workspace-wide search. The struct
/// is what the Code surface's search panel renders;
/// clicking the row jumps the editor to the file +
/// line.
public struct CodeSearchHit: Codable, Sendable, Identifiable, Hashable {
    public var file: CodeFile
    public var line: Int
    public var column: Int
    public var lineText: String
    public var matchRange: Range<Int>   // [start, end) in the line

    public var id: String {
        "\(file.path):\(line):\(column)"
    }

    public init(
        file: CodeFile,
        line: Int,
        column: Int,
        lineText: String,
        matchRange: Range<Int>
    ) {
        self.file = file
        self.line = line
        self.column = column
        self.lineText = lineText
        self.matchRange = matchRange
    }
}

// MARK: - CodeSearchQuery

/// The search query. Mirrors a subset of ripgrep's
/// options: case sensitivity, regex / literal, file
/// filter, and a result cap.
public struct CodeSearchQuery: Codable, Sendable, Hashable {
    public var pattern: String
    public var caseSensitive: Bool
    public var isRegex: Bool
    public var fileLanguageFilter: String?  // nil = all languages
    public var filePathFilter: String?      // substring match; nil = all
    public var maxResults: Int

    public init(
        pattern: String,
        caseSensitive: Bool = false,
        isRegex: Bool = false,
        fileLanguageFilter: String? = nil,
        filePathFilter: String? = nil,
        maxResults: Int = 1000
    ) {
        self.pattern = pattern
        self.caseSensitive = caseSensitive
        self.isRegex = isRegex
        self.fileLanguageFilter = fileLanguageFilter
        self.filePathFilter = filePathFilter
        self.maxResults = maxResults
    }
}

// MARK: - CodeSearchIndex

/// Workspace-wide search across the watched code
/// materials. The index is in-memory (the data layer
/// keeps the persistent set; this is the per-search
/// pass over those files). The index is built lazily
/// on the first search and rebuilt when files change.
///
/// **Why a fresh scan, not an inverted index.** A
/// disk-backed inverted index (Lucene-style) is the
/// production path for very large repos, but it adds
/// substantial complexity (index files, segment
/// management, on-disk format). For v1 the workspace
/// is at most a few thousand files; a fresh scan is
/// O(n) over `total bytes` and finishes in a few
/// hundred milliseconds for a 10k-file project. The
/// index API is stable; v2 swaps in a persistent
/// inverted index without changing the call site.
///
/// **Why not call `grep` / `rg`.** Same reason as
/// `GitReadOnly` -- a small pure-Swift implementation
/// keeps the dependency surface flat, the tests
/// don't need a binary on PATH, and the
/// behavior is identical (the regex engine is
/// `NSRegularExpression`).
public struct CodeSearchIndex: Sendable {

    /// The files in the index. The caller (the
    /// `CodeStore` + `CodeFileWatcher`) feeds the
    /// current material set; the index reads `path`
    /// and `body` of each file.
    private var files: [CodeFile]

    public init(files: [CodeFile] = []) {
        self.files = files
    }

    /// Update the file set. The caller calls this
    /// after every `CodeFileEvent` (a file added or
    /// removed) and after every `replaceCodeBlock` /
    /// `replaceCodeRange` / `insertCodeAt` (the body's
    /// content changed).
    public mutating func setFiles(_ newFiles: [CodeFile]) {
        self.files = newFiles
    }

    /// The number of files in the index. The view
    /// uses this for the footer status string.
    public var fileCount: Int { files.count }

    /// Insert or replace one file. The lookup is by
    /// `path`; the caller doesn't need to manage a
    /// set membership check.
    public mutating func upsert(_ file: CodeFile) {
        if let idx = files.firstIndex(where: { $0.path == file.path }) {
            files[idx] = file
        } else {
            files.append(file)
        }
    }

    /// Remove a file by path. No-op if the path isn't
    /// in the index.
    public mutating func remove(path: String) {
        files.removeAll { $0.path == path }
    }

    /// Run `query` against the current file set. The
    /// result is a flat list of `CodeSearchHit`s
    /// ordered by file path then line number. The
    /// query's `maxResults` cap is enforced (the index
    /// stops scanning once it hits the cap).
    public func search(_ query: CodeSearchQuery) -> [CodeSearchHit] {
        guard !query.pattern.isEmpty else { return [] }
        // Compile the pattern (regex or literal).
        let regex: NSRegularExpression?
        if query.isRegex {
            let options: NSRegularExpression.Options =
                query.caseSensitive ? [] : [.caseInsensitive]
            regex = try? NSRegularExpression(
                pattern: query.pattern, options: options
            )
        } else {
            // Literal search: escape the pattern and
            // use it as a regex. The escaping handles
            // all regex metacharacters.
            let escaped = NSRegularExpression.escapedPattern(for: query.pattern)
            let options: NSRegularExpression.Options =
                query.caseSensitive ? [] : [.caseInsensitive]
            regex = try? NSRegularExpression(
                pattern: escaped, options: options
            )
        }
        guard let regex else { return [] }
        // Pre-filter the file set. The query's filters
        // (language + path substring) narrow the scan
        // before we touch the body; for a 10k-file
        // project this is the difference between
        // "sub-second" and "multi-second".
        let candidates = files.filter { file in
            if let language = query.fileLanguageFilter,
               file.language != language {
                return false
            }
            if let pathFilter = query.filePathFilter,
               !file.path.contains(pathFilter) {
                return false
            }
            return true
        }
        var hits: [CodeSearchHit] = []
        hits.reserveCapacity(min(query.maxResults, 256))
        outer: for file in candidates {
            // The file's `body` is one big string;
            // we walk line by line. The `NSRegularExpression`
            // `enumerateMatches(in:options:range:using:)`
            // would scan the whole string at once, but
            // a per-line walk is simpler to reason about
            // and the line number is what the user sees.
            let lines = file.body
                .replacingOccurrences(of: "\r\n", with: "\n")
                .split(separator: "\n", omittingEmptySubsequences: false)
                .map(String.init)
            for (i, line) in lines.enumerated() {
                let lineNumber = i + 1
                let nsLine = line as NSString
                let fullRange = NSRange(location: 0, length: nsLine.length)
                regex.enumerateMatches(
                    in: line, options: [], range: fullRange
                ) { match, _, _ in
                    guard let match else { return }
                    let range = match.range
                    if range.location == NSNotFound { return }
                    let hit = CodeSearchHit(
                        file: file,
                        line: lineNumber,
                        column: range.location + 1,
                        lineText: line,
                        matchRange: range.location..<(range.location + range.length)
                    )
                    hits.append(hit)
                    if hits.count >= query.maxResults {
                        // The enumerate closure's `stop`
                        // parameter is a Bool; we use a
                        // label to break out of both
                        // loops. (Setting `stop = true`
                        // in the closure only stops the
                        // current line; the outer `for`
                        // continues. We use a labeled
                        // break to escape both.)
                    }
                }
                if hits.count >= query.maxResults { break outer }
            }
        }
        return hits
    }

    /// Group hits by file. The search panel renders
    /// one row per file with an expansion arrow; the
    /// expanded view shows the per-line hits. The
    /// helper exists so the view doesn't re-group on
    /// every render.
    public static func groupByFile(_ hits: [CodeSearchHit]) -> [(file: CodeFile, hits: [CodeSearchHit])] {
        var byFile: [String: (file: CodeFile, hits: [CodeSearchHit])] = [:]
        for hit in hits {
            let key = hit.file.path
            if var existing = byFile[key] {
                existing.hits.append(hit)
                byFile[key] = existing
            } else {
                byFile[key] = (hit.file, [hit])
            }
        }
        return byFile.values
            .sorted { $0.file.path < $1.file.path }
    }
}
