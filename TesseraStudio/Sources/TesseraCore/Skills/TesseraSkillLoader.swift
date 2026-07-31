import Foundation

/// Filesystem loader for agent skills (absorption I1). Scans one or more
/// directories for `SKILL.md` (and `<name>/SKILL.md`), parses each manifest
/// via `TesseraSkill`, and keeps only skills whose folder name matches the
/// frontmatter `name` - a mismatch is skipped and logged. Matching skills
/// can be rendered into a system-prompt fragment on demand. This is markdown
/// plus a loader: no new abstraction, no new subsystem.
public struct TesseraSkillLoader: Sendable {
    public let searchDirectories: [URL]

    public init(searchDirectories: [URL]? = nil) {
        self.searchDirectories = searchDirectories ?? Self.defaultSearchDirectories()
    }

    /// Default search dirs: skills bundled inside the module resource bundle
    /// (if any are shipped) and user-authored skills under Documents.
    public static func defaultSearchDirectories() -> [URL] {
        var dirs: [URL] = []
        if let bundled = Bundle.module.resourceURL?.appendingPathComponent("Skills", isDirectory: true) {
            dirs.append(bundled)
        }
        if let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            dirs.append(documents.appendingPathComponent("TesseraStudio/Skills", isDirectory: true))
        }
        return dirs
    }

    /// All valid skills found across the search directories, deduped by name
    /// (first occurrence wins, so earlier directories shadow later ones).
    public func skills() -> [TesseraSkill] {
        var result: [TesseraSkill] = []
        var seen: Set<String> = []
        for skill in scan() where seen.insert(skill.name).inserted {
            result.append(skill)
        }
        return result
    }

    public func skill(named name: String) -> TesseraSkill? {
        skills().first { $0.name == name }
    }

    /// The bodies of every skill whose name, description, or When-to-Use
    /// text matches the query (case-insensitive substring), joined into a
    /// block suitable for injection into the system prompt. Empty when
    /// nothing matches or the query is blank.
    public func systemPromptFragment(for query: String) -> String {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !needle.isEmpty else { return "" }

        let matches = skills().filter { skill in
            let haystack = "\(skill.name) \(skill.description) \(skill.whenToUse)".lowercased()
            return haystack.contains(needle)
        }
        guard !matches.isEmpty else { return "" }

        return matches
            .map { "## Skill: \($0.name)\n\n\($0.rawBody)" }
            .joined(separator: "\n\n")
    }

    // MARK: Scanning

    private func scan() -> [TesseraSkill] {
        var result: [TesseraSkill] = []
        let fm = FileManager.default

        for dir in searchDirectories {
            // A SKILL.md placed directly in a search directory.
            let direct = dir.appendingPathComponent("SKILL.md")
            if fm.fileExists(atPath: direct.path) {
                load(direct, expectedName: dir.lastPathComponent, into: &result)
            }

            // A `<name>/SKILL.md` for each immediate subdirectory.
            let entries = (try? fm.contentsOfDirectory(
                at: dir,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )) ?? []
            for entry in entries.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                let isDirectory = (try? entry.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory ?? false
                guard isDirectory else { continue }
                let manifest = entry.appendingPathComponent("SKILL.md")
                guard fm.fileExists(atPath: manifest.path) else { continue }
                load(manifest, expectedName: entry.lastPathComponent, into: &result)
            }
        }
        return result
    }

    private func load(_ url: URL, expectedName: String, into result: inout [TesseraSkill]) {
        let skill: TesseraSkill
        do {
            skill = try TesseraSkill.parse(contentsOf: url)
        } catch {
            Self.log("skipped \(url.path): \(error.localizedDescription)")
            return
        }
        guard skill.name == expectedName else {
            Self.log("skipped \(url.path): directory '\(expectedName)' != frontmatter name '\(skill.name)'")
            return
        }
        result.append(skill)
    }

    // MARK: Logging

    /// One-line diagnostic for a skipped manifest, gated by the app's
    /// configured log level (a skip is a warning-level event).
    private static func log(_ message: String) {
        guard Self.showsWarnings else { return }
        let line = "[TesseraSkillLoader] \(message)\n"
        if let data = line.data(using: .utf8) {
            FileHandle.standardError.write(data)
        }
    }

    private static var showsWarnings: Bool {
        switch TesseraSettings.logLevel {
        case .debug, .info, .warning: return true
        case .error: return false
        }
    }
}
