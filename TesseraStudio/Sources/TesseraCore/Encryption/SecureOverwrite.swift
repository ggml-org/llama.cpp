import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Overwrites a file with random bytes (N passes) and then zeros, before
/// unlinking it. Defense in depth: even if a future attack breaks AES-256
/// (not realistic on the 10-20 year horizon, but documented as a residual
/// risk in the design), the original ciphertext on disk is gone.
///
/// Modern SSDs cannot guarantee that overwritten blocks are actually
/// erased at the physical layer (the FTL may keep copies for wear
/// leveling). That is the entire reason crypto-shred exists. The
/// overwrite here is the second line of defense AFTER the key has
/// already been destroyed - the data is already cryptographically
/// unrecoverable; the overwrite is belt-and-suspenders for the
/// "what if AES falls tomorrow" worry.
public enum SecureOverwrite {

    /// Errors the overwrite can produce. The executor maps these to
    /// `WipeOutcome.partialFailure` and continues.
    public enum OverwriteError: Error, LocalizedError {
        case statFailed(path: String, underlying: Int32)
        case openFailed(path: String, underlying: Int32)
        case writeFailed(path: String, underlying: Int32)
        case fsyncFailed(path: String, underlying: Int32)
        case unlinkFailed(path: String, underlying: Int32)
        case parentFsyncFailed(path: String, underlying: Int32)

        public var errorDescription: String? {
            switch self {
            case .statFailed(let p, let c):
                return "stat(\(p)): errno=\(c)"
            case .openFailed(let p, let c):
                return "open(\(p)): errno=\(c)"
            case .writeFailed(let p, let c):
                return "write(\(p)): errno=\(c)"
            case .fsyncFailed(let p, let c):
                return "fsync(\(p)): errno=\(c)"
            case .unlinkFailed(let p, let c):
                return "unlink(\(p)): errno=\(c)"
            case .parentFsyncFailed(let p, let c):
                return "fsync(parent of \(p)): errno=\(c)"
            }
        }
    }

    /// Injectable random source so tests can stage deterministic data.
    /// Production calls `arc4random_buf` directly.
    public typealias RandomFill = (UnsafeMutableRawPointer, Int) -> Void

    /// Default random fill: `arc4random_buf` on Darwin, the system RNG
    /// on Linux. Pulled out so tests can substitute a known sequence.
    public static let defaultRandomFill: RandomFill = { ptr, n in
        #if canImport(Darwin)
        arc4random_buf(ptr, n)
        #else
        // Linux / non-Apple: best-effort. The executor itself is
        // macOS-only at runtime; this branch exists so the type
        // compiles in `swift test` on Linux CI.
        for i in 0..<n {
            ptr.storeBytes(of: UInt8.random(in: 0...UInt8.max), toByteOffset: i, as: UInt8.self)
        }
        #endif
    }

    /// Overwrite each path with `passes` random passes plus a final
    /// zero pass, fsync, then unlink. The parent's fsync runs once
    /// at the end (the executor does that in its own step 8).
    public static func randomPassesThenDelete(
        paths: [URL],
        passes: Int = 3,
        randomFill: RandomFill = defaultRandomFill
    ) throws {
        for url in paths {
            try overwriteAndDelete(url: url, passes: passes, delete: true, randomFill: randomFill)
        }
    }

    /// Walk `root` recursively and overwrite every regular file in
    /// place with `passes` random passes plus a final zero pass, then
    /// fsync. The file is left on disk at its original size - this
    /// helper does not delete anything. Used by the volume migrator
    /// to scrub the sandbox before copying data into the new bundle,
    /// and by the volume reset path to scrub the bands before
    /// recreating the bundle.
    public static func randomPasses(
        under root: URL,
        passes: Int = 3,
        randomFill: RandomFill = defaultRandomFill
    ) throws {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return }
        for case let fileURL as URL in enumerator {
            let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey])
            guard values?.isRegularFile == true else { continue }
            try overwriteAndDelete(url: fileURL, passes: passes, delete: false, randomFill: randomFill)
        }
    }

    /// Synchronous variant of ``randomPasses(under:passes:randomFill:)``
    /// for callers that are not in an async context (the volume
    /// migrator's pre-migration scrub). Same semantics: overwrite in
    /// place, do not delete.
    public static func randomPassesSync(
        under root: URL,
        passes: Int = 1,
        randomFill: RandomFill = defaultRandomFill
    ) throws {
        try randomPasses(under: root, passes: passes, randomFill: randomFill)
    }

    /// fsync the directory that contains `path`. Important on Unix:
    /// deleting a file does not guarantee the directory entry is
    /// durable until the directory itself is fsynced.
    public static func fsyncParentDirectory(of path: URL) throws {
        let parent = path.deletingLastPathComponent()
        let fd = parent.withUnsafeFileSystemRepresentation { ptr -> Int32 in
            guard let ptr = ptr else { return -1 }
            return open(ptr, O_RDONLY)
        }
        guard fd >= 0 else {
            throw OverwriteError.parentFsyncFailed(path: parent.path, underlying: currentErrno())
        }
        defer { close(fd) }
        if fsync(fd) != 0 {
            throw OverwriteError.parentFsyncFailed(path: parent.path, underlying: currentErrno())
        }
    }

    // MARK: - internals

    /// Read the current `errno` value. Swift imports `errno` as a
    /// global `Int32` rather than as a function, so the canonical
    /// way to read it is via `__error()`.
    @inline(never)
    private static func currentErrno() -> Int32 {
        return __error().pointee
    }

    private static func overwriteAndDelete(
        url: URL,
        passes: Int,
        delete: Bool,
        randomFill: RandomFill
    ) throws {
        let path = url.path
        // 1. stat to get the size. If the file is already gone, treat
        // it as success (the user / a prior wipe may have removed it).
        var st = stat()
        let statRC = path.withCString { stat($0, &st) }
        if statRC != 0 {
            if currentErrno() == ENOENT { return }
            throw OverwriteError.statFailed(path: path, underlying: currentErrno())
        }
        let size = Int(st.st_size)
        if size == 0 {
            if delete { try unlink(path: path) }
            return
        }

        // 2. Open for writing. O_NOFOLLOW so a symlinked target does
        // not redirect the wipe to a different file.
        let fd = path.withCString { ptr -> Int32 in
            open(ptr, O_WRONLY | O_NOFOLLOW)
        }
        guard fd >= 0 else {
            if currentErrno() == ENOENT { return }
            throw OverwriteError.openFailed(path: path, underlying: currentErrno())
        }
        defer { close(fd) }

        // 3. N random passes. Buffer is 1 MiB; loop until size is
        // written. randomFill never returns early.
        let chunk = 1 << 20
        var buffer = [UInt8](repeating: 0, count: chunk)
        for _ in 0..<passes {
            try writeRandomPass(fd: fd, size: size, chunk: chunk,
                                buffer: &buffer, randomFill: randomFill)
            if lseek(fd, 0, SEEK_SET) == -1 {
                throw OverwriteError.writeFailed(path: path, underlying: currentErrno())
            }
        }
        // 4. Final zero pass.
        for i in 0..<buffer.count { buffer[i] = 0 }
        try writeRandomPass(fd: fd, size: size, chunk: chunk,
                            buffer: &buffer, randomFill: randomFill)

        // 5. fsync, truncate to 0, close.
        if fsync(fd) != 0 {
            throw OverwriteError.fsyncFailed(path: path, underlying: currentErrno())
        }
        if delete {
            if ftruncate(fd, 0) != 0 {
                throw OverwriteError.writeFailed(path: path, underlying: currentErrno())
            }
        }
        // 6. Unlink (only when requested).
        if delete {
            try unlink(path: path)
        }
    }

    private static func writeRandomPass(
        fd: Int32,
        size: Int,
        chunk: Int,
        buffer: inout [UInt8],
        randomFill: RandomFill
    ) throws {
        if lseek(fd, 0, SEEK_SET) == -1 {
            throw OverwriteError.writeFailed(path: "<fd=\(fd)>", underlying: currentErrno())
        }
        var written = 0
        while written < size {
            let n = min(chunk, size - written)
            randomFill(&buffer, n)
            // Zero the tail of the buffer so a previous pass's
            // residue does not leak past this write's end.
            for i in n..<buffer.count { buffer[i] = 0 }
            let rc = buffer.withUnsafeBufferPointer { ptr -> Int in
                let base = ptr.baseAddress!
                var total = 0
                while total < n {
                    let r = write(fd, base.advanced(by: total), n - total)
                    if r < 0 {
                        if currentErrno() == EINTR { continue }
                        return -1
                    }
                    total += r
                }
                return total
            }
            if rc < 0 {
                throw OverwriteError.writeFailed(path: "<fd=\(fd)>", underlying: currentErrno())
            }
            written += rc
        }
    }

    private static func unlink(path: String) throws {
        let rc = path.withCString { Darwin.unlink($0) }
        if rc != 0 && currentErrno() != ENOENT {
            throw OverwriteError.unlinkFailed(path: path, underlying: currentErrno())
        }
    }
}
