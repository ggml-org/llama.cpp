import Foundation

/// Defense-in-depth overwrite helper. Used by the volume reset path to
/// overwrite the sparse bundle's bands with random bytes before the
/// directory is removed.
///
/// The crypto-shred property comes from deleting the Keychain entry -
/// once the key is gone, the encrypted bundle is unrecoverable
/// regardless of what its bytes contain. The overwrite is the spec's
/// "defense in depth" step (Section 7.1, steps 6-7): it protects
/// against the residual concern that a future AES-256 cryptanalytic
/// breakthrough would expose the ciphertext.
///
/// Scope: the helper walks the directory and overwrites every regular
/// file in place. Sparse bundles hold the actual band data in
/// `bands/0`, `bands/1`, ...; the `Info.plist` and `token` are tiny
/// metadata and are overwritten too, though the OS recreates them
/// when it next opens the bundle.
///
/// Threading: synchronous file I/O. Bundle size is bounded (the dev
/// preview is 1 GiB; 3 passes = 3 GiB of writes) and the call sites
/// are reset flows, not the hot path.
public enum SecureOverwrite {

    /// Overwrite every file under `root` with `passes` rounds of
    /// random data. The file is left at its original size; the
    /// caller is responsible for deleting the directory after this
    /// returns.
    ///
    /// On any per-file error the helper continues to the next file
    /// and surfaces the first error in the returned value. The
    /// wipe must continue even if some bands cannot be written -
    /// crypto-shred has already happened by the time this is called.
    public static func randomPasses(under root: URL, passes: Int) async throws {
        try await Task.detached(priority: .utility) {
            try randomPassesSync(under: root, passes: passes)
        }.value
    }

    /// Synchronous variant for tests that want to assert the call's
    /// effect without awaiting the actor hop.
    public static func randomPassesSync(under root: URL, passes: Int) throws {
        let fm = FileManager.default
        guard fm.fileExists(atPath: root.path) else { return }
        guard let enumerator = fm.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return }

        var firstError: Error?
        for case let fileURL as URL in enumerator {
            let resourceValues = try? fileURL.resourceValues(
                forKeys: [.isRegularFileKey, .fileSizeKey]
            )
            guard resourceValues?.isRegularFile == true else { continue }
            let size = resourceValues?.fileSize ?? 0
            guard size > 0 else { continue }

            do {
                try overwriteFile(at: fileURL, size: size, passes: passes)
            } catch {
                if firstError == nil { firstError = error }
            }
        }
        if let err = firstError { throw err }
    }

    /// Overwrite a single file. Opens the file in write mode (which
    /// truncates to size 0) and writes `size` random bytes, `passes`
    /// times, then fsyncs.
    private static func overwriteFile(at url: URL, size: Int, passes: Int) throws {
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        // 1 MiB buffer is a reasonable balance between syscall count
        // and memory pressure for a 1 GiB bundle.
        let bufferSize = min(size, 1024 * 1024)
        var buffer = Data(count: bufferSize)
        for _ in 0..<passes {
            try handle.seek(toOffset: 0)
            var written = 0
            while written < size {
                let chunk = min(bufferSize, size - written)
                try randomBytes(into: &buffer, count: chunk)
                try handle.write(contentsOf: buffer.prefix(chunk))
                written += chunk
            }
            try handle.synchronize()
        }
        // Truncate back to the original size; the caller decides
        // whether to delete the file.
        try handle.truncate(atOffset: UInt64(size))
    }

    /// Fill the first `count` bytes of `buffer` with random data from
    /// `arc4random_buf`. We use the C builtin rather than
    /// `SecRandomCopyBytes` because it's simpler and equally suitable
    /// for the overwrite (the random data here is not security-bearing;
    /// the crypto-shred has already happened by the time this runs).
    private static func randomBytes(into buffer: inout Data, count: Int) {
        buffer.withUnsafeMutableBytes { rawBuf in
            guard let base = rawBuf.baseAddress else { return }
            arc4random_buf(base, count)
        }
    }
}
