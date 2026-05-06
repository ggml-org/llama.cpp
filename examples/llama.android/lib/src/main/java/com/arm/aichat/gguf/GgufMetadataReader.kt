package com.arm.aichat.gguf

import android.content.Context
import android.net.Uri
import java.io.File
import java.io.IOException
import java.io.InputStream

/**
 * Interface for reading GGUF metadata from model files.
 * Use `GgufMetadataReader.create()` to get an instance.
 */
interface GgufMetadataReader {
    /**
     * Reads the magic number from the specified file path.
     *
     * @param file Java File to the GGUF file with absolute path
     * @return true if file is valid GGUF, otherwise false
     * @throws InvalidFileFormatException if file format is invalid
     */
    suspend fun ensureSourceFileFormat(file: File): Boolean

    /**
     * Reads the magic number from the specified file path.
     *
     * @param context Context for obtaining [android.content.ContentProvider]
     * @param uri Uri to the GGUF file provided by [android.content.ContentProvider]
     * @return true if file is valid GGUF, otherwise false
     * @throws InvalidFileFormatException if file format is invalid
     */
    suspend fun ensureSourceFileFormat(context: Context, uri: Uri): Boolean

    /**
     * Reads and parses GGUF metadata from the specified file path.
     *
     * @param input the [InputStream] obtained from a readable file or content
     * @return Structured metadata extracted from the file
     * @throws IOException if file is damaged or cannot be read
     * @throws InvalidFileFormatException if file format is invalid
     */
    suspend fun readStructuredMetadata(input: InputStream): GgufMetadata

    companion object {
        val DEFAULT_SKIP_KEYS = setOf(
            "tokenizer.chat_template",
            "tokenizer.ggml.scores",
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.token_type"
        )
    }
}

class InvalidFileFormatException : IOException()
