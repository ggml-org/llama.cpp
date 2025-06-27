package android.llama.cpp.gguf

import android.llama.cpp.internal.gguf.GgufMetadataReaderImpl
import java.io.IOException

/**
 * Interface for reading GGUF metadata from model files.
 * Use `GgufMetadataReader.create()` to get an instance.
 */
interface GgufMetadataReader {
    /**
     * Reads and parses GGUF metadata from the specified file path.
     *
     * @param path The absolute path to the GGUF file
     * @return Structured metadata extracted from the file
     * @throws IOException if file cannot be read
     * @throws IllegalArgumentException if file format is invalid
     */
    suspend fun readStructuredMetadata(path: String): GgufMetadata

    companion object {
        private val DEFAULT_SKIP_KEYS = setOf(
            "tokenizer.chat_template",
            "tokenizer.ggml.scores",
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.token_type"
        )

        /**
         * Creates a default GgufMetadataReader instance
         */
        fun create(): GgufMetadataReader = GgufMetadataReaderImpl(
            skipKeys = DEFAULT_SKIP_KEYS,
            arraySummariseThreshold = 1_000
        )

        /**
         * Creates a GgufMetadataReader with custom configuration
         *
         * @param skipKeys Keys whose value should be skipped entirely (not kept in the result map)
         * @param arraySummariseThreshold If ≥0, arrays longer get summarised, not materialised;
         *                                If -1, never summarise.
         */
        fun create(
            skipKeys: Set<String> = DEFAULT_SKIP_KEYS,
            arraySummariseThreshold: Int = 1_000
        ): GgufMetadataReader = GgufMetadataReaderImpl(
            skipKeys = skipKeys,
            arraySummariseThreshold = arraySummariseThreshold
        )
    }
}
