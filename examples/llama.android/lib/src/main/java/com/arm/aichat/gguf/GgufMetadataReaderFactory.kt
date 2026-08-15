package com.arm.aichat.gguf

import com.arm.aichat.internal.gguf.GgufMetadataReaderImpl

/**
 * Creates a default GgufMetadataReader instance
 */
fun GgufMetadataReader.Companion.create(): GgufMetadataReader = GgufMetadataReaderImpl(
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
fun GgufMetadataReader.Companion.create(
    skipKeys: Set<String> = DEFAULT_SKIP_KEYS,
    arraySummariseThreshold: Int = 1_000
): GgufMetadataReader = GgufMetadataReaderImpl(
    skipKeys = skipKeys,
    arraySummariseThreshold = arraySummariseThreshold
)
