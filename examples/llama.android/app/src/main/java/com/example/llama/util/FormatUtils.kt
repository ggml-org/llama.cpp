package com.example.llama.util

import java.util.Locale

/**
 * Formats milliseconds into a human-readable time string
 * e.g., 2300ms -> "2.3 sec"
 */
fun formatMilliSeconds(millis: Long): String {
    val seconds = millis / 1000.0
    return if (seconds < 1.0) {
        "${(seconds * 1000).toInt()} ms"
    } else if (seconds < 60.0) {
        "%.1f sec".format(seconds)
    } else {
        val minutes = seconds / 60.0
        "%.1f min".format(minutes)
    }
}

/**
 * Convert bytes into human readable sizes
 */
fun formatFileByteSize(sizeInBytes: Long) = when {
    sizeInBytes >= 1_000_000_000 -> {
        val sizeInGb = sizeInBytes / 1_000_000_000.0
        String.Companion.format(Locale.getDefault(), "%.1f GB", sizeInGb)
    }
    sizeInBytes >= 1_000_000 -> {
        val sizeInMb = sizeInBytes / 1_000_000.0
        String.Companion.format(Locale.getDefault(), "%.0f MB", sizeInMb)
    }
    else -> {
        val sizeInKb = sizeInBytes / 1_000.0
        String.Companion.format(Locale.getDefault(), "%.0f KB", sizeInKb)
    }
}

/**
 * Formats numbers to human-readable form (K, M)
 */
fun formatContextLength(contextLength: Int): String {
    return when {
        contextLength >= 1_000_000 -> String.Companion.format(Locale.getDefault(), "%.1fM", contextLength / 1_000_000.0)
        contextLength >= 1_000 -> String.Companion.format(Locale.getDefault(), "%.0fK", contextLength / 1_000.0)
        else -> contextLength.toString()
    }
}
