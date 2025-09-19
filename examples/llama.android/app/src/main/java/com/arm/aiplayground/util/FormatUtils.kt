package com.arm.aiplayground.util

import java.util.concurrent.TimeUnit
import java.util.Locale
import kotlin.math.round

/**
 * Maps [TimeUnit] to English names, with plural form support
 */
fun TimeUnit.toEnglishName(plural: Boolean = false): String = when (this) {
    TimeUnit.NANOSECONDS  -> if (plural) "nanoseconds" else "nanosecond"
    TimeUnit.MICROSECONDS -> if (plural) "microseconds" else "microsecond"
    TimeUnit.MILLISECONDS -> if (plural) "milliseconds" else "millisecond"
    TimeUnit.SECONDS      -> if (plural) "seconds" else "second"
    TimeUnit.MINUTES      -> if (plural) "minutes" else "minute"
    TimeUnit.HOURS        -> if (plural) "hours" else "hour"
    TimeUnit.DAYS         -> if (plural) "days" else "day"
}

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

data class DurationValue(
    val value: Double,
    val unit: TimeUnit
)

/**
 * Converts milliseconds into a structured DurationValue.
 *
 * Rules:
 *  - < 100 seconds -> show in SECONDS
 *  - < 100 minutes -> show in MINUTES
 *  - < 100 hours   -> show in HOURS
 */
fun formatMilliSecondstructured(millis: Long): DurationValue {
    val seconds = millis / 1000.0
    return when {
        seconds < 100 -> DurationValue(round2(seconds), TimeUnit.SECONDS)
        seconds < 100 * 60 -> DurationValue(round2(seconds / 60.0), TimeUnit.MINUTES)
        else -> DurationValue(round2(seconds / 3600.0), TimeUnit.HOURS)
    }
}

private fun round2(v: Double): Double = round(v * 100) / 100


/**
 * Convert bytes into human readable sizes
 */
fun formatFileByteSize(sizeInBytes: Long) = when {
    sizeInBytes >= 1_000_000_000 -> {
        val sizeInGb = sizeInBytes / 1_000_000_000.0
        String.format(Locale.getDefault(), "%.1f GB", sizeInGb)
    }
    sizeInBytes >= 1_000_000 -> {
        val sizeInMb = sizeInBytes / 1_000_000.0
        String.format(Locale.getDefault(), "%.0f MB", sizeInMb)
    }
    else -> {
        val sizeInKb = sizeInBytes / 1_000.0
        String.format(Locale.getDefault(), "%.0f KB", sizeInKb)
    }
}

/**
 * Formats numbers to human-readable form (K, M)
 */
fun formatContextLength(contextLength: Int): String {
    return when {
        contextLength >= 1_000_000 -> String.format(Locale.getDefault(), "%.1fM", contextLength / 1_000_000.0)
        contextLength >= 1_000 -> String.format(Locale.getDefault(), "%.0fK", contextLength / 1_000.0)
        else -> contextLength.toString()
    }
}
