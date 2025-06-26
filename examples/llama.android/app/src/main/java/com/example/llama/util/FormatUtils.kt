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


/**
 * Maps ISO 639-1 language codes into ISO 3166-1 alpha-2 country codes, and then map to Emoji.
 *
 */
fun languageCodeToFlagEmoji(languageCode: String): String? {
    val countryCode = LANGUAGE_TO_COUNTRY[languageCode.lowercase()] ?: return null

    return countryCodeToFlagEmoji(countryCode)
}

/**
 * Formats ISO 3166-1 alpha-2 country code into corresponding Emoji.
 */
private fun countryCodeToFlagEmoji(countryCode: String): String? {
    if (countryCode.length != 2) return null

    // Convert each character to a Regional Indicator Symbol
    val firstChar = Character.codePointAt(countryCode.uppercase(), 0) - 'A'.code + 0x1F1E6
    val secondChar = Character.codePointAt(countryCode.uppercase(), 1) - 'A'.code + 0x1F1E6

    return String(Character.toChars(firstChar)) + String(Character.toChars(secondChar))
}

private val LANGUAGE_TO_COUNTRY by lazy {
    mapOf(
        "en" to "US",
        "de" to "DE",
        "fr" to "FR",
        "it" to "IT",
        "es" to "ES",
        "pt" to "BR",
        "hi" to "IN",
        "th" to "TH",
        "ja" to "JP",
        "ko" to "KR",
        "zh" to "CN",
        "ru" to "RU",
        "ar" to "SA",
        "nl" to "NL",
        "sv" to "SE",
        "fi" to "FI",
        "pl" to "PL",
        "tr" to "TR",
        "vi" to "VN",
        "el" to "GR",
        "he" to "IL",
        "id" to "ID",
        "ms" to "MY",
        "no" to "NO",
        "da" to "DK",
        "cs" to "CZ",
        "hu" to "HU",
        "ro" to "RO",
        "sk" to "SK",
        "uk" to "UA",
    )
}
