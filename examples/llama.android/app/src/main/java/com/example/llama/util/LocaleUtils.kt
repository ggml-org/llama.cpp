package com.example.llama.util


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
        "af" to "ZA",
        "am" to "ET",
        "ar" to "SA",
        "bg" to "BG",
        "bn" to "BD",
        "cs" to "CZ",
        "da" to "DK",
        "de" to "DE",
        "el" to "GR",
        "en" to "US",
        "es" to "ES",
        "et" to "EE",
        "fa" to "IR",
        "fi" to "FI",
        "fil" to "PH",
        "fr" to "FR",
        "he" to "IL",
        "hi" to "IN",
        "hr" to "HR",
        "hu" to "HU",
        "id" to "ID",
        "it" to "IT",
        "ja" to "JP",
        "kn" to "IN",
        "ko" to "KR",
        "lt" to "LT",
        "lv" to "LV",
        "ml" to "IN",
        "mr" to "IN",
        "ms" to "MY",
        "nl" to "NL",
        "no" to "NO",
        "pa" to "IN",
        "pl" to "PL",
        "pt" to "PT",
        "ro" to "RO",
        "ru" to "RU",
        "sk" to "SK",
        "sl" to "SI",
        "sr" to "RS",
        "sv" to "SE",
        "sw" to "KE",
        "ta" to "LK",
        "te" to "IN",
        "th" to "TH",
        "tr" to "TR",
        "uk" to "UA",
        "ur" to "PK",
        "vi" to "VN",
        "zh" to "CN",
        "zu" to "ZA",
    )
}
