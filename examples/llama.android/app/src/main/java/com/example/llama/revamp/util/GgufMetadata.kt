package com.example.llama.revamp.util

import androidx.room.TypeConverter
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.IOException


/**
 * Structured metadata of GGUF
 */
@Serializable
data class GgufMetadata(
    // Basic file info
    val version: GgufVersion,
    val tensorCount: Long,
    val kvCount: Long,

    // General info
    val basic: BasicInfo,
    val author: AuthorInfo? = null,
    val additional: AdditionalInfo? = null,
    val architecture: ArchitectureInfo? = null,
    val baseModels: List<BaseModelInfo>? = null,
    val tokenizer: TokenizerInfo? = null,

    // Derivative info
    val dimensions: DimensionsInfo? = null,
    val attention: AttentionInfo? = null,
    val rope: RopeInfo? = null,
    val experts: ExpertsInfo? = null
) {
    /** Human-readable full model name + size */
    val fullModelName: String?
        get() = when {
            basic.nameLabel != null -> basic.nameLabel
            basic.name != null && basic.sizeLabel != null -> "${basic.name}-${basic.sizeLabel}"
            basic.name != null -> basic.name
            else -> null
        }

    /** Human‑readable model name (spaces). */
    val primaryName: String?
        get() = basic.nameLabel
            ?: baseModels?.firstNotNullOfOrNull { it.name }
            ?: basic.name

    /** CLI‑friendly slug (hyphens). */
    val primaryBasename: String?
        get() = basic.name
            ?: baseModels?.firstNotNullOfOrNull { it.name?.replace(' ', '-') }

    /** URL pointing to model homepage/repo. */
    val primaryUrl: String?
        get() = author?.url
            ?: baseModels?.firstNotNullOfOrNull { it.url }

    val primaryRepoUrl: String?
        get() = author?.repoUrl
            ?: baseModels?.firstNotNullOfOrNull { it.repoUrl }

    /** Organisation string. */
    val primaryOrganization: String?
        get() = author?.organization
            ?: baseModels?.firstNotNullOfOrNull { it.organization }

    /** Author string. */
    val primaryAuthor: String?
        get() = author?.author
            ?: baseModels?.firstNotNullOfOrNull { it.author }

    /** Context length with unit, e.g. “32768 tokens”. */
    val formattedContextLength: String?
        get() = dimensions?.contextLength?.let { "$it tokens" }

    @Serializable
    enum class GgufVersion(val code: Int, val label: String) {
        /** First public draft; little‑endian only, no alignment key. */
        LEGACY_V1(1, "Legacy v1"),

        /** Added split‑file support and some extra metadata keys. */
        EXTENDED_V2(2, "Extended v2"),

        /** Current spec: endian‑aware, mandatory alignment, fully validated. */
        VALIDATED_V3(3, "Validated v3");

        companion object {
            fun fromCode(code: Int): GgufVersion =
                entries.firstOrNull { it.code == code }
                    ?: throw IOException("Unknown GGUF version code $code")
        }

        override fun toString(): String = "$label (code=$code)"
    }

    @Serializable
    data class BasicInfo(
        val uuid: String? = null,
        val name: String? = null,
        val nameLabel: String? = null,
        val sizeLabel: String? = null,  // Size label like "7B"
    )

    @Serializable
    data class AuthorInfo(
        val organization: String? = null,
        val author: String? = null,
        val doi: String? = null,
        val url: String? = null,
        val repoUrl: String? = null,
        val license: String? = null,
        val licenseLink: String? = null,
    )

    @Serializable
    data class AdditionalInfo(
        val type: String? = null,
        val description: String? = null,
        val tags: List<String>? = null,
        val languages: List<String>? = null,
    )

    @Serializable
    data class ArchitectureInfo(
        val architecture: String? = null,
        val fileType: Int? = null,
        val vocabSize: Int? = null,
        val finetune: String? = null,
        val quantizationVersion: Int? = null,
    )

    @Serializable
    data class BaseModelInfo(
        val name: String? = null,
        val author: String? = null,
        val version: String? = null,
        val organization: String? = null,
        val url: String? = null,
        val doi: String? = null,
        val uuid: String? = null,
        val repoUrl: String? = null,
    )

    @Serializable
    data class TokenizerInfo(
        val model: String? = null,
        val bosTokenId: Int? = null,
        val eosTokenId: Int? = null,
        val unknownTokenId: Int? = null,
        val paddingTokenId: Int? = null,
        val addBosToken: Boolean? = null,
        val addEosToken: Boolean? = null,
        val chatTemplate: String? = null,
    )

    @Serializable
    data class DimensionsInfo(
        val contextLength: Int? = null,
        val embeddingSize: Int? = null,
        val blockCount: Int? = null,
        val feedForwardSize: Int? = null,
    )

    @Serializable
    data class AttentionInfo(
        val headCount: Int? = null,
        val headCountKv: Int? = null,
        val keyLength: Int? = null,
        val valueLength: Int? = null,
        val layerNormEpsilon: Float? = null,
        val layerNormRmsEpsilon: Float? = null,
    )

    @Serializable
    data class RopeInfo(
        val frequencyBase: Float? = null,
        val dimensionCount: Int? = null,
        val scalingType: String? = null,
        val scalingFactor: Float? = null,
        val attnFactor: Float? = null,
        val originalContextLength: Int? = null,
        val finetuned: Boolean? = null,
    )

    @Serializable
    data class ExpertsInfo(
        val count: Int? = null,
        val usedCount: Int? = null,
    )
}

class GgufMetadataConverters {
    private val json = Json { encodeDefaults = false; ignoreUnknownKeys = true }

    @TypeConverter
    fun toJson(value: GgufMetadata?): String? =
        value?.let { json.encodeToString(GgufMetadata.serializer(), it) }

    @TypeConverter
    fun fromJson(value: String?): GgufMetadata? =
        value?.let { json.decodeFromString(GgufMetadata.serializer(), it) }
}

/**
 * Numerical codes used by `general.file_type` (see llama.cpp repo's `constants.py`).
 * The `label` matches what llama‑cli prints.
 */
enum class FileType(val code: Int, val label: String) {
    ALL_F32(0, "all F32"),
    MOSTLY_F16(1, "F16"),
    MOSTLY_Q4_0(2, "Q4_0"),
    MOSTLY_Q4_1(3, "Q4_1"),
    // 4 removed
    MOSTLY_Q8_0(7, "Q8_0"),
    MOSTLY_Q5_0(8, "Q5_0"),
    MOSTLY_Q5_1(9, "Q5_1"),

    /* K‑quants ------------------------------------------------------------ */
    MOSTLY_Q2_K      (10, "Q2_K - Medium"),
    MOSTLY_Q3_K_S    (11, "Q3_K - Small"),
    MOSTLY_Q3_K_M    (12, "Q3_K - Medium"),
    MOSTLY_Q3_K_L    (13, "Q3_K - Large"),
    MOSTLY_Q4_K_S    (14, "Q4_K - Small"),
    MOSTLY_Q4_K_M    (15, "Q4_K - Medium"),
    MOSTLY_Q5_K_S    (16, "Q5_K - Small"),
    MOSTLY_Q5_K_M    (17, "Q5_K - Medium"),
    MOSTLY_Q6_K      (18, "Q6_K"),

    /* IQ quants ----------------------------------------------------------- */
    MOSTLY_IQ2_XXS   (19, "IQ2_XXS - 2.06 bpw"),
    MOSTLY_IQ2_XS    (20, "IQ2_XS - 2.31 bpw"),
    MOSTLY_Q2_K_S    (21, "Q2_K - Small"),
    MOSTLY_IQ3_XS    (22, "IQ3_XS - 3.30 bpw"),
    MOSTLY_IQ3_XXS   (23, "IQ3_XXS - 3.06 bpw"),
    MOSTLY_IQ1_S     (24, "IQ1_S - 1.56 bpw"),
    MOSTLY_IQ4_NL    (25, "IQ4_NL - 4.5 bpw"),
    MOSTLY_IQ3_S     (26, "IQ3_S - 3.44 bpw"),
    MOSTLY_IQ3_M     (27, "IQ3_M - 3.66 bpw"),
    MOSTLY_IQ2_S     (28, "IQ2_S - 2.50 bpw"),
    MOSTLY_IQ2_M     (29, "IQ2_M - 2.70 bpw"),
    MOSTLY_IQ4_XS    (30, "IQ4_XS - 4.25 bpw"),
    MOSTLY_IQ1_M     (31, "IQ1_M - 1.75 bpw"),

    /* BF16 & Ternary ------------------------------------------------------ */
    MOSTLY_BF16      (32, "BF16"),
    MOSTLY_TQ1_0     (36, "TQ1_0 - 1.69 bpw ternary"),
    MOSTLY_TQ2_0     (37, "TQ2_0 - 2.06 bpw ternary"),

    /* Special flag -------------------------------------------------------- */
    GUESSED(1024, "(guessed)"),

    UNKNOWN(-1, "unknown");

    companion object {
        private val map = entries.associateBy(FileType::code)

        fun fromCode(code: Int?): FileType = map[code] ?: UNKNOWN
    }
}
