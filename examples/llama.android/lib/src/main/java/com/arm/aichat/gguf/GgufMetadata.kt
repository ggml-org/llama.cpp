package com.arm.aichat.gguf

import java.io.IOException


/**
 * Structured metadata of GGUF
 */
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

    data class BasicInfo(
        val uuid: String? = null,
        val name: String? = null,
        val nameLabel: String? = null,
        val sizeLabel: String? = null,  // Size label like "7B"
    )

    data class AuthorInfo(
        val organization: String? = null,
        val author: String? = null,
        val doi: String? = null,
        val url: String? = null,
        val repoUrl: String? = null,
        val license: String? = null,
        val licenseLink: String? = null,
    )

    data class AdditionalInfo(
        val type: String? = null,
        val description: String? = null,
        val tags: List<String>? = null,
        val languages: List<String>? = null,
    )

    data class ArchitectureInfo(
        val architecture: String? = null,
        val fileType: Int? = null,
        val vocabSize: Int? = null,
        val finetune: String? = null,
        val quantizationVersion: Int? = null,
    )

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

    data class DimensionsInfo(
        val contextLength: Int? = null,
        val embeddingSize: Int? = null,
        val blockCount: Int? = null,
        val feedForwardSize: Int? = null,
    )

    data class AttentionInfo(
        val headCount: Int? = null,
        val headCountKv: Int? = null,
        val keyLength: Int? = null,
        val valueLength: Int? = null,
        val layerNormEpsilon: Float? = null,
        val layerNormRmsEpsilon: Float? = null,
    )

    data class RopeInfo(
        val frequencyBase: Float? = null,
        val dimensionCount: Int? = null,
        val scalingType: String? = null,
        val scalingFactor: Float? = null,
        val attnFactor: Float? = null,
        val originalContextLength: Int? = null,
        val finetuned: Boolean? = null,
    )

    data class ExpertsInfo(
        val count: Int? = null,
        val usedCount: Int? = null,
    )
}
