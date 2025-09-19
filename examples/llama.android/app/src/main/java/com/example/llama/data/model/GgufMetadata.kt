package com.example.llama.data.model

import kotlinx.serialization.Serializable
import com.arm.aichat.gguf.GgufMetadata as Domain


/**
 * A local serializable domain replicate of [com.arm.aichat.gguf.GgufMetadata]
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

    @Serializable
    enum class GgufVersion(val code: Int, val label: String) {
        /** First public draft; little‑endian only, no alignment key. */
        LEGACY_V1(1, "Legacy v1"),

        /** Added split‑file support and some extra metadata keys. */
        EXTENDED_V2(2, "Extended v2"),

        /** Current spec: endian‑aware, mandatory alignment, fully validated. */
        VALIDATED_V3(3, "Validated v3");

        companion object {
            fun fromDomain(domain: Domain.GgufVersion): GgufVersion = when (domain) {
                Domain.GgufVersion.LEGACY_V1 -> LEGACY_V1
                Domain.GgufVersion.EXTENDED_V2 -> EXTENDED_V2
                Domain.GgufVersion.VALIDATED_V3 -> VALIDATED_V3
            }
        }

        override fun toString(): String = "$label (code=$code)"
    }

    @Serializable
    data class BasicInfo(
        val uuid: String? = null,
        val name: String? = null,
        val nameLabel: String? = null,
        val sizeLabel: String? = null,  // Size label like "7B"
    ) {
        companion object {
            fun fromDomain(domain: Domain.BasicInfo) = BasicInfo(
                uuid = domain.uuid,
                name = domain.name,
                nameLabel = domain.nameLabel,
                sizeLabel = domain.sizeLabel
            )
        }
    }

    @Serializable
    data class AuthorInfo(
        val organization: String? = null,
        val author: String? = null,
        val doi: String? = null,
        val url: String? = null,
        val repoUrl: String? = null,
        val license: String? = null,
        val licenseLink: String? = null,
    ) {
        companion object {
            fun fromDomain(domain: Domain.AuthorInfo) = AuthorInfo(
                organization = domain.organization,
                author = domain.author,
                doi = domain.doi,
                url = domain.url,
                repoUrl = domain.repoUrl,
                license = domain.license,
                licenseLink = domain.licenseLink
            )
        }
    }

    @Serializable
    data class AdditionalInfo(
        val type: String? = null,
        val description: String? = null,
        val tags: List<String>? = null,
        val languages: List<String>? = null,
    ) {
        companion object {
            fun fromDomain(domain: Domain.AdditionalInfo) = AdditionalInfo(
                type = domain.type,
                description = domain.description,
                tags = domain.tags,
                languages = domain.languages
            )
        }
    }

    @Serializable
    data class ArchitectureInfo(
        val architecture: String? = null,
        val fileType: Int? = null,
        val vocabSize: Int? = null,
        val finetune: String? = null,
        val quantizationVersion: Int? = null,
    ) {
        companion object {
            fun fromDomain(domain: Domain.ArchitectureInfo) = ArchitectureInfo(
                architecture = domain.architecture,
                fileType = domain.fileType,
                vocabSize = domain.vocabSize,
                finetune = domain.finetune,
                quantizationVersion = domain.quantizationVersion
            )
        }
    }

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
    ) {
        companion object {
            fun fromDomain(domain: Domain.BaseModelInfo) = BaseModelInfo(
                name = domain.name,
                author = domain.author,
                version = domain.version,
                organization = domain.organization,
                url = domain.url,
                doi = domain.doi,
                uuid = domain.uuid,
                repoUrl = domain.repoUrl
            )
        }
    }

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
    ) {
        companion object {
            fun fromDomain(domain: Domain.TokenizerInfo) = TokenizerInfo(
                model = domain.model,
                bosTokenId = domain.bosTokenId,
                eosTokenId = domain.eosTokenId,
                unknownTokenId = domain.unknownTokenId,
                paddingTokenId = domain.paddingTokenId,
                addBosToken = domain.addBosToken,
                addEosToken = domain.addEosToken,
                chatTemplate = domain.chatTemplate
            )
        }
    }

    @Serializable
    data class DimensionsInfo(
        val contextLength: Int? = null,
        val embeddingSize: Int? = null,
        val blockCount: Int? = null,
        val feedForwardSize: Int? = null,
    ) {
        companion object {
            fun fromDomain(domain: Domain.DimensionsInfo) = DimensionsInfo(
                contextLength = domain.contextLength,
                embeddingSize = domain.embeddingSize,
                blockCount = domain.blockCount,
                feedForwardSize = domain.feedForwardSize
            )
        }
    }

    @Serializable
    data class AttentionInfo(
        val headCount: Int? = null,
        val headCountKv: Int? = null,
        val keyLength: Int? = null,
        val valueLength: Int? = null,
        val layerNormEpsilon: Float? = null,
        val layerNormRmsEpsilon: Float? = null,
    ) {
        companion object {
            fun fromDomain(domain: Domain.AttentionInfo) = AttentionInfo(
                headCount = domain.headCount,
                headCountKv = domain.headCountKv,
                keyLength = domain.keyLength,
                valueLength = domain.valueLength,
                layerNormEpsilon = domain.layerNormEpsilon,
                layerNormRmsEpsilon = domain.layerNormRmsEpsilon
            )
        }
    }

    @Serializable
    data class RopeInfo(
        val frequencyBase: Float? = null,
        val dimensionCount: Int? = null,
        val scalingType: String? = null,
        val scalingFactor: Float? = null,
        val attnFactor: Float? = null,
        val originalContextLength: Int? = null,
        val finetuned: Boolean? = null,
    ) {
        companion object {
            fun fromDomain(domain: Domain.RopeInfo) = RopeInfo(
                frequencyBase = domain.frequencyBase,
                dimensionCount = domain.dimensionCount,
                scalingType = domain.scalingType,
                scalingFactor = domain.scalingFactor,
                attnFactor = domain.attnFactor,
                originalContextLength = domain.originalContextLength,
                finetuned = domain.finetuned
            )
        }
    }

    @Serializable
    data class ExpertsInfo(
        val count: Int? = null,
        val usedCount: Int? = null,
    ) {
        companion object {
            fun fromDomain(domain: Domain.ExpertsInfo) = ExpertsInfo(
                count = domain.count,
                usedCount = domain.usedCount
            )
        }
    }

    companion object {
        fun fromDomain(domain: Domain) = GgufMetadata(
            version = GgufVersion.fromDomain(domain.version),
            tensorCount = domain.tensorCount,
            kvCount = domain.kvCount,
            basic = BasicInfo.fromDomain(domain.basic),
            author = domain.author?.let { AuthorInfo.fromDomain(it) },
            additional = domain.additional?.let { AdditionalInfo.fromDomain(it) },
            architecture = domain.architecture?.let { ArchitectureInfo.fromDomain(it) },
            baseModels = domain.baseModels?.map { BaseModelInfo.fromDomain(it) },
            tokenizer = domain.tokenizer?.let { TokenizerInfo.fromDomain(it) },
            dimensions = domain.dimensions?.let { DimensionsInfo.fromDomain(it) },
            attention = domain.attention?.let { AttentionInfo.fromDomain(it) },
            rope = domain.rope?.let { RopeInfo.fromDomain(it) },
            experts = domain.experts?.let { ExpertsInfo.fromDomain(it) }
        )
    }
}
