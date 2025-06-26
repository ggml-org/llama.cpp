package com.example.llama.data.model

import android.llama.cpp.gguf.FileType
import com.example.llama.util.formatContextLength
import com.example.llama.util.formatFileByteSize


/**
 * Data class containing information about an LLM model.
 *
 * This class represents a language model with its associated metadata, including
 * file information, architecture details, and usage statistics.
 *
 * @property id Unique identifier for the model
 * @property name Display name of the model
 * @property path File path to the model on device storage
 * @property sizeInBytes Size of the model file in bytes
 * @property metadata Structured metadata extracted from the GGUF file
 * @property dateAdded Timestamp when the model was added to the app
 * @property dateLastUsed Timestamp when the model was last used, or null if never used
 */
data class ModelInfo(
    val id: String,
    val name: String,
    val path: String,
    val sizeInBytes: Long,
    val metadata: GgufMetadata,
    val dateAdded: Long,
    val dateLastUsed: Long? = null,
) {
    /**
     * Full model name including version and parameter size if available, otherwise fallback to file name.
     */
    val formattedFullName: String
        get() = metadata.fullModelName ?: name

    /**
     * Human-readable file size with appropriate unit (KB, MB, GB).
     */
    val formattedFileSize: String
        get() = formatFileByteSize(sizeInBytes)

    /**
     * Architecture name of the model (e.g., "llama", "mistral"), or "-" if unavailable.
     */
    val formattedArchitecture: String
        get() = metadata.architecture?.architecture ?: "-"

    /**
     * Model parameter size with suffix (e.g., "7B", "13B"), or "-" if unavailable.
     */
    val formattedParamSize: String
        get() = metadata.basic.sizeLabel ?: "-"

    /**
     * Human-readable context length (e.g., "4K", "8K tokens"), or "-" if unavailable.
     */
    val formattedContextLength: String
        get() = metadata.dimensions?.contextLength?.let { formatContextLength(it) } ?: "-"

    /**
     * Quantization format of the model (e.g., "Q4_0", "Q5_K_M"), or "-" if unavailable.
     */
    val formattedQuantization: String
        get() = metadata.architecture?.fileType?.let { FileType.fromCode(it).label } ?: "-"

    /**
     * Tags associated with the model, or null if none are defined.
     */
    val tags: List<String>? = metadata.additional?.tags?.takeIf { it.isNotEmpty() }

    /**
     * Languages supported by the model, or null if none are defined.
     */
    val languages: List<String>? = metadata.additional?.languages?.takeIf { it.isNotEmpty() }
}

/**
 * Filters models by search query.
 *
 * Searches through model names, tags, languages, and architecture.
 * Returns the original list if the query is blank.
 *
 * @param query The search term to filter by
 * @return List of models matching the search criteria
 */
fun List<ModelInfo>.queryBy(query: String): List<ModelInfo> {
    if (query.isBlank()) return this

    return filter { model ->
        model.name.contains(query, ignoreCase = true) ||
            model.metadata.fullModelName?.contains(query, ignoreCase = true) == true ||
            model.metadata.additional?.tags?.any { it.contains(query, ignoreCase = true) } == true ||
            model.metadata.additional?.languages?.any { it.contains(query, ignoreCase = true) } == true ||
            model.metadata.architecture?.architecture?.contains(query, ignoreCase = true) == true
    }
}

/**
 * Sorting options for model lists.
 */
enum class ModelSortOrder {
    NAME_ASC,
    NAME_DESC,
    SIZE_ASC,
    SIZE_DESC,
    LAST_USED
}

/**
 * Sorts models according to the specified order.
 *
 * @param order The sort order to apply
 * @return Sorted list of models
 */
fun List<ModelInfo>.sortByOrder(order: ModelSortOrder): List<ModelInfo> {
    return when (order) {
        ModelSortOrder.NAME_ASC -> sortedBy { it.name }
        ModelSortOrder.NAME_DESC -> sortedByDescending { it.name }
        ModelSortOrder.SIZE_ASC -> sortedBy { it.sizeInBytes }
        ModelSortOrder.SIZE_DESC -> sortedByDescending { it.sizeInBytes }
        ModelSortOrder.LAST_USED -> sortedWith(
            compareByDescending<ModelInfo> { it.dateLastUsed }
                .thenBy { it.name }
        )
    }
}

/**
 * Filters for categorizing and filtering models.
 *
 * @property displayName Human-readable name shown in the UI
 * @property predicate Function that determines if a model matches this filter
 */
enum class ModelFilter(val displayName: String, val predicate: (ModelInfo) -> Boolean) {
    // Parameter size filters
    TINY_PARAMS("Tiny (<1B parameters)", {
        it.metadata.basic.sizeLabel?.let { size ->
            size.contains("M") || (size.contains("B") && size.replace("B", "").toFloatOrNull()?.let { n -> n < 1f } == true)
        } == true
    }),
    SMALL_PARAMS("Small (1-3B parameters)", {
        it.metadata.basic.sizeLabel?.let { size ->
            size.contains("B") && size.replace("B", "").toFloatOrNull()?.let { n -> n >= 1f && n <= 3f } == true
        } == true
    }),
    MEDIUM_PARAMS("Medium (4-7B parameters)", {
        it.metadata.basic.sizeLabel?.let { size ->
            size.contains("B") && size.replace("B", "").toFloatOrNull()?.let { n -> n >= 4f && n <= 7f } == true
        } == true
    }),
    LARGE_PARAMS("Large (8-13B parameters)", {
        it.metadata.basic.sizeLabel?.let { size ->
            size.contains("B") && size.replace("B", "").toFloatOrNull()?.let { n -> n >= 8f && n <= 13f } == true
        } == true
    }),
    XLARGE_PARAMS("X-Large (>13B parameters)", {
        it.metadata.basic.sizeLabel?.let { size ->
            size.contains("B") && size.replace("B", "").toFloatOrNull()?.let { n -> n > 13f } == true
        } == true
    }),

    // Context length filters
    TINY_CONTEXT("Tiny context (<4K)", {
        it.metadata.dimensions?.contextLength?.let { it < 4096 } == true
    }),
    SHORT_CONTEXT("Short context (4-8K)", {
        it.metadata.dimensions?.contextLength?.let { it >= 4096 && it <= 8192 } == true
    }),
    MEDIUM_CONTEXT("Medium context (8-32K)", {
        it.metadata.dimensions?.contextLength?.let { it > 8192 && it <= 32768 } == true
    }),
    LONG_CONTEXT("Long context (32-128K)", {
        it.metadata.dimensions?.contextLength?.let { it > 32768 && it <= 131072 } == true
    }),
    XLARGE_CONTEXT("Extended context (>128K)", {
        it.metadata.dimensions?.contextLength?.let { it > 131072 } == true
    }),

    // Quantization filters
    INT2_QUANT("2-bit quantization", {
        it.formattedQuantization.let { it.contains("Q2") || it.contains("IQ2") }
    }),
    INT3_QUANT("3-bit quantization", {
        it.formattedQuantization.let { it.contains("Q3") || it.contains("IQ3") }
    }),
    INT4_QUANT("4-bit quantization", {
        it.formattedQuantization.let { it.contains("Q4") || it.contains("IQ4") }
    }),

    // Special features
    MULTILINGUAL("Multilingual", {
        it.languages?.let { languages ->
            languages.size > 1 || languages.any { it.contains("multi", ignoreCase = true) }
        } == true
    }),
    HAS_TAGS("Has tags", {
        !it.tags.isNullOrEmpty()
    });

    companion object {
        // Group filters by category for UI
        private val PARAMETER_FILTERS = listOf(TINY_PARAMS, SMALL_PARAMS, MEDIUM_PARAMS, LARGE_PARAMS)
        private val CONTEXT_FILTERS = listOf(SHORT_CONTEXT, MEDIUM_CONTEXT, LONG_CONTEXT)
        private val QUANTIZATION_FILTERS = listOf(INT2_QUANT, INT3_QUANT, INT4_QUANT)
        private val FEATURE_FILTERS = listOf(MULTILINGUAL, HAS_TAGS)

        // All filters flattened
        val ALL_FILTERS = PARAMETER_FILTERS + CONTEXT_FILTERS + QUANTIZATION_FILTERS + FEATURE_FILTERS
    }
}

/**
 * Filters models based on a set of active filters.
 *
 * @param filters Map of filters to their enabled state
 * @return List of models that match all active filters
 */
fun List<ModelInfo>.filterBy(filters: Map<ModelFilter, Boolean>): List<ModelInfo> {
    val activeFilters = filters.filterValues { it }
    return if (activeFilters.isEmpty()) {
        this
    } else {
        filter { model ->
            activeFilters.keys.all { filter ->
                filter.predicate(model)
            }
        }
    }
}
