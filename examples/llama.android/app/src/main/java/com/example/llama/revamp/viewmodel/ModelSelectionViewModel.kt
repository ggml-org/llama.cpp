package com.example.llama.revamp.viewmodel

import androidx.compose.foundation.text.input.TextFieldState
import androidx.compose.foundation.text.input.clearText
import androidx.compose.runtime.snapshotFlow
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.data.repository.ModelRepository
import com.example.llama.revamp.engine.InferenceService
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.FlowPreview
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.debounce
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject


@OptIn(FlowPreview::class)
@HiltViewModel
class ModelSelectionViewModel @Inject constructor(
    private val inferenceService: InferenceService,
    modelRepository: ModelRepository
) : ViewModel() {

    // UI state: search mode
    private val _isSearchActive = MutableStateFlow(false)
    val isSearchActive: StateFlow<Boolean> = _isSearchActive.asStateFlow()

    fun toggleSearchState(active: Boolean) {
        _isSearchActive.value = active
        if (active) {
            resetSelection()
        } else {
            searchFieldState.clearText()
        }
    }

    val searchFieldState = TextFieldState()

    // UI state: sort menu
    private val _sortOrder = MutableStateFlow(ModelSortOrder.LAST_USED)
    val sortOrder: StateFlow<ModelSortOrder> = _sortOrder.asStateFlow()

    fun setSortOrder(order: ModelSortOrder) {
        _sortOrder.value = order
    }

    private val _showSortMenu = MutableStateFlow(false)
    val showSortMenu: StateFlow<Boolean> = _showSortMenu.asStateFlow()

    fun toggleSortMenu(visible: Boolean) {
        _showSortMenu.value = visible
    }

    // UI state: filters
    // TODO-han.yin: Refactor this into Enums!
    private val _activeFilters = MutableStateFlow<Map<String, Boolean>>(mapOf(
        "Has context length" to false,
        "Support system prompt" to false,
        "7B models" to false,
        "13B models" to false,
        "70B models" to false
    ))
    val activeFilters: StateFlow<Map<String, Boolean>> = _activeFilters.asStateFlow()

    fun toggleFilter(filter: String, enabled: Boolean) {
        _activeFilters.update { current ->
            current.toMutableMap().apply {
                this[filter] = enabled
            }
        }
    }

    fun clearFilters() {
        _activeFilters.update { current ->
            current.mapValues { false }
        }
    }

    private val _showFilterMenu = MutableStateFlow(false)
    val showFilterMenu: StateFlow<Boolean> = _showFilterMenu.asStateFlow()

    fun toggleFilterMenu(visible: Boolean) {
        _showFilterMenu.value = visible
    }

    // Data: filtered & sorted models
    private val _filteredModels = MutableStateFlow<List<ModelInfo>>(emptyList())
    val filteredModels: StateFlow<List<ModelInfo>> = _filteredModels.asStateFlow()

    // Data: queried models
    private val _queryResults = MutableStateFlow<List<ModelInfo>>(emptyList())
    val queryResults: StateFlow<List<ModelInfo>> = _queryResults.asStateFlow()

    // Data: pre-selected model in expansion mode
    private val _preselectedModel = MutableStateFlow<ModelInfo?>(null)
    val preselectedModel: StateFlow<ModelInfo?> = _preselectedModel.asStateFlow()

    init {
        viewModelScope.launch {
            combine(
                modelRepository.getModels(),
                _activeFilters,
                _sortOrder,
            ) { models, filters, sortOrder ->
                models.filterBy(filters).sortByOrder(sortOrder)
            }.collect {
                _filteredModels.value = it
            }
        }

        viewModelScope.launch {
            combine(
                modelRepository.getModels(),
                snapshotFlow { searchFieldState.text }.debounce(QUERY_DEBOUNCE_TIMEOUT_MS)
            ) { models, query ->
                if (query.isBlank()) {
                    emptyList()
                } else {
                    models.queryBy(query.toString()).sortedBy { it.dateLastUsed ?: it.dateAdded }
                }
            }.collectLatest {
                _queryResults.value = it
            }
        }
    }

    private fun List<ModelInfo>.queryBy(query: String): List<ModelInfo> {
        if (query.isBlank()) return this

        return filter { model ->
            model.name.contains(query, ignoreCase = true) ||
                model.metadata.fullModelName?.contains(query, ignoreCase = true) == true ||
                model.metadata.additional?.tags?.any { it.contains(query, ignoreCase = true) } == true ||
                model.metadata.additional?.languages?.any { it.contains(query, ignoreCase = true) } == true ||
                model.metadata.architecture?.architecture?.contains(query, ignoreCase = true) == true
        }
    }

    // TODO-han.yin: Refactor this into Enums!
    private fun List<ModelInfo>.filterBy(filters: Map<String, Boolean>): List<ModelInfo> {
        val activeFilters = filters.filterValues { it }
        if (activeFilters.isEmpty()) return this

        return filter { model ->
            activeFilters.all { (filter, _) ->
                when (filter) {
                    "Has context length" -> model.metadata.dimensions?.contextLength != null
                    "Support system prompt" -> true
                    "7B models" -> model.metadata.basic.sizeLabel?.contains("7B") == true
                    "13B models" -> model.metadata.basic.sizeLabel?.contains("13B") == true
                    "70B models" -> model.metadata.basic.sizeLabel?.contains("70B") == true
                    else -> true
                }
            }
        }
    }

    private fun List<ModelInfo>.sortByOrder(order: ModelSortOrder): List<ModelInfo> {
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
     * Pre-select a model
     */
    fun preselectModel(modelInfo: ModelInfo, preselected: Boolean) =
        _preselectedModel.update { current ->
            if (preselected) modelInfo else null
        }

    /**
     * Confirm currently selected model
     */
    fun confirmSelectedModel(modelInfo: ModelInfo) =
        inferenceService.setCurrentModel(modelInfo)

    /**
     * Reset selected model to none (before navigating away)
     */
    fun resetSelection() {
        _preselectedModel.value = null
    }

    /**
     * Handle back press from both back button and top bar
     */
    fun onBackPressed() {
        if (_preselectedModel.value != null) {
            resetSelection()
        }
    }

    companion object {
        private val TAG = ModelSelectionViewModel::class.java.simpleName

        private const val QUERY_DEBOUNCE_TIMEOUT_MS = 500L
    }
}
