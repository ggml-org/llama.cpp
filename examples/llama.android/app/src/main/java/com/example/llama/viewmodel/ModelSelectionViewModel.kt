package com.example.llama.viewmodel

import androidx.compose.foundation.text.input.TextFieldState
import androidx.compose.foundation.text.input.clearText
import androidx.compose.runtime.snapshotFlow
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.data.model.ModelFilter
import com.example.llama.data.model.ModelInfo
import com.example.llama.data.model.ModelSortOrder
import com.example.llama.data.model.filterBy
import com.example.llama.data.model.queryBy
import com.example.llama.data.model.sortByOrder
import com.example.llama.data.repo.ModelRepository
import com.example.llama.engine.InferenceService
import com.example.llama.monitoring.PerformanceMonitor
import com.example.llama.viewmodel.Preselection.RamWarning
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.FlowPreview
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.debounce
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import javax.inject.Inject


@OptIn(FlowPreview::class)
@HiltViewModel
class ModelSelectionViewModel @Inject constructor(
    modelRepository: ModelRepository,
    private val performanceMonitor: PerformanceMonitor,
    private val inferenceService: InferenceService,
) : ViewModel() {

    // UI state: search mode
    private val _isSearchActive = MutableStateFlow(false)
    val isSearchActive = _isSearchActive.asStateFlow()

    fun toggleSearchState(active: Boolean) {
        _isSearchActive.value = active
        if (active) {
            resetPreselection()
        } else {
            searchFieldState.clearText()
        }
    }

    val searchFieldState = TextFieldState()

    // UI state: sort menu
    private val _sortOrder = MutableStateFlow(ModelSortOrder.LAST_USED)
    val sortOrder = _sortOrder.asStateFlow()

    fun setSortOrder(order: ModelSortOrder) {
        _sortOrder.value = order
    }

    private val _showSortMenu = MutableStateFlow(false)
    val showSortMenu = _showSortMenu.asStateFlow()

    fun toggleSortMenu(visible: Boolean) {
        _showSortMenu.value = visible
    }

    // UI state: filter menu
    private val _activeFilters = MutableStateFlow<Map<ModelFilter, Boolean>>(
        ModelFilter.ALL_FILTERS.associateWith { false }
    )
    val activeFilters: StateFlow<Map<ModelFilter, Boolean>> = _activeFilters.asStateFlow()

    fun toggleFilter(filter: ModelFilter, enabled: Boolean) {
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
    val showFilterMenu = _showFilterMenu.asStateFlow()

    fun toggleFilterMenu(visible: Boolean) {
        _showFilterMenu.value = visible
    }

    // Data: filtered & sorted models
    private val _filteredModels = MutableStateFlow<List<ModelInfo>>(emptyList())
    val filteredModels = _filteredModels.asStateFlow()

    // Data: queried models
    private val _queryResults = MutableStateFlow<List<ModelInfo>>(emptyList())
    val queryResults = _queryResults.asStateFlow()

    // Data: pre-selected model in expansion mode
    private val _preselectedModel = MutableStateFlow<ModelInfo?>(null)
    private val _showRamWarning = MutableStateFlow(false)
    val preselection = combine(
        _preselectedModel,
        performanceMonitor.monitorMemoryUsage(),
        _showRamWarning,
    ) { model, memory, show ->
        if (model == null) {
            null
        } else {
            if (memory.availableMem >= model.sizeInBytes + RAM_LOAD_MODEL_BUFFER_BYTES) {
                Preselection(model, null)
            } else {
                Preselection(model, RamWarning(model.sizeInBytes, memory.availableMem, show))
            }
        }
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(SUBSCRIPTION_TIMEOUT_MS),
        initialValue = null
    )

    init {
        viewModelScope.launch {
            combine(
                modelRepository.getModels(),
                _activeFilters,
                _sortOrder,
            ) { models, filters, sortOrder ->
                models.filterBy(filters).sortByOrder(sortOrder)
            }.collectLatest {
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

    /**
     * Pre-select a model to expand its details and show Run FAB
     */
    fun preselectModel(modelInfo: ModelInfo, preselected: Boolean) {
        _preselectedModel.value = if (preselected) modelInfo else null
        _showRamWarning.value = false
    }

    /**
     * Reset preselected model to none (before navigating away)
     */
    fun resetPreselection() {
        _preselectedModel.value = null
        _showRamWarning.value = false
    }

    /**
     * Select the currently pre-selected model
     *
     * @return True if RAM enough, otherwise False.
     */
    fun selectModel(preselection: Preselection) =
        when (preselection.ramWarning?.showing) {
            null -> {
                inferenceService.setCurrentModel(preselection.modelInfo)
                true
            }
            false -> {
                _showRamWarning.value = true
                false
            }
            else -> false
        }

    /**
     * Dismiss the RAM warnings
     */
    fun dismissRamWarning() {
        _showRamWarning.value = false
    }

    /**
     * Acknowledge RAM warnings and confirm currently pre-selected model
     *
     * @return True if confirmed, otherwise False.
     */
    fun confirmSelectedModel(modelInfo: ModelInfo, ramWarning: RamWarning): Boolean =
        if (ramWarning.showing) {
            inferenceService.setCurrentModel(modelInfo)
            _showRamWarning.value = false
            true
        } else {
            false
        }

    /**
     * Handle back press from both back button and top bar
     */
    fun onBackPressed() {
        if (_preselectedModel.value != null) {
            resetPreselection()
        }
    }

    companion object {
        private val TAG = ModelSelectionViewModel::class.java.simpleName

        private const val SUBSCRIPTION_TIMEOUT_MS = 5000L
        private const val QUERY_DEBOUNCE_TIMEOUT_MS = 500L

        private const val RAM_LOAD_MODEL_BUFFER_BYTES = 300 * 1024
    }
}

data class Preselection(
    val modelInfo: ModelInfo,
    val ramWarning: RamWarning?,
) {
    data class RamWarning(
        val requiredRam: Long,
        val availableRam: Long,
        val showing: Boolean,
    )
}
