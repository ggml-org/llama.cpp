package com.arm.aiplayground.viewmodel

import androidx.compose.foundation.text.input.TextFieldState
import androidx.compose.foundation.text.input.clearText
import androidx.compose.runtime.snapshotFlow
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.arm.aiplayground.data.model.ModelFilter
import com.arm.aiplayground.data.model.ModelInfo
import com.arm.aiplayground.data.model.ModelSortOrder
import com.arm.aiplayground.data.model.filterBy
import com.arm.aiplayground.data.model.queryBy
import com.arm.aiplayground.data.model.sortByOrder
import com.arm.aiplayground.data.repo.ModelRepository
import com.arm.aiplayground.engine.InferenceService
import com.arm.aiplayground.monitoring.PerformanceMonitor
import com.arm.aiplayground.viewmodel.PreselectedModelToRun.RamWarning
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
class ModelsViewModel @Inject constructor(
    private val modelRepository: ModelRepository,
    private val performanceMonitor: PerformanceMonitor,
    private val inferenceService: InferenceService,
) : ViewModel() {

    // UI state: model management mode
    private val _allModels = MutableStateFlow<List<ModelInfo>?>(null)
    val allModels = _allModels.asStateFlow()

    private val _modelScreenUiMode = MutableStateFlow(ModelScreenUiMode.BROWSING)
    val modelScreenUiMode = _modelScreenUiMode.asStateFlow()

    fun toggleMode(newMode: ModelScreenUiMode): Boolean {
        val oldMode = _modelScreenUiMode.value
        when (oldMode) {
            ModelScreenUiMode.BROWSING -> {
                when (newMode) {
                    ModelScreenUiMode.SEARCHING -> {
                        resetPreselection()
                    }
                    ModelScreenUiMode.MANAGING -> {
                        resetPreselection()
                    }
                    ModelScreenUiMode.DELETING -> { return false }
                    else -> { /* No-op */ }
                }
            }
            ModelScreenUiMode.SEARCHING -> {
                when (newMode) {
                    ModelScreenUiMode.BROWSING -> {
                        searchFieldState.clearText()
                    }
                    else -> { return false }
                }
            }
            ModelScreenUiMode.MANAGING -> {
                when (newMode) {
                    ModelScreenUiMode.SEARCHING -> { return false }
                    else -> { /* No-op */ }
                }
            }
            ModelScreenUiMode.DELETING -> {
                when (newMode) {
                    ModelScreenUiMode.BROWSING, ModelScreenUiMode.SEARCHING -> { return false }
                    else -> { /* No-op */ }
                }
            }
        }
        _modelScreenUiMode.value = newMode
        return true
    }

    // UI state: search mode
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
    private val _filteredModels = MutableStateFlow<List<ModelInfo>?>(null)
    val filteredModels = _filteredModels.asStateFlow()

    // Data: queried models
    private val _queryResults = MutableStateFlow<List<ModelInfo>?>(null)
    val queryResults = _queryResults.asStateFlow()

    // Data: pre-selected model in expansion mode
    private val _preselectedModelToRun = MutableStateFlow<ModelInfo?>(null)
    private val _showRamWarning = MutableStateFlow(false)
    val preselectedModelToRun = combine(
        _preselectedModelToRun,
        performanceMonitor.monitorMemoryUsage(),
        _showRamWarning,
    ) { model, memory, show ->
        if (model == null) {
            null
        } else {
            if (memory.availableMem >= model.sizeInBytes + RAM_LOAD_MODEL_BUFFER_BYTES) {
                PreselectedModelToRun(model, null)
            } else {
                PreselectedModelToRun(model, RamWarning(model.sizeInBytes, memory.availableMem, show))
            }
        }
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(SUBSCRIPTION_TIMEOUT_MS),
        initialValue = null
    )


    init {
        viewModelScope.launch {
            launch {
                modelRepository.getModels().collectLatest {
                    _allModels.value = it
                }
            }

            launch {
                combine(
                    _allModels,
                    _activeFilters,
                    _sortOrder,
                ) { models, filters, sortOrder ->
                    models?.filterBy(filters)?.sortByOrder(sortOrder)
                }.collectLatest {
                    _filteredModels.value = it
                }
            }

            launch {
                combine(
                    _allModels,
                    snapshotFlow { searchFieldState.text }.debounce(QUERY_DEBOUNCE_TIMEOUT_MS)
                ) { models, query ->
                    if (query.isBlank()) {
                        emptyList()
                    } else {
                        models?.queryBy(query.toString())?.sortedBy {
                            it.dateLastUsed ?: it.dateAdded
                        }
                    }
                }.collectLatest {
                    _queryResults.value = it
                }
            }
        }
    }

    /**
     * Pre-select a model to expand its details and show Run FAB
     */
    fun preselectModel(modelInfo: ModelInfo, preselected: Boolean) {
        _preselectedModelToRun.value = if (preselected) modelInfo else null
        _showRamWarning.value = false
    }

    /**
     * Reset preselected model to none (before navigating away)
     */
    fun resetPreselection() {
        _preselectedModelToRun.value = null
        _showRamWarning.value = false
    }

    /**
     * Select the currently pre-selected model
     *
     * @return True if RAM enough, otherwise False.
     */
    fun selectModel(preselectedModelToRun: PreselectedModelToRun) =
        when (preselectedModelToRun.ramWarning?.showing) {
            null -> {
                inferenceService.setCurrentModel(preselectedModelToRun.modelInfo)
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

            resetPreselection()

            true
        } else {
            false
        }


    companion object {
        private val TAG = ModelsViewModel::class.java.simpleName

        private const val SUBSCRIPTION_TIMEOUT_MS = 5000L
        private const val QUERY_DEBOUNCE_TIMEOUT_MS = 500L

        private const val RAM_LOAD_MODEL_BUFFER_BYTES = 300 * 1024
    }
}

enum class ModelScreenUiMode {
    BROWSING,
    SEARCHING,
    MANAGING,
    DELETING
}

data class PreselectedModelToRun(
    val modelInfo: ModelInfo,
    val ramWarning: RamWarning?,
) {
    data class RamWarning(
        val requiredRam: Long,
        val availableRam: Long,
        val showing: Boolean,
    )
}
