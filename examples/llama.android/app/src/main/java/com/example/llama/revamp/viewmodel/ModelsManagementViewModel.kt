package com.example.llama.revamp.viewmodel

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.data.repository.ModelRepository
import com.example.llama.revamp.data.repository.StorageMetrics
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class ModelsManagementViewModel @Inject constructor(
    private val modelRepository: ModelRepository
) : ViewModel() {

    val storageMetrics: StateFlow<StorageMetrics?> = modelRepository.getStorageMetrics()
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(SUBSCRIPTION_TIMEOUT_MS),
            initialValue = null
        )

    private val _availableModels: StateFlow<List<ModelInfo>> = modelRepository.getModels()
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(SUBSCRIPTION_TIMEOUT_MS),
            initialValue = emptyList()
        )

    private val _sortOrder = MutableStateFlow(ModelSortOrder.NAME_ASC)
    val sortOrder: StateFlow<ModelSortOrder> = _sortOrder.asStateFlow()

    private val _sortedModels = MutableStateFlow<List<ModelInfo>>(emptyList())
    val sortedModels: StateFlow<List<ModelInfo>> = _sortedModels.asStateFlow()

    init {
        viewModelScope.launch {
            combine(_availableModels, _sortOrder, ::sortModels)
                .collect { _sortedModels.value = it }
        }
    }

    private fun sortModels(models: List<ModelInfo>, order: ModelSortOrder) =
        when (order) {
            ModelSortOrder.NAME_ASC -> models.sortedBy { it.name }
            ModelSortOrder.NAME_DESC -> models.sortedByDescending { it.name }
            ModelSortOrder.SIZE_ASC -> models.sortedBy { it.sizeInBytes }
            ModelSortOrder.SIZE_DESC -> models.sortedByDescending { it.sizeInBytes }
            ModelSortOrder.LAST_USED -> models.sortedByDescending { it.lastUsed ?: 0 }
        }

    fun setSortOrder(order: ModelSortOrder) {
        _sortOrder.value = order
    }

    fun viewModelDetails(modelId: String) {
        // TODO-han.yin: Stub for now. Would navigate to model details screen or show dialog
    }

    fun deleteModel(modelId: String) =
        viewModelScope.launch {
            modelRepository.deleteModel(modelId)
        }

    fun deleteModels(models: Map<String, ModelInfo>) =
        viewModelScope.launch {
            modelRepository.deleteModels(models.keys)
        }

    private val _importState = MutableStateFlow<ModelImportState>(ModelImportState.Idle)
    val importState: StateFlow<ModelImportState> = _importState.asStateFlow()

    fun importLocalModel(uri: Uri) =
        viewModelScope.launch {
            try {
                // Get filename for progress updates
                val filename = uri.lastPathSegment ?: throw Exception("Model name unknown")
                _importState.value = ModelImportState.Importing(0f, filename)

                // Import with progress reporting
                val model = modelRepository.importModel(uri) { progress ->
                    _importState.value = ModelImportState.Importing(progress, filename)
                }
                _importState.value = ModelImportState.Success(model)

                // Reset state after a delay
                delay(1000)
                _importState.value = ModelImportState.Idle
            } catch (e: Exception) {
                _importState.value = ModelImportState.Error(e.message ?: "Unknown error")
            }
        }

    fun resetImportState() {
        _importState.value = ModelImportState.Idle
    }

    fun importFromHuggingFace() {
        // TODO-han.yin: Stub for now. Would need to investigate HuggingFace APIs
    }

    companion object {
        private val TAG = ModelsManagementViewModel::class.java.simpleName

        private const val SUBSCRIPTION_TIMEOUT_MS = 5000L
    }
}

enum class ModelSortOrder {
    NAME_ASC,
    NAME_DESC,
    SIZE_ASC,
    SIZE_DESC,
    LAST_USED
}

sealed class ModelImportState {
    object Idle : ModelImportState()
    data class Importing(val progress: Float = 0f, val filename: String = "") : ModelImportState()
    data class Success(val model: ModelInfo) : ModelImportState()
    data class Error(val message: String) : ModelImportState()
}
