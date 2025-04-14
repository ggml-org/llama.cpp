package com.example.llama.revamp.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.data.repository.ModelRepository
import com.example.llama.revamp.data.repository.StorageMetrics
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class ModelsManagementViewModel @Inject constructor(
    private val modelRepository: ModelRepository
) : ViewModel() {

    // Sort order state
    private val _sortOrder = MutableStateFlow(ModelSortOrder.NAME_ASC)
    val sortOrder: StateFlow<ModelSortOrder> = _sortOrder.asStateFlow()

    // Available models
    private val _availableModels = MutableStateFlow<List<ModelInfo>>(emptyList())
    val availableModels: StateFlow<List<ModelInfo>> = _availableModels.asStateFlow()

    // Storage metrics
    private val _storageMetrics = MutableStateFlow(StorageMetrics(0f, 0f))
    val storageMetrics: StateFlow<StorageMetrics> = _storageMetrics.asStateFlow()

    init {
        // Initial data load
        viewModelScope.launch {
            loadModels()
            loadStorageMetrics()
        }

        // Observe sort order changes and apply sorting
        viewModelScope.launch {
            sortOrder.collect { order -> sortModels(order) }
        }
    }

    private fun loadModels() {
        // TODO-han.yin: Stub for now. Would load from the repository
        _availableModels.value = ModelInfo.getSampleModels()
        sortModels(_sortOrder.value)
    }

    private fun loadStorageMetrics() {
        // TODO-han.yin: Stub for now. Would load from storage
        _storageMetrics.value = StorageMetrics(14.6f, 32.0f)
    }

    private fun sortModels(order: ModelSortOrder) {
        val sorted = when (order) {
            ModelSortOrder.NAME_ASC -> _availableModels.value.sortedBy { it.name }
            ModelSortOrder.NAME_DESC -> _availableModels.value.sortedByDescending { it.name }
            ModelSortOrder.SIZE_ASC -> _availableModels.value.sortedBy { it.sizeInBytes }
            ModelSortOrder.SIZE_DESC -> _availableModels.value.sortedByDescending { it.sizeInBytes }
            ModelSortOrder.LAST_USED -> _availableModels.value.sortedByDescending { it.lastUsed ?: 0 }
        }
        _availableModels.value = sorted
    }

    fun setSortOrder(order: ModelSortOrder) {
        _sortOrder.value = order
    }

    fun viewModelDetails(modelId: String) {
        // TODO-han.yin: Stub for now. Would navigate to model details screen or show dialog
    }

    fun deleteModel(modelId: String) {
        // Remove model from list
        _availableModels.value = _availableModels.value.filter { it.id != modelId }

        viewModelScope.launch {
            // TODO-han.yin: Stub for now this would delete from storage
            modelRepository.deleteModel(modelId)
            updateStorageMetrics()
        }
    }

    fun deleteModels(models: Map<String, ModelInfo>) {
        val modelIds = models.keys
        _availableModels.value = _availableModels.value.filter { !modelIds.contains(it.id) }

        viewModelScope.launch {
            modelRepository.deleteModels(modelIds)
            updateStorageMetrics()
        }
    }

    fun importLocalModel() {
        // TODO-han.yin: Stub for now. Would open file picker and import model
    }

    private fun updateStorageMetrics() {
        // Recalculate storage metrics after model changes
        // TODO-han.yin: Stub for now. Would query actual storage
        val totalSize = _availableModels.value.sumOf { it.sizeInBytes }
        _storageMetrics.value = StorageMetrics(
            (totalSize / 1_000_000_000.0).toFloat(),
            32.0f
        )
    }
}

enum class ModelSortOrder {
    NAME_ASC,
    NAME_DESC,
    SIZE_ASC,
    SIZE_DESC,
    LAST_USED
}


