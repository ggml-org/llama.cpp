package com.example.llama.revamp.viewmodel

import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.data.model.ModelSortOrder
import com.example.llama.revamp.data.repository.InsufficientStorageException
import com.example.llama.revamp.data.repository.ModelRepository
import com.example.llama.revamp.util.getFileNameFromUri
import com.example.llama.revamp.util.getFileSizeFromUri
import com.example.llama.revamp.viewmodel.ModelManagementState.Deletion
import com.example.llama.revamp.viewmodel.ModelManagementState.Importation
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.FileNotFoundException
import javax.inject.Inject

@HiltViewModel
class ModelsManagementViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val modelRepository: ModelRepository
) : ViewModel() {

    // Data: available models
    private val _availableModels: StateFlow<List<ModelInfo>> = modelRepository.getModels()
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(SUBSCRIPTION_TIMEOUT_MS),
            initialValue = emptyList()
        )

    private val _sortedModels = MutableStateFlow<List<ModelInfo>>(emptyList())
    val sortedModels: StateFlow<List<ModelInfo>> = _sortedModels.asStateFlow()

    // UI state: multi-selection mode
    private val _isMultiSelectionMode = MutableStateFlow(false)
    val isMultiSelectionMode: StateFlow<Boolean> = _isMultiSelectionMode.asStateFlow()

    fun toggleSelectionMode(enabled: Boolean) {
        _isMultiSelectionMode.value = enabled
        if (!enabled) {
            toggleAllSelection(selectAll = false)
        }
    }

    // UI state: models selected in multi-selection
    private val _selectedModels = MutableStateFlow<Map<String, ModelInfo>>(emptyMap())
    val selectedModels: StateFlow<Map<String, ModelInfo>> = _selectedModels.asStateFlow()

    fun toggleModelSelectionById(modelId: String) {
        val current = _selectedModels.value.toMutableMap()
        val model = _sortedModels.value.find { it.id == modelId }

        if (model != null) {
            if (current.containsKey(modelId)) {
                current.remove(modelId)
            } else {
                current[modelId] = model
            }
            _selectedModels.value = current
        }
    }

    fun toggleAllSelection(selectAll: Boolean) {
        if (selectAll) {
            _selectedModels.value = _sortedModels.value.associateBy { it.id }
        } else {
            _selectedModels.value = emptyMap()
        }
    }

    // UI state: sort menu
    private val _sortOrder = MutableStateFlow(ModelSortOrder.NAME_ASC)
    val sortOrder: StateFlow<ModelSortOrder> = _sortOrder.asStateFlow()

    fun setSortOrder(order: ModelSortOrder) {
        _sortOrder.value = order
    }

    private val _showSortMenu = MutableStateFlow(false)
    val showSortMenu: StateFlow<Boolean> = _showSortMenu.asStateFlow()

    fun toggleSortMenu(show: Boolean) {
        _showSortMenu.value = show
    }

    // UI state: import menu
    private val _showImportModelMenu = MutableStateFlow(false)
    val showImportModelMenu: StateFlow<Boolean> = _showImportModelMenu.asStateFlow()

    fun toggleImportMenu(show: Boolean) {
        _showImportModelMenu.value = show
    }

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
            ModelSortOrder.LAST_USED -> models.sortedByDescending { it.dateLastUsed ?: 0 }
        }

    // Internal state
    private val _managementState = MutableStateFlow<ModelManagementState>(ModelManagementState.Idle)
    val managementState: StateFlow<ModelManagementState> = _managementState.asStateFlow()

    fun resetManagementState() {
        _managementState.value = ModelManagementState.Idle
    }

    fun viewModelDetails(model: ModelInfo) {
        // TODO-han.yin: print logs for now. Would navigate to model details screen or show dialog
        Log.i(TAG, "${model.name} metadata: \n" + model.metadata.toString())
    }

    /**
     * First show confirmation instead of starting import immediately
     */
    fun localModelFileSelected(uri: Uri) = viewModelScope.launch {
        try {
            val fileName = getFileNameFromUri(context, uri) ?: throw FileNotFoundException("File size N/A")
            val fileSize = getFileSizeFromUri(context, uri) ?: throw FileNotFoundException("File name N/A")
            _managementState.value = Importation.Confirming(uri, fileName, fileSize)
        } catch (e: Exception) {
            _managementState.value = Importation.Error(
                message = e.message ?: "Unknown error preparing import"
            )
        }
    }

    /**
     * Import a local model file from device storage while updating UI states with realtime progress
     */
    fun importLocalModelFile(uri: Uri, fileName: String, fileSize: Long) = viewModelScope.launch {
        try {
            _managementState.value = Importation.Importing(0f, fileName, fileSize)
            val model = modelRepository.importModel(uri, fileName, fileSize) { progress ->
                _managementState.value = Importation.Importing(progress, fileName, fileSize)
            }
            _managementState.value = Importation.Success(model)
        } catch (e: InsufficientStorageException) {
            _managementState.value = Importation.Error(
                message = e.message ?: "Insufficient storage space to import $uri",
            )
        } catch (e: Exception) {
            _managementState.value = Importation.Error(
                message = e.message ?: "Unknown error importing $uri",
            )
        }
    }

    fun cancelOngoingLocalModelImport() = viewModelScope.launch {
        viewModelScope.launch {
            // First update UI to show we're attempting to cancel
            _managementState.update { current ->
                if (current is Importation.Importing) {
                    current.copy(isCancelling = true)
                } else {
                    current
                }
            }

            // Attempt to cancel
            when (modelRepository.cancelImport()) {
                null, true -> { _managementState.value = ModelManagementState.Idle }
                false -> {
                    _managementState.value = Importation.Error(
                        message = "Failed to cancel import. Try again later."
                    )
                }
            }
        }
    }

    // TODO-han.yin: Stub for now. Would need to investigate HuggingFace APIs
    fun importFromHuggingFace() {}

    /**
     * First show confirmation instead of starting deletion immediately
     */
    fun batchDeletionClicked(models: Map<String, ModelInfo>) {
        _managementState.value = Deletion.Confirming(models)
    }

    /**
     * Delete multiple models one by one while updating UI states with realtime progress
     */
    fun deleteModels(models: Map<String, ModelInfo>) = viewModelScope.launch {
        val total = models.size
        if (total == 0) return@launch

        try {
            _managementState.value = Deletion.Deleting(0f, models)
            var deleted = 0
            models.keys.toList().forEach {
                modelRepository.deleteModel(it)
                deleted++
                _managementState.value = Deletion.Deleting(deleted.toFloat() / total, models)
            }
            _managementState.value = Deletion.Success(models.values.toList())

            // Reset state after a delay
            delay(SUCCESS_RESET_TIMEOUT_MS)
            _managementState.value = ModelManagementState.Idle
        } catch (e: Exception) {
            _managementState.value = Deletion.Error(
                message = e.message ?: "Error deleting $total models"
            )
        }
    }

    companion object {
        private val TAG = ModelsManagementViewModel::class.java.simpleName

        private const val SUBSCRIPTION_TIMEOUT_MS = 5000L
        private const val SUCCESS_RESET_TIMEOUT_MS = 1000L
    }
}

sealed class ModelManagementState {
    object Idle : ModelManagementState()

    sealed class Importation : ModelManagementState() {
        data class Confirming(val uri: Uri, val fileName: String, val fileSize: Long) : Importation()
        data class Importing(val progress: Float = 0f, val fileName: String, val fileSize: Long, val isCancelling: Boolean = false) : Importation()
        data class Success(val model: ModelInfo) : Importation()
        data class Error(val message: String) : Importation()
    }

    sealed class Deletion : ModelManagementState() {
        data class Confirming(val models: Map<String, ModelInfo>): ModelManagementState()
        data class Deleting(val progress: Float = 0f, val models: Map<String, ModelInfo>) : ModelManagementState()
        data class Success(val models: List<ModelInfo>) : Deletion()
        data class Error(val message: String) : Deletion()
    }
}
