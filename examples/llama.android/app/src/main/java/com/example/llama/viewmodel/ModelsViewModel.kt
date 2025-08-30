package com.example.llama.viewmodel

import android.app.DownloadManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Context.RECEIVER_EXPORTED
import android.content.Intent
import android.content.IntentFilter
import android.llama.cpp.gguf.InvalidFileFormatException
import android.net.Uri
import android.util.Log
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
import com.example.llama.data.repo.InsufficientStorageException
import com.example.llama.data.repo.ModelRepository
import com.example.llama.data.source.remote.HuggingFaceDownloadInfo
import com.example.llama.data.source.remote.HuggingFaceModel
import com.example.llama.engine.InferenceService
import com.example.llama.monitoring.PerformanceMonitor
import com.example.llama.util.formatFileByteSize
import com.example.llama.util.getFileNameFromUri
import com.example.llama.util.getFileSizeFromUri
import com.example.llama.viewmodel.ModelManagementState.Deletion
import com.example.llama.viewmodel.ModelManagementState.Download
import com.example.llama.viewmodel.ModelManagementState.Importation
import com.example.llama.viewmodel.PreselectedModelToRun.RamWarning
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.FlowPreview
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
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
import java.io.FileNotFoundException
import java.io.IOException
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import javax.inject.Inject
import kotlin.coroutines.cancellation.CancellationException


@OptIn(FlowPreview::class)
@HiltViewModel
class ModelsViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val modelRepository: ModelRepository,
    private val performanceMonitor: PerformanceMonitor,
    private val inferenceService: InferenceService,
) : ViewModel() {

    // UI state: model management mode
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
    private val _filteredModels = MutableStateFlow<List<ModelInfo>>(emptyList())
    val filteredModels = _filteredModels.asStateFlow()

    // Data: queried models
    private val _queryResults = MutableStateFlow<List<ModelInfo>>(emptyList())
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

    // UI state: models selected in deleting mode
    private val _selectedModelsToDelete = MutableStateFlow<Map<String, ModelInfo>>(emptyMap())
    val selectedModelsToDelete: StateFlow<Map<String, ModelInfo>> = _selectedModelsToDelete.asStateFlow()

    fun toggleModelSelectionById(modelId: String) {
        val current = _selectedModelsToDelete.value.toMutableMap()
        val model = _filteredModels.value.find { it.id == modelId }

        if (model != null) {
            if (current.containsKey(modelId)) {
                current.remove(modelId)
            } else {
                current[modelId] = model
            }
            _selectedModelsToDelete.value = current
        }
    }

    fun toggleAllSelectedModelsToDelete(selectAll: Boolean) {
        if (selectAll) {
            _selectedModelsToDelete.value = _filteredModels.value.associateBy { it.id }
        } else {
            _selectedModelsToDelete.value = emptyMap()
        }
    }

    // UI state: import menu
    private val _showImportModelMenu = MutableStateFlow(false)
    val showImportModelMenu: StateFlow<Boolean> = _showImportModelMenu.asStateFlow()

    fun toggleImportMenu(show: Boolean) {
        _showImportModelMenu.value = show
    }

    // HuggingFace: ongoing query jobs
    private var huggingFaceQueryJob: Job? = null

    // HuggingFace: Ongoing download jobs
    private val activeDownloads = mutableMapOf<Long, HuggingFaceModel>()
    private val downloadReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            intent.getLongExtra(DownloadManager.EXTRA_DOWNLOAD_ID, -1).let { id ->
                if (id in activeDownloads) {
                    handleDownloadComplete(id)
                }
            }
        }
    }

    // Internal state
    private val _managementState = MutableStateFlow<ModelManagementState>(ModelManagementState.Idle)
    val managementState: StateFlow<ModelManagementState> = _managementState.asStateFlow()

    fun resetManagementState() {
        huggingFaceQueryJob?.let {
            if (it.isActive) { it.cancel() }
        }
        _managementState.value = ModelManagementState.Idle
    }

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

        val filter = IntentFilter(DownloadManager.ACTION_DOWNLOAD_COMPLETE)
        context.registerReceiver(downloadReceiver, filter, RECEIVER_EXPORTED)
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

    /**
     * First show confirmation instead of starting import local file immediately
     */
    fun importLocalModelFileSelected(uri: Uri) = viewModelScope.launch {
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
    fun importLocalModelFileConfirmed(uri: Uri, fileName: String, fileSize: Long) = viewModelScope.launch {
        try {
            _managementState.value = Importation.Importing(0f, fileName, fileSize)
            val model = modelRepository.importModel(uri, fileName, fileSize) { progress ->
                _managementState.value = Importation.Importing(progress, fileName, fileSize)
            }
            _managementState.value = Importation.Success(model)
        } catch (_: InvalidFileFormatException) {
            _managementState.value = Importation.Error(
                message = "Not a valid GGUF model!",
                learnMoreUrl = "https://huggingface.co/docs/hub/en/gguf",
            )
        } catch (e: InsufficientStorageException) {
            _managementState.value = Importation.Error(
                message = e.message ?: "Insufficient storage space to import $fileName",
                learnMoreUrl = "https://support.google.com/android/answer/7431795?hl=en",
            )
        } catch (e: Exception) {
            Log.e(TAG, "Unknown exception importing $fileName", e)
            _managementState.value = Importation.Error(
                message = e.message ?: "Unknown error importing $fileName",
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

    /**
     * Query models on HuggingFace available for download even without signing in
     */
    fun queryModelsFromHuggingFace() {
        huggingFaceQueryJob = viewModelScope.launch {
            _managementState.emit(Download.Querying)
            try {
                modelRepository.searchHuggingFaceModels().fold(
                    onSuccess = { models ->
                        Log.d(TAG, "Fetched ${models.size} models from HuggingFace:")
                        _managementState.emit(Download.Ready(models))
                    },
                    onFailure = { throw it }
                )
            } catch (_: CancellationException) {
                // no-op
            } catch (_: UnknownHostException) {
                _managementState.value = Download.Error(message = "No internet connection")
            } catch (_: SocketTimeoutException) {
                _managementState.value = Download.Error(message = "Connection timed out")
            } catch (_: FileNotFoundException) {
                _managementState.emit(Download.Error(message = "No eligible models"))
            } catch (e: IOException) {
                _managementState.value = Download.Error(message = "Network error: ${e.message}")
            } catch (e: Exception) {
                _managementState.emit(Download.Error(message = e.message ?: "Unknown error"))
            }
        }
    }

    /**
     * Dispatch download request to [DownloadManager] and update UI
     */
    fun downloadHuggingFaceModelConfirmed(model: HuggingFaceModel) = viewModelScope.launch {
        try {
            require(!model.gated) { "Model is gated!" }
            require(!model.private) { "Model is private!" }
            val downloadInfo = model.toDownloadInfo()
            requireNotNull(downloadInfo) { "Download URL is missing!" }

            modelRepository.getHuggingFaceModelFileSize(downloadInfo).fold(
                onSuccess = { actualSize ->
                    Log.d(TAG, "Model file size: ${formatFileByteSize(actualSize)}")
                    modelRepository.downloadHuggingFaceModel(downloadInfo, actualSize)
                        .onSuccess { downloadId ->
                            activeDownloads[downloadId] = model
                            _managementState.value = Download.Dispatched(downloadInfo)
                        }
                        .onFailure { throw it }
                },
                onFailure = { throw it }
            )
        } catch (_: UnknownHostException) {
            _managementState.value = Download.Error(message = "No internet connection")
        } catch (_: SocketTimeoutException) {
            _managementState.value = Download.Error(message = "Connection timed out")
        } catch (e: IOException) {
            _managementState.value = Download.Error(message = "Network error: ${e.message}")
        } catch (e: InsufficientStorageException) {
            _managementState.value = Download.Error(
                message = e.message ?: "Insufficient storage space to download ${model.modelId}",
            )
        } catch (e: Exception) {
            _managementState.value = Download.Error(
                message = e.message ?: "Unknown error downloading ${model.modelId}",
            )
        }
    }

    private fun handleDownloadComplete(downloadId: Long) = viewModelScope.launch {
        val model = activeDownloads.remove(downloadId) ?: return@launch

        (context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager)
            .getUriForDownloadedFile(downloadId)?.let { uri ->
                try {
                    val fileName = getFileNameFromUri(context, uri) ?: throw FileNotFoundException("File size N/A")
                    val fileSize = getFileSizeFromUri(context, uri) ?: throw FileNotFoundException("File name N/A")
                    _managementState.emit(Download.Completed(model, uri, fileName, fileSize))
                } catch (e: Exception) {
                    _managementState.value = Download.Error(
                        message = e.message ?: "Unknown error downloading ${model.modelId}"
                    )
                }
            }
    }

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
            toggleAllSelectedModelsToDelete(false)

            // Reset state after a delay
            delay(DELETE_SUCCESS_RESET_TIMEOUT_MS)
            _managementState.value = ModelManagementState.Idle
        } catch (e: Exception) {
            _managementState.value = Deletion.Error(
                message = e.message ?: "Error deleting $total models"
            )
        }
    }

    companion object {
        private val TAG = ModelsViewModel::class.java.simpleName

        private const val SUBSCRIPTION_TIMEOUT_MS = 5000L
        private const val QUERY_DEBOUNCE_TIMEOUT_MS = 500L

        private const val DELETE_SUCCESS_RESET_TIMEOUT_MS = 1000L

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

sealed class ModelManagementState {
    object Idle : ModelManagementState()

    sealed class Importation : ModelManagementState() {
        data class Confirming(val uri: Uri, val fileName: String, val fileSize: Long) : Importation()
        data class Importing(val progress: Float = 0f, val fileName: String, val fileSize: Long, val isCancelling: Boolean = false) : Importation()
        data class Success(val model: ModelInfo) : Importation()
        data class Error(val message: String, val learnMoreUrl: String? = null) : Importation()
    }

    sealed class Download: ModelManagementState() {
        object Querying : Download()
        data class Ready(val models: List<HuggingFaceModel>) : Download()
        data class Dispatched(val downloadInfo: HuggingFaceDownloadInfo) : Download()
        data class Completed(val model: HuggingFaceModel, val uri: Uri, val fileName: String, val fileSize: Long) : Download()
        data class Error(val message: String) : Download()
    }

    sealed class Deletion : ModelManagementState() {
        data class Confirming(val models: Map<String, ModelInfo>): ModelManagementState()
        data class Deleting(val progress: Float = 0f, val models: Map<String, ModelInfo>) : ModelManagementState()
        data class Success(val models: List<ModelInfo>) : Deletion()
        data class Error(val message: String) : Deletion()
    }
}
