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
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.data.model.ModelInfo
import com.example.llama.data.repo.InsufficientStorageException
import com.example.llama.data.repo.ModelRepository
import com.example.llama.data.source.remote.HuggingFaceDownloadInfo
import com.example.llama.data.source.remote.HuggingFaceModel
import com.example.llama.util.formatFileByteSize
import com.example.llama.util.getFileNameFromUri
import com.example.llama.util.getFileSizeFromUri
import com.example.llama.viewmodel.ModelManagementState.Deletion
import com.example.llama.viewmodel.ModelManagementState.Download
import com.example.llama.viewmodel.ModelManagementState.Importation
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.FileNotFoundException
import java.io.IOException
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import javax.inject.Inject
import kotlin.coroutines.cancellation.CancellationException

@HiltViewModel
class ModelsManagementViewModel @Inject constructor(
    @ApplicationContext private val context: Context,
    private val modelRepository: ModelRepository,
) : ViewModel() {
    // UI state: models selected to be batch-deleted
    private val _selectedModelsToDelete = MutableStateFlow<Map<String, ModelInfo>>(emptyMap())
    val selectedModelsToDelete: StateFlow<Map<String, ModelInfo>> = _selectedModelsToDelete.asStateFlow()

    fun toggleModelSelectionById(filteredModels: List<ModelInfo>, modelId: String) {
        val current = _selectedModelsToDelete.value.toMutableMap()
        val model = filteredModels.find { it.id == modelId }

        if (model != null) {
            if (current.containsKey(modelId)) {
                current.remove(modelId)
            } else {
                current[modelId] = model
            }
            _selectedModelsToDelete.value = current
        }
    }

    fun selectModelsToDelete(models: List<ModelInfo>) {
        _selectedModelsToDelete.value = models.associateBy { it.id }
    }

    fun clearSelectedModelsToDelete() {
        _selectedModelsToDelete.value = emptyMap()
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
        clearSelectedModelsToDelete()
        _managementState.value = ModelManagementState.Idle
    }

    init {
        context.registerReceiver(
            downloadReceiver,
            IntentFilter(DownloadManager.ACTION_DOWNLOAD_COMPLETE),
            RECEIVER_EXPORTED
        )
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
    fun deleteModels(modelsToDelete: Map<String, ModelInfo>) = viewModelScope.launch {
        val total = modelsToDelete.size
        if (total == 0) return@launch

        try {
            _managementState.value = Deletion.Deleting(0f, modelsToDelete)
            var deleted = 0
            modelsToDelete.keys.toList().forEach {
                modelRepository.deleteModel(it)
                deleted++
                _managementState.value = Deletion.Deleting(deleted.toFloat() / total, modelsToDelete)
            }
            _managementState.value = Deletion.Success(modelsToDelete.values.toList())
            clearSelectedModelsToDelete()

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
        private val TAG = ModelsManagementViewModel::class.java.simpleName

        private const val DELETE_SUCCESS_RESET_TIMEOUT_MS = 1000L
    }
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
