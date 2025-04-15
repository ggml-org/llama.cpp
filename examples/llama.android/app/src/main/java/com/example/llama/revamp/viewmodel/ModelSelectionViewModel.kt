package com.example.llama.revamp.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.data.repository.ModelRepository
import com.example.llama.revamp.engine.InferenceManager
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject


@HiltViewModel
class ModelSelectionViewModel @Inject constructor(
    private val inferenceManager: InferenceManager,
    private val modelRepository: ModelRepository
) : ViewModel() {

    /**
     * Available models for selection
     */
    val availableModels: StateFlow<List<ModelInfo>> = modelRepository.getModels()
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(SUBSCRIPTION_TIMEOUT_MS),
            initialValue = emptyList()
        )

    /**
     * Access to currently selected model
     */
    val selectedModel = inferenceManager.currentModel

    /**
     * Select a model and update its last used timestamp
     */
    fun selectModel(modelInfo: ModelInfo) {
        inferenceManager.setCurrentModel(modelInfo)

        viewModelScope.launch {
            modelRepository.updateModelLastUsed(modelInfo.id)
        }
    }

    /**
     * Unload model when navigating away
     */
    suspend fun unloadModel() = inferenceManager.unloadModel()

    companion object {
        private val TAG = ModelSelectionViewModel::class.java.simpleName

        private const val SUBSCRIPTION_TIMEOUT_MS = 5000L
    }
}
