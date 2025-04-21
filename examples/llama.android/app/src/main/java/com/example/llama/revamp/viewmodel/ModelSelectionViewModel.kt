package com.example.llama.revamp.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.data.repository.ModelRepository
import com.example.llama.revamp.engine.InferenceService
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import javax.inject.Inject


@HiltViewModel
class ModelSelectionViewModel @Inject constructor(
    private val inferenceService: InferenceService,
    modelRepository: ModelRepository
) : ViewModel() {

    private val _preselectedModel = MutableStateFlow<ModelInfo?>(null)
    val preselectedModel: StateFlow<ModelInfo?> = _preselectedModel.asStateFlow()

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

        private const val SUBSCRIPTION_TIMEOUT_MS = 5000L
    }
}
