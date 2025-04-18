package com.example.llama.revamp.viewmodel

import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.data.model.SystemPrompt
import com.example.llama.revamp.data.repository.SystemPromptRepository
import com.example.llama.revamp.engine.ModelLoadingService
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class ModelLoadingViewModel @Inject constructor(
    private val modelLoadingService: ModelLoadingService,
    private val repository: SystemPromptRepository
) : ModelUnloadingViewModel(modelLoadingService) {

    /**
     * Currently selected model to be loaded
     */
    val selectedModel = modelLoadingService.currentSelectedModel

    /**
     * Preset prompts
     */
    val presetPrompts: StateFlow<List<SystemPrompt>> = repository.getPresetPrompts()
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(SUBSCRIPTION_TIMEOUT_MS),
            initialValue = emptyList()
        )

    /**
     * Recent prompts
     */
    val recentPrompts: StateFlow<List<SystemPrompt>> = repository.getRecentPrompts()
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(SUBSCRIPTION_TIMEOUT_MS),
            initialValue = emptyList()
        )

    /**
     * Save a prompt to the recents list.
     */
    fun savePromptToRecents(prompt: SystemPrompt) {
        viewModelScope.launch {
            repository.savePromptToRecents(prompt)
        }
    }

    /**
     * Create and save a custom prompt.
     */
    fun saveCustomPromptToRecents(content: String) {
        viewModelScope.launch {
            repository.saveCustomPrompt(content)
        }
    }

    /**
     * Delete a prompt by ID.
     */
    fun deletePrompt(id: String) {
        viewModelScope.launch {
            repository.deletePrompt(id)
        }
    }

    /**
     * Clear all recent prompts.
     */
    fun clearRecentPrompts() {
        viewModelScope.launch {
            repository.deleteAllPrompts()
        }
    }

    /**
     * Prepares the engine for benchmark mode.
     */
    suspend fun prepareForBenchmark() =
        modelLoadingService.loadModelForBenchmark()

    /**
     * Prepare for conversation
     */
    suspend fun prepareForConversation(systemPrompt: String? = null) =
        modelLoadingService.loadModelForConversation(systemPrompt)


    companion object {
        private val TAG = ModelLoadingViewModel::class.java.simpleName

        private const val SUBSCRIPTION_TIMEOUT_MS = 5000L
    }
}
