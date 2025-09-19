package com.arm.aiplayground.viewmodel

import androidx.lifecycle.viewModelScope
import com.arm.aiplayground.data.model.SystemPrompt
import com.arm.aiplayground.data.repo.ModelRepository
import com.arm.aiplayground.data.repo.SystemPromptRepository
import com.arm.aiplayground.engine.ModelLoadingMetrics
import com.arm.aiplayground.engine.ModelLoadingService
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class ModelLoadingViewModel @Inject constructor(
    private val modelLoadingService: ModelLoadingService,
    private val systemPromptRepository: SystemPromptRepository,
    private val modelRepository: ModelRepository,
) : ModelUnloadingViewModel(modelLoadingService) {

    /**
     * Currently selected model to be loaded
     */
    val selectedModel = modelLoadingService.currentSelectedModel

    /**
     * Preset prompts
     */
    val presetPrompts: StateFlow<List<SystemPrompt>> = systemPromptRepository.getPresetPrompts()
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(SUBSCRIPTION_TIMEOUT_MS),
            initialValue = emptyList()
        )

    /**
     * Recent prompts
     */
    val recentPrompts: StateFlow<List<SystemPrompt>> = systemPromptRepository.getRecentPrompts()
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
            systemPromptRepository.savePromptToRecents(prompt)
        }
    }

    /**
     * Create and save a custom prompt.
     */
    fun saveCustomPromptToRecents(content: String) {
        viewModelScope.launch {
            systemPromptRepository.saveCustomPrompt(content)
        }
    }

    /**
     * Delete a prompt by ID.
     */
    fun deletePrompt(id: String) {
        viewModelScope.launch {
            systemPromptRepository.deletePrompt(id)
        }
    }

    /**
     * Clear all recent prompts.
     */
    fun clearRecentPrompts() {
        viewModelScope.launch {
            systemPromptRepository.deleteAllPrompts()
        }
    }

    /**
     * Loads the model, then navigate to [BenchmarkScreen] with [ModelLoadingMetrics]
     */
    fun onBenchmarkSelected(onNavigateToBenchmark: (ModelLoadingMetrics) -> Unit) =
        viewModelScope.launch {
            selectedModel.value?.let { model ->
                modelLoadingService.loadModelForBenchmark()?.let { metrics ->
                    modelRepository.updateModelLastUsed(model.id)
                    onNavigateToBenchmark(metrics)
                }
            }
        }

    /**
     * Loads the model, process system prompt if any,
     * then navigate to [ConversationScreen] with [ModelLoadingMetrics]
     */
    fun onConversationSelected(
        systemPrompt: String? = null,
        onNavigateToConversation: (ModelLoadingMetrics) -> Unit
    ) = viewModelScope.launch {
        selectedModel.value?.let { model ->
            modelLoadingService.loadModelForConversation(systemPrompt)?.let { metrics ->
                modelRepository.updateModelLastUsed(model.id)
                onNavigateToConversation(metrics)
            }
        }
    }

    companion object {
        private val TAG = ModelLoadingViewModel::class.java.simpleName

        private const val SUBSCRIPTION_TIMEOUT_MS = 5000L
    }
}
