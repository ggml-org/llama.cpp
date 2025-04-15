package com.example.llama.revamp.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.data.model.SystemPrompt
import com.example.llama.revamp.data.repository.SystemPromptRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * ViewModel for handling system prompts.
 */
@HiltViewModel
class SystemPromptViewModel @Inject constructor(
    private val repository: SystemPromptRepository
) : ViewModel() {

    // Preset prompts
    val presetPrompts: StateFlow<List<SystemPrompt>> = repository.getPresetPrompts()
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(SUBSCRIPTION_TIMEOUT_MS),
            initialValue = emptyList()
        )

    // Recent prompts
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

    companion object {
        private val TAG = SystemPromptViewModel::class.java.simpleName

        private const val SUBSCRIPTION_TIMEOUT_MS = 5000L
    }
}
