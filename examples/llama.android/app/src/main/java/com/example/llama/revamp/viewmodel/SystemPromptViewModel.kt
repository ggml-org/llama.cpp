package com.example.llama.revamp.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
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
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = emptyList()
        )

    // Recent prompts
    val recentPrompts: StateFlow<List<SystemPrompt>> = repository.getRecentPrompts()
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
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
     * Factory for creating SystemPromptViewModel instances.
     */
    class Factory(
        private val repository: SystemPromptRepository
    ) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            if (modelClass.isAssignableFrom(SystemPromptViewModel::class.java)) {
                return SystemPromptViewModel(repository) as T
            }
            throw IllegalArgumentException("Unknown ViewModel class")
        }
    }
}
