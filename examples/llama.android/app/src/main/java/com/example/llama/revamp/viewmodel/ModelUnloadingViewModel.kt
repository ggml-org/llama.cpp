package com.example.llama.revamp.viewmodel

import android.llama.cpp.InferenceEngine
import android.llama.cpp.InferenceEngine.State
import android.llama.cpp.isModelLoaded
import android.llama.cpp.isUninterruptible
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.engine.InferenceService
import com.example.llama.revamp.ui.components.UnloadDialogState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Base ViewModel class for screens that requires additional model unloading functionality
 */
abstract class ModelUnloadingViewModel(
    private val inferenceService: InferenceService
) : ViewModel() {

    /**
     * [InferenceEngine]'s core state
     */
    val engineState: StateFlow<State> = inferenceService.engineState

    /**
     * Determine if the screen is in an uninterruptible state
     *
     * Subclass can override this default implementation
     */
    protected open val isUninterruptible: Boolean
        get() = engineState.value.isUninterruptible

    protected open val isModelLoaded: Boolean
        get() = engineState.value.isModelLoaded

    /**
     * [UnloadModelConfirmationDialog]'s UI states
     */
    private val _unloadDialogState = MutableStateFlow<UnloadDialogState>(UnloadDialogState.Hidden)
    val unloadDialogState: StateFlow<UnloadDialogState> = _unloadDialogState.asStateFlow()

    /**
     * Handle back press from both back button and top bar
     */
    open fun onBackPressed(onNavigateBack: () -> Unit) =
        if (isUninterruptible) {
            // During uninterruptible operations, ignore back navigation requests
        } else if (!isModelLoaded) {
            // If model not loaded, no need to unload at all, directly perform back navigation
            onNavigateBack.invoke()
        } else {
            // If model is loaded, show confirmation dialog
            _unloadDialogState.value = UnloadDialogState.Confirming
        }

    /**
     * Handle confirmation from unload dialog
     */
    fun onUnloadConfirmed(onNavigateBack: () -> Unit) =
        viewModelScope.launch {
            // Set unloading state to show progress
            _unloadDialogState.value = UnloadDialogState.Unloading

            try {
                // Perform screen-specific cleanup
                performCleanup()

                // Unload the model
                inferenceService.unloadModel()

                // Reset state and navigate back
                _unloadDialogState.value = UnloadDialogState.Hidden
                onNavigateBack()
            } catch (e: Exception) {
                // Handle error
                _unloadDialogState.value = UnloadDialogState.Error(
                    e.message ?: "Unknown error while unloading the model"
                )
            }
        }

    /**
     * Handle dismissal of unload dialog
     */
    fun onUnloadDismissed() =
        when (_unloadDialogState.value) {
            is UnloadDialogState.Unloading -> {
                // Ignore dismissing requests during unloading
            }
            else -> _unloadDialogState.value = UnloadDialogState.Hidden
        }

    /**
     * Perform any screen-specific cleanup before unloading the model
     *
     * To be implemented by subclasses if needed
     */
    protected open suspend fun performCleanup() {
        // Default empty implementation
    }
}
