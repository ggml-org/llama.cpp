package com.example.llama.revamp.viewmodel

import android.llama.cpp.InferenceEngine
import android.llama.cpp.InferenceEngine.State
import android.llama.cpp.isModelLoaded
import android.llama.cpp.isUninterruptible
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.engine.InferenceService
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch


/**
 * UI states to be consumed by [ModelUnloadDialogHandler], etc.
 */
sealed class UnloadModelState {
    object Hidden : UnloadModelState()
    object Confirming : UnloadModelState()
    object Unloading : UnloadModelState()
    data class Error(val message: String) : UnloadModelState()
}

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
    private val _unloadModelState = MutableStateFlow<UnloadModelState>(UnloadModelState.Hidden)
    val unloadModelState: StateFlow<UnloadModelState> = _unloadModelState.asStateFlow()

    /**
     * Handle back press from both back button and top bar
     *
     * Subclass can override this default implementation
     */
    open fun onBackPressed(onNavigateBack: () -> Unit) =
        if (isUninterruptible) {
            // During uninterruptible operations, ignore back navigation requests
        } else if (!isModelLoaded) {
            // If model not loaded, no need to unload at all, directly perform back navigation
            onNavigateBack.invoke()
        } else {
            // If model is loaded, show confirmation dialog
            _unloadModelState.value = UnloadModelState.Confirming
        }

    /**
     * Handle confirmation from unload dialog
     */
    fun onUnloadConfirmed(onNavigateBack: () -> Unit) =
        viewModelScope.launch {
            // Set unloading state to show progress
            _unloadModelState.value = UnloadModelState.Unloading

            try {
                // Perform screen-specific cleanup
                performCleanup()

                // Unload the model
                inferenceService.unloadModel()

                // Reset state and navigate back
                _unloadModelState.value = UnloadModelState.Hidden
                onNavigateBack()
            } catch (e: Exception) {
                // Handle error
                _unloadModelState.value = UnloadModelState.Error(
                    e.message ?: "Unknown error while unloading the model"
                )
            }
        }

    /**
     * Handle dismissal of unload dialog
     */
    fun onUnloadDismissed() =
        when (_unloadModelState.value) {
            is UnloadModelState.Unloading -> {
                // Ignore dismissing requests during unloading
            }
            else -> _unloadModelState.value = UnloadModelState.Hidden
        }

    /**
     * Perform any screen-specific cleanup before unloading the model
     *
     * To be implemented by subclasses if needed
     */
    protected open suspend fun performCleanup() {}
}
