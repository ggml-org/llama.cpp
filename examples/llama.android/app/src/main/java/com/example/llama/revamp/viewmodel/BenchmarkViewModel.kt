package com.example.llama.revamp.viewmodel

import android.llama.cpp.InferenceEngine.State
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.engine.BenchmarkService
import com.example.llama.revamp.ui.components.UnloadDialogState
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class BenchmarkViewModel @Inject constructor(
    private val benchmarkService: BenchmarkService
) : ViewModel() {
    /**
     * Core states
     */
    val engineState: StateFlow<State> = benchmarkService.engineState
    val selectedModel: StateFlow<ModelInfo?> = benchmarkService.currentSelectedModel
    val benchmarkResults: StateFlow<String?> = benchmarkService.benchmarkResults

    /**
     * Model unloading dialog state
     */
    private val _unloadDialogState = MutableStateFlow<UnloadDialogState>(UnloadDialogState.Hidden)
    val unloadDialogState: StateFlow<UnloadDialogState> = _unloadDialogState.asStateFlow()

    /**
     * Run benchmark with specified parameters
     */
    fun runBenchmark(pp: Int = 512, tg: Int = 128, pl: Int = 1, nr: Int = 3) =
        viewModelScope.launch {
            benchmarkService.benchmark(pp, tg, pl, nr)
        }

    /**
     * Handle back press from both back button and top bar
     */
    fun onBackPressed() {
        when (engineState.value) {
            State.Benchmarking -> {
                // Ignore back navigation requests during active benchmarking
            }
            else -> _unloadDialogState.value = UnloadDialogState.Confirming
        }
    }

    /**
     * Handle confirmation from unload dialog
     */
    fun onUnloadConfirmed(onNavigateBack: () -> Unit) {
        viewModelScope.launch {
            // Set unloading state to show progress
            _unloadDialogState.value = UnloadDialogState.Unloading
            android.util.Log.d("JOJO", "onUnloadConfirmed $ state -> Unloading")

            try {
                // Unload the model
                benchmarkService.unloadModel()
                android.util.Log.d("JOJO", "onUnloadConfirmed $ service unload model finished!")

                // Reset state and navigate back
                _unloadDialogState.value = UnloadDialogState.Hidden
                android.util.Log.d("JOJO", "onUnloadConfirmed $ state -> Hidden!")
                onNavigateBack()
            } catch (e: Exception) {
                // Handle error if needed
                _unloadDialogState.value = UnloadDialogState.Hidden
            }
        }
    }

    /**
     * Handle dismissal of unload dialog
     */
    fun onUnloadDismissed() {
        when (_unloadDialogState.value) {
            is UnloadDialogState.Unloading -> {
                // Ignore dismissing requests during active benchmarking
            }
            else -> _unloadDialogState.value = UnloadDialogState.Hidden
        }
    }
}
