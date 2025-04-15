package com.example.llama.revamp.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.engine.InferenceEngine
import com.example.llama.revamp.engine.InferenceManager
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class BenchmarkViewModel @Inject constructor(
    private val inferenceManager: InferenceManager
) : ViewModel() {

    val engineState: StateFlow<InferenceEngine.State> = inferenceManager.engineState
    val benchmarkResults: StateFlow<String?> = inferenceManager.benchmarkResults
    val selectedModel: StateFlow<ModelInfo?> = inferenceManager.currentModel

    /**
     * Run benchmark with specified parameters
     */
    fun runBenchmark(pp: Int = 512, tg: Int = 128, pl: Int = 1, nr: Int = 3) =
        viewModelScope.launch {
            inferenceManager.benchmark(pp, tg, pl, nr)
        }
}
