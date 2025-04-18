package com.example.llama.revamp.viewmodel

import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.engine.BenchmarkService
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class BenchmarkViewModel @Inject constructor(
    private val benchmarkService: BenchmarkService
) : ModelUnloadingViewModel(benchmarkService) {
    /**
     * UI states
     */
    val selectedModel: StateFlow<ModelInfo?> = benchmarkService.currentSelectedModel
    val benchmarkResults: StateFlow<String?> = benchmarkService.benchmarkResults

    /**
     * Run benchmark with specified parameters
     */
    fun runBenchmark(pp: Int = 512, tg: Int = 128, pl: Int = 1, nr: Int = 3) =
        viewModelScope.launch {
            benchmarkService.benchmark(pp, tg, pl, nr)
        }
}
