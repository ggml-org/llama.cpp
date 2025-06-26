package com.example.llama.viewmodel

import android.llama.cpp.isUninterruptible
import androidx.lifecycle.viewModelScope
import com.example.llama.data.model.ModelInfo
import com.example.llama.engine.BenchmarkService
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.flow.zip
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class BenchmarkViewModel @Inject constructor(
    private val benchmarkService: BenchmarkService
) : ModelUnloadingViewModel(benchmarkService) {

    // Data
    val selectedModel: StateFlow<ModelInfo?> = benchmarkService.currentSelectedModel

    private val _benchmarkDuration = MutableSharedFlow<Long>()

    private val _benchmarkResults = MutableStateFlow<List<BenchmarkResult>>(emptyList())
    val benchmarkResults: StateFlow<List<BenchmarkResult>> = _benchmarkResults.asStateFlow()

     // UI state: Model card
    private val _showModelCard = MutableStateFlow(true)
    val showModelCard = _showModelCard.asStateFlow()

    fun toggleModelCard(show: Boolean) {
        _showModelCard.value = show
    }

    init {
        viewModelScope.launch {
            benchmarkService.benchmarkResults
                .filterNotNull()
                .zip(_benchmarkDuration) { result, duration ->
                    _benchmarkResults.update { oldResults ->
                        oldResults.toMutableList().apply {
                            add(BenchmarkResult(result, duration))
                        }
                    }
                }.collect()
        }
    }

    /**
     * Run benchmark with specified parameters
     */
    fun runBenchmark(pp: Int = 512, tg: Int = 128, pl: Int = 1, nr: Int = 3): Boolean {
        if (engineState.value.isUninterruptible) {
            return false
        }

        viewModelScope.launch {
            val benchmarkStartTs = System.currentTimeMillis()
            benchmarkService.benchmark(pp, tg, pl, nr)
            val benchmarkEndTs = System.currentTimeMillis()
            _benchmarkDuration.emit(benchmarkEndTs - benchmarkStartTs)
        }
        return true
    }

    override suspend fun performCleanup() = clearResults()

    fun clearResults() {
        _benchmarkResults.value = emptyList()
    }
}

data class BenchmarkResult(
    val text: String,
    val duration: Long
)
