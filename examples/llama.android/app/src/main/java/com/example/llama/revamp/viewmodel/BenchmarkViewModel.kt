package com.example.llama.revamp.viewmodel

import android.llama.cpp.isUninterruptible
import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.engine.BenchmarkService
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
    /**
     * UI states
     */
    val selectedModel: StateFlow<ModelInfo?> = benchmarkService.currentSelectedModel

    private val _benchmarkDuration = MutableSharedFlow<Long>()

    private val _benchmarkResults = MutableStateFlow<List<BenchmarkResult>>(emptyList())
    val benchmarkResults: StateFlow<List<BenchmarkResult>> = _benchmarkResults.asStateFlow()

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
}

data class BenchmarkResult(
    val text: String,
    val duration: Long
)
