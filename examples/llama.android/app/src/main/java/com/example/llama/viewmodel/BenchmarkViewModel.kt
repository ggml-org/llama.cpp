package com.example.llama.viewmodel

import android.llama.cpp.isUninterruptible
import androidx.lifecycle.viewModelScope
import com.example.llama.data.model.ModelInfo
import com.example.llama.engine.BenchmarkService
import com.example.llama.ui.scaffold.ScaffoldEvent
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
    private val _showModelCard = MutableStateFlow(false)
    val showModelCard = _showModelCard.asStateFlow()

    fun toggleModelCard(show: Boolean) {
        _showModelCard.value = show
    }

    // UI state: Share FAB
    private val _showShareFab = MutableStateFlow(false)
    val showShareFab = _showShareFab.asStateFlow()

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
            _showShareFab.value = false
            val benchmarkStartTs = System.currentTimeMillis()
            benchmarkService.benchmark(pp, tg, pl, nr)
            val benchmarkEndTs = System.currentTimeMillis()
            _benchmarkDuration.emit(benchmarkEndTs - benchmarkStartTs)
            _showShareFab.value = true
        }
        return true
    }

    override suspend fun performCleanup() { clearResults(null) }

    fun clearResults(onScaffoldEvent: ((ScaffoldEvent) -> Unit)?) =
        if (engineState.value.isUninterruptible) {
            false
        } else {
            _benchmarkResults.value = emptyList()
            _showShareFab.value = false
            onScaffoldEvent?.invoke(ScaffoldEvent.ShowSnackbar(
                message = "All benchmark results cleared."
            ))
            true
        }

    /**
     * Rerun the benchmark
     */
    fun rerunBenchmark(onScaffoldEvent: (ScaffoldEvent) -> Unit) {
        if (engineState.value.isUninterruptible) {
            onScaffoldEvent(ScaffoldEvent.ShowSnackbar(
                message = "Benchmark already in progress!\n" +
                    "Please wait for the current run to complete."
            ))
        } else {
            runBenchmark()
        }
    }

    fun shareResult(onScaffoldEvent: (ScaffoldEvent) -> Unit) {
        _benchmarkResults.value.lastOrNull()?.let{
            onScaffoldEvent(ScaffoldEvent.ShareText(it.text))
        }
    }
}

data class BenchmarkResult(
    val text: String,
    val duration: Long
)
