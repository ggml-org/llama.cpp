package com.example.llama.revamp.engine

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.flow

/**
 * LLM inference engine that handles model loading and text generation.
 */
class InferenceEngine {
    companion object {
        const val DEFAULT_PREDICT_LENGTH = 1024
    }

    sealed class State {
        object Uninitialized : State()
        object LibraryLoaded : State()

        object LoadingModel : State()
        object ModelLoaded : State()

        object ProcessingSystemPrompt : State()
        object AwaitingUserPrompt : State()

        object ProcessingUserPrompt : State()
        object Generating : State()

        object Benchmarking : State()

        data class Error(
            val errorMessage: String = ""
        ) : State()
    }

    private val _state = MutableStateFlow<State>(State.Uninitialized)
    val state: StateFlow<State> = _state

    // Keep track of current benchmark results
    private var _benchmarkResults: String? = null
    private val _benchmarkResultsFlow = MutableStateFlow<String?>(null)
    val benchmarkResults: StateFlow<String?> = _benchmarkResultsFlow

    init {
        // Simulate library loading
        _state.value = State.LibraryLoaded
    }

    /**
     * Loads a model from the given path with an optional system prompt.
     */
    suspend fun loadModel(pathToModel: String, systemPrompt: String? = null) {
        try {
            _state.value = State.LoadingModel

            // Simulate model loading
            delay(2000)

            _state.value = State.ModelLoaded

            if (systemPrompt != null) {
                _state.value = State.ProcessingSystemPrompt

                // Simulate processing system prompt
                delay(3000)
            }

            _state.value = State.AwaitingUserPrompt
        } catch (e: CancellationException) {
            // If coroutine is cancelled, propagate cancellation
            throw e
        } catch (e: Exception) {
            _state.value = State.Error(e.message ?: "Unknown error during model loading")
        }
    }

    /**
     * Sends a user prompt to the loaded model and returns a Flow of generated tokens.
     */
    fun sendUserPrompt(message: String, predictLength: Int = DEFAULT_PREDICT_LENGTH): Flow<String> {
        _state.value = State.ProcessingUserPrompt

        // This would be replaced with actual token generation logic
        return flow {
            try {
                // Simulate longer processing time (1.5 seconds)
                delay(1500)

                _state.value = State.Generating

                // Simulate token generation
                val response = "This is a simulated response from the LLM model. The actual implementation would generate tokens one by one based on the input: $message"
                val words = response.split(" ")

                for (word in words) {
                    emit(word + " ")
                    // Slower token generation (200ms per token instead of 50ms)
                    delay(200)
                }

                _state.value = State.AwaitingUserPrompt
            } catch (e: CancellationException) {
                // Handle cancellation gracefully
                _state.value = State.AwaitingUserPrompt
                throw e
            } catch (e: Exception) {
                _state.value = State.Error(e.message ?: "Unknown error during generation")
                throw e
            }
        }.catch { e ->
            // If it's not a cancellation, update state to error
            if (e !is CancellationException) {
                _state.value = State.Error(e.message ?: "Unknown error during generation")
            }
            throw e
        }
    }

    /**
     * Runs a benchmark with the specified parameters.
     */
    suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1): String {
        _state.value = State.Benchmarking

        try {
            // Simulate benchmark running
            delay(4000)

            // Generate fake benchmark results
            val modelDesc = "LlamaModel"
            val model_size = "7"
            val model_n_params = "7"
            val backend = "CPU"

            // Random values for benchmarks
            val pp_avg = (15.0 + Math.random() * 10.0).toFloat()
            val pp_std = (0.5 + Math.random() * 2.0).toFloat()
            val tg_avg = (20.0 + Math.random() * 15.0).toFloat()
            val tg_std = (0.7 + Math.random() * 3.0).toFloat()

            val result = StringBuilder()
            result.append("| model | size | params | backend | test | t/s |\n")
            result.append("| --- | --- | --- | --- | --- | --- |\n")
            result.append("| $modelDesc | ${model_size}GiB | ${model_n_params}B | ")
            result.append("$backend | pp $pp | $pp_avg ± $pp_std |\n")
            result.append("| $modelDesc | ${model_size}GiB | ${model_n_params}B | ")
            result.append("$backend | tg $tg | $tg_avg ± $tg_std |\n")

            _benchmarkResults = result.toString()
            _benchmarkResultsFlow.value = _benchmarkResults

            _state.value = State.AwaitingUserPrompt

            return _benchmarkResults ?: ""
        } catch (e: CancellationException) {
            // If coroutine is cancelled, propagate cancellation
            _state.value = State.AwaitingUserPrompt
            throw e
        } catch (e: Exception) {
            _state.value = State.Error(e.message ?: "Unknown error during benchmarking")
            return "Error: ${e.message}"
        }
    }

    /**
     * Unloads the currently loaded model.
     */
    suspend fun unloadModel() {
        // Simulate model unloading time
        delay(2000)
        _state.value = State.LibraryLoaded
        _benchmarkResults = null
        _benchmarkResultsFlow.value = null
    }

    /**
     * Cleans up resources when the engine is no longer needed.
     */
    fun destroy() {
        // In a real implementation, this would release native resources
    }
}
