package com.example.llama.revamp.engine

import android.llama.cpp.InferenceEngine
import android.llama.cpp.InferenceEngine.State
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.flow
import org.jetbrains.annotations.TestOnly
import org.jetbrains.annotations.VisibleForTesting
import javax.inject.Singleton

/**
 * A stub [InferenceEngine] for agile development & testing
 */
@VisibleForTesting
@TestOnly
@Singleton
class StubInferenceEngine : InferenceEngine {
    companion object {
        private val TAG = StubInferenceEngine::class.java.simpleName

        private const val STUB_MODEL_LOADING_TIME = 2000L
        private const val STUB_BENCHMARKING_TIME = 4000L
        private const val STUB_SYSTEM_PROMPT_PROCESSING_TIME = 3000L
        private const val STUB_USER_PROMPT_PROCESSING_TIME = 1500L
        private const val STUB_TOKEN_GENERATION_TIME = 200L
    }

    private val _state = MutableStateFlow<State>(State.Uninitialized)
    override val state: StateFlow<State> = _state

    init {
        Log.i(TAG, "Initiated!")

        // Simulate library loading
        _state.value = State.LibraryLoaded
    }

    /**
     * Loads a model from the given path with an optional system prompt.
     */
    override suspend fun loadModel(pathToModel: String, systemPrompt: String?) {
        Log.i(TAG, "loadModel! state: ${_state.value}")

        try {
            _state.value = State.LoadingModel

            // Simulate model loading
            delay(STUB_MODEL_LOADING_TIME)

            _state.value = State.ModelLoaded

            if (systemPrompt != null) {
                _state.value = State.ProcessingSystemPrompt

                // Simulate processing system prompt
                delay(STUB_SYSTEM_PROMPT_PROCESSING_TIME)
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
    override fun sendUserPrompt(message: String, predictLength: Int): Flow<String> {
        Log.i(TAG, "sendUserPrompt! state: ${_state.value}")

        _state.value = State.ProcessingUserPrompt

        // This would be replaced with actual token generation logic
        return flow {
            try {
                // Simulate longer processing time (1.5 seconds)
                delay(STUB_USER_PROMPT_PROCESSING_TIME)

                _state.value = State.Generating

                // Simulate token generation
                val response = "This is a simulated response from the LLM model. The actual implementation would generate tokens one by one based on the input: $message"
                response.split(" ").forEach {
                    emit("$it ")
                    delay(STUB_TOKEN_GENERATION_TIME)
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
    override suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int): String {
        Log.i(TAG, "bench! state: ${_state.value}")

        _state.value = State.Benchmarking

        try {
            // Simulate benchmark running
            delay(STUB_BENCHMARKING_TIME)

            // Generate fake benchmark results
            val modelDesc = "Kleidi Llama"
            val model_size = "7"
            val model_n_params = "7"
            val backend = "CPU"

            // Random values for benchmarks
            val pp_avg = (51.4 + Math.random() * 5.14).toFloat()
            val pp_std = (5.14 + Math.random() * 0.514).toFloat()
            val tg_avg = (11.4 + Math.random() * 1.14).toFloat()
            val tg_std = (1.14 + Math.random() * 0.114).toFloat()

            val result = StringBuilder()
            result.append("| model | size | params | backend | test | t/s |\n")
            result.append("| --- | --- | --- | --- | --- | --- |\n")
            result.append("| $modelDesc | ${model_size}GiB | ${model_n_params}B | ")
            result.append("$backend | pp $pp | $pp_avg ± $pp_std |\n")
            result.append("| $modelDesc | ${model_size}GiB | ${model_n_params}B | ")
            result.append("$backend | tg $tg | $tg_avg ± $tg_std |\n")

            _state.value = State.AwaitingUserPrompt

            return result.toString()
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
    override suspend fun unloadModel() {
        Log.i(TAG, "unloadModel! state: ${_state.value}")

        // Simulate model unloading time
        delay(2000)
        _state.value = State.LibraryLoaded
    }

    /**
     * Cleans up resources when the engine is no longer needed.
     */
    override fun destroy() {
        Log.i(TAG, "destroy! state: ${_state.value}")

        _state.value = State.Uninitialized
    }
}
