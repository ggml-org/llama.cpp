package com.example.llama.engine

import android.llama.cpp.InferenceEngine
import android.llama.cpp.InferenceEngine.State
import android.util.Log
import com.example.llama.APP_NAME
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import javax.inject.Singleton

/**
 * A stub [InferenceEngine] for agile development & testing
 */
@Singleton
class StubInferenceEngine : InferenceEngine {
    companion object {
        private val TAG = StubInferenceEngine::class.java.simpleName

        private const val STUB_LIBRARY_LOADING_TIME = 2_000L
        private const val STUB_MODEL_LOADING_TIME = 3_000L
        private const val STUB_MODEL_UNLOADING_TIME = 2_000L
        private const val STUB_BENCHMARKING_TIME = 8_000L
        private const val STUB_SYSTEM_PROMPT_PROCESSING_TIME = 4_000L
        private const val STUB_USER_PROMPT_PROCESSING_TIME = 2_000L
        private const val STUB_TOKEN_GENERATION_TIME = 200L
    }

    private val _state = MutableStateFlow<State>(State.Uninitialized)
    override val state: StateFlow<State> = _state

    private var _readyForSystemPrompt = false

    private val llamaDispatcher = Dispatchers.IO.limitedParallelism(1)
    private val llamaScope = CoroutineScope(llamaDispatcher + SupervisorJob())

    init {
        llamaScope.launch {
            Log.i(TAG, "Loading and initializing native library!")
            _state.value = State.Initializing

            // Simulate library loading
            delay(STUB_LIBRARY_LOADING_TIME)

            Log.i(TAG, "Native library initialized!")
            _state.value = State.Initialized
        }
    }

    /**
     * Loads a model from the given path.
     */
    override suspend fun loadModel(pathToModel: String) =
        withContext(llamaDispatcher) {
            Log.i(TAG, "loadModel! state: ${_state.value.javaClass.simpleName}")
            check(_state.value is State.Initialized) {
                "Cannot load model at ${_state.value.javaClass.simpleName}"
            }

            try {
                _readyForSystemPrompt = false
                _state.value = State.LoadingModel

                // Simulate model loading
                delay(STUB_MODEL_LOADING_TIME)

                _readyForSystemPrompt = true
                _state.value = State.ModelReady

            } catch (e: CancellationException) {
                // If coroutine is cancelled, propagate cancellation
                throw e
            } catch (e: Exception) {
                _state.value = State.Error(e.message ?: "Unknown error during model loading")
            }
        }

    /**
     * Process the plain text system prompt
     */
    override suspend fun setSystemPrompt(prompt: String) =
        withContext(llamaDispatcher) {
            check(_state.value is State.ModelReady) {
                "Cannot load model at ${_state.value.javaClass.simpleName}"
            }
            check(_readyForSystemPrompt) {
                "System prompt must be set ** RIGHT AFTER ** model loaded!"
            }

            try {
                _state.value = State.ProcessingSystemPrompt

                // Simulate processing system prompt
                delay(STUB_SYSTEM_PROMPT_PROCESSING_TIME)

                _state.value = State.ModelReady
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
    override fun sendUserPrompt(message: String, predictLength: Int): Flow<String> = flow {
        require(message.isNotEmpty()) { "User prompt discarded due to being empty!" }
        check(_state.value is State.ModelReady) {
            "Cannot load model at ${_state.value.javaClass.simpleName}"
        }

        try {
            Log.i(TAG, "sendUserPrompt! \n$message")
            _state.value = State.ProcessingUserPrompt

            // Simulate longer processing time
            delay(STUB_USER_PROMPT_PROCESSING_TIME)

            _state.value = State.Generating

            // Simulate token generation
            val response = "This is a simulated response from the LLM model. The actual implementation would generate tokens one by one based on the input: $message"
            response.split(" ").forEach {
                emit("$it ")
                delay(STUB_TOKEN_GENERATION_TIME)
            }

            _state.value = State.ModelReady
        } catch (e: CancellationException) {
            // Handle cancellation gracefully
            _state.value = State.ModelReady
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

    /**
     * Runs a benchmark with the specified parameters.
     */
    override suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int): String =
        withContext(llamaDispatcher) {
            check(_state.value is State.ModelReady) {
                "Cannot load model at ${_state.value.javaClass.simpleName}"
            }

            try {
                Log.i(TAG, "bench! state: ${_state.value}")
                _state.value = State.Benchmarking

                // Simulate benchmark running
                delay(STUB_BENCHMARKING_TIME)

                // Generate fake benchmark results
                val modelDesc = APP_NAME
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

                _state.value = State.ModelReady

                result.toString()
            } catch (e: CancellationException) {
                // If coroutine is cancelled, propagate cancellation
                Log.w(TAG, "Unexpected user cancellation while benchmarking!")
                _state.value = State.ModelReady
                throw e
            } catch (e: Exception) {
                _state.value = State.Error(e.message ?: "Unknown error during benchmarking")
                "Error: ${e.message}"
            }
        }

    /**
     * Unloads the currently loaded model.
     */
    override suspend fun cleanUp() =
        withContext(llamaDispatcher) {
            when(val state = _state.value) {
                is State.ModelReady, is State.Error -> {
                    Log.i(TAG, "unloadModel! state: ${_state.value.javaClass.simpleName}")
                    _state.value = State.UnloadingModel

                    // Simulate model unloading time
                    delay(STUB_MODEL_UNLOADING_TIME)

                    _state.value = State.Initialized
                }
                else -> throw IllegalStateException(
                    "Cannot load model at ${_state.value.javaClass.simpleName}"
                )
            }
        }

    /**
     * Cleans up resources when the engine is no longer needed.
     */
    override fun destroy() {
        Log.i(TAG, "destroy! state: ${_state.value}")

        _state.value = State.Uninitialized
    }
}
