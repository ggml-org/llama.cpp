package android.llama.cpp

import android.llama.cpp.InferenceEngine.State
import android.llama.cpp.LLamaAndroid.Companion.instance
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * JNI wrapper for the llama.cpp library providing Android-friendly access to large language models.
 *
 * This class implements a singleton pattern for managing the lifecycle of a single LLM instance.
 * All operations are executed on a dedicated single-threaded dispatcher to ensure thread safety
 * with the underlying C++ native code.
 *
 * The typical usage flow is:
 * 1. Get instance via [instance]
 * 2. Load a model with [loadModel]
 * 3. Send prompts with [sendUserPrompt]
 * 4. Generate responses as token streams
 * 5. Unload the model with [unloadModel] when switching models
 * 6. Call [destroy] when completely done
 *
 * State transitions are managed automatically and validated at each operation.
 *
 * @see llama-android.cpp for the native implementation details
 */
class LLamaAndroid private constructor() : InferenceEngine {
    /**
     * JNI methods
     * @see llama-android.cpp
     */
    private external fun init()
    private external fun load(modelPath: String): Int
    private external fun prepare(): Int

    private external fun systemInfo(): String
    private external fun benchModel(pp: Int, tg: Int, pl: Int, nr: Int): String

    private external fun processSystemPrompt(systemPrompt: String): Int
    private external fun processUserPrompt(userPrompt: String, predictLength: Int): Int
    private external fun generateNextToken(): String?

    private external fun unload()
    private external fun shutdown()

    private val _state = MutableStateFlow<State>(State.Uninitialized)
    override val state: StateFlow<State> = _state

    private var _readyForSystemPrompt = false

    /**
     * Single-threaded coroutine dispatcher & scope for LLama asynchronous operations
     */
    @OptIn(ExperimentalCoroutinesApi::class)
    private val llamaDispatcher = Dispatchers.IO.limitedParallelism(1)
    private val llamaScope = CoroutineScope(llamaDispatcher + SupervisorJob())

    init {
        llamaScope.launch {
            try {
                check(_state.value is State.Uninitialized) {
                    "Cannot load native library in ${_state.value.javaClass.simpleName}!"
                }

                _state.value = State.Initializing
                System.loadLibrary(LIB_LLAMA_ANDROID)
                init()
                _state.value = State.Initialized
                Log.i(TAG, "Native library loaded! System info: \n${systemInfo()}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load native library", e)
                throw e
            }
        }
    }

    /**
     * Load the LLM
     */
    override suspend fun loadModel(pathToModel: String) =
        withContext(llamaDispatcher) {
            check(_state.value is State.Initialized) {
                "Cannot load model in ${_state.value.javaClass.simpleName}!"
            }
            File(pathToModel).let {
                require(it.exists()) { "Model file not found: $pathToModel" }
                require(it.isFile) { "Model file is not a file: $pathToModel" }
            }

            Log.i(TAG, "Loading model... \n$pathToModel")
            _readyForSystemPrompt = false
            _state.value = State.LoadingModel
            load(pathToModel).let { result ->
                if (result != 0) throw IllegalStateException("Failed to Load model: $result")
            }
            prepare().let { result ->
                if (result != 0) throw IllegalStateException("Failed to prepare resources: $result")
            }
            Log.i(TAG, "Model loaded!")
            _readyForSystemPrompt = true
            _state.value = State.ModelReady
        }

    /**
     * Process the plain text system prompt
     *
     * TODO-han.yin: return error code if system prompt not correct processed?
     */
    override suspend fun setSystemPrompt(prompt: String) =
        withContext(llamaDispatcher) {
            require(prompt.isNotBlank()) { "Cannot process empty system prompt!" }
            check(_readyForSystemPrompt) { "System prompt must be set ** RIGHT AFTER ** model loaded!" }
            check(_state.value is State.ModelReady) {
                "Cannot process system prompt in ${_state.value.javaClass.simpleName}!"
            }

            Log.i(TAG, "Sending system prompt...")
            _readyForSystemPrompt = false
            _state.value = State.ProcessingSystemPrompt
            processSystemPrompt(prompt).let { result ->
                if (result != 0) {
                    val errorMessage = "Failed to process system prompt: $result"
                    _state.value = State.Error(errorMessage)
                    throw IllegalStateException(errorMessage)
                }
            }
            Log.i(TAG, "System prompt processed! Awaiting user prompt...")
            _state.value = State.ModelReady
        }

    /**
     * Send plain text user prompt to LLM, which starts generating tokens in a [Flow]
     */
    override fun sendUserPrompt(
        message: String,
        predictLength: Int,
    ): Flow<String> = flow {
        require(message.isNotEmpty()) { "User prompt discarded due to being empty!" }
        check(_state.value is State.ModelReady) {
            "User prompt discarded due to: ${_state.value.javaClass.simpleName}"
        }

        try {
            Log.i(TAG, "Sending user prompt...")
            _readyForSystemPrompt = false
            _state.value = State.ProcessingUserPrompt

            processUserPrompt(message, predictLength).let { result ->
                if (result != 0) {
                    Log.e(TAG, "Failed to process user prompt: $result")
                    return@flow
                }
            }

            Log.i(TAG, "User prompt processed. Generating assistant prompt...")
            _state.value = State.Generating
            while (true) {
                generateNextToken()?.let { utf8token ->
                    if (utf8token.isNotEmpty()) emit(utf8token)
                } ?: break
            }
            Log.i(TAG, "Assistant generation complete. Awaiting user prompt...")
            _state.value = State.ModelReady
        } catch (e: CancellationException) {
            Log.i(TAG, "Generation cancelled by user.")
            _state.value = State.ModelReady
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Error during generation!", e)
            _state.value = State.Error(e.message ?: "Unknown error")
            throw e
        }
    }.flowOn(llamaDispatcher)

    /**
     * Benchmark the model
     */
    override suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int): String =
        withContext(llamaDispatcher) {
            check(_state.value is State.ModelReady) {
                "Benchmark request discarded due to: $state"
            }
            Log.i(TAG, "Start benchmark (pp: $pp, tg: $tg, pl: $pl, nr: $nr)")
            _readyForSystemPrompt = false   // Just to be safe
            _state.value = State.Benchmarking
            benchModel(pp, tg, pl, nr).also {
                _state.value = State.ModelReady
            }
        }

    /**
     * Unloads the model and frees resources
     */
    override suspend fun unloadModel() =
        withContext(llamaDispatcher) {
            when(val state = _state.value) {
                is State.ModelReady, is State.Error -> {
                    Log.i(TAG, "Unloading model and free resources...")
                    _readyForSystemPrompt = false
                    _state.value = State.UnloadingModel

                    unload()

                    _state.value = State.Initialized
                    Log.i(TAG, "Model unloaded!")
                    Unit
                }
                else -> throw IllegalStateException("Cannot unload model in ${state.javaClass.simpleName}")
            }
        }

    /**
     * Cancel all ongoing coroutines and free GGML backends
     */
    override fun destroy() {
        _readyForSystemPrompt = false
        llamaScope.cancel()
        when(_state.value) {
            is State.Uninitialized -> {}
            is State.Initialized -> shutdown()
            else -> { unload(); shutdown() }
        }
    }

    companion object {
        private val TAG = LLamaAndroid::class.simpleName

        private const val LIB_LLAMA_ANDROID = "llama-android"
        private const val DEFAULT_PREDICT_LENGTH = 64

        // Enforce only one instance of Llm.
        private val _instance: LLamaAndroid = LLamaAndroid()
        fun instance(): LLamaAndroid = _instance
    }
}
