package android.llama.cpp.internal

import android.llama.cpp.InferenceEngine
import android.llama.cpp.LLamaTier
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
internal class InferenceEngineImpl private constructor(
    private val tier: LLamaTier
) : InferenceEngine {

    companion object {
        private val TAG = InferenceEngineImpl::class.java.simpleName

        private var initialized = false

        /**
         * Create [InferenceEngineImpl] instance with specific tier
         */
        internal fun createWithTier(tier: LLamaTier): InferenceEngineImpl? {
            if (initialized) {
                Log.w(TAG, "LLamaAndroid already initialized")
                return null
            }

            try {
                Log.i(TAG, "Instantiating InferenceEngineImpl w/ ${tier.libraryName}")
                val instance = InferenceEngineImpl(tier)
                initialized = true
                return instance

            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load ${tier.libraryName}", e)
                return null
            }
        }
    }

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

    private val _state =
        MutableStateFlow<InferenceEngine.State>(InferenceEngine.State.Uninitialized)
    override val state: StateFlow<InferenceEngine.State> = _state

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
                check(_state.value is InferenceEngine.State.Uninitialized) {
                    "Cannot load native library in ${_state.value.javaClass.simpleName}!"
                }
                _state.value = InferenceEngine.State.Initializing
                Log.i(TAG, "Loading native library for $tier")

                System.loadLibrary(tier.libraryName)
                init()
                _state.value = InferenceEngine.State.Initialized
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
            check(_state.value is InferenceEngine.State.Initialized) {
                "Cannot load model in ${_state.value.javaClass.simpleName}!"
            }
            File(pathToModel).let {
                require(it.exists()) { "Model file not found: $pathToModel" }
                require(it.isFile) { "Model file is not a file: $pathToModel" }
            }

            Log.i(TAG, "Loading model... \n$pathToModel")
            _readyForSystemPrompt = false
            _state.value = InferenceEngine.State.LoadingModel
            load(pathToModel).let { result ->
                if (result != 0) throw IllegalStateException("Failed to Load model: $result")
            }
            prepare().let { result ->
                if (result != 0) throw IllegalStateException("Failed to prepare resources: $result")
            }
            Log.i(TAG, "Model loaded!")
            _readyForSystemPrompt = true
            _state.value = InferenceEngine.State.ModelReady
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
            check(_state.value is InferenceEngine.State.ModelReady) {
                "Cannot process system prompt in ${_state.value.javaClass.simpleName}!"
            }

            Log.i(TAG, "Sending system prompt...")
            _readyForSystemPrompt = false
            _state.value = InferenceEngine.State.ProcessingSystemPrompt
            processSystemPrompt(prompt).let { result ->
                if (result != 0) {
                    val errorMessage = "Failed to process system prompt: $result"
                    _state.value = InferenceEngine.State.Error(errorMessage)
                    throw IllegalStateException(errorMessage)
                }
            }
            Log.i(TAG, "System prompt processed! Awaiting user prompt...")
            _state.value = InferenceEngine.State.ModelReady
        }

    /**
     * Send plain text user prompt to LLM, which starts generating tokens in a [kotlinx.coroutines.flow.Flow]
     */
    override fun sendUserPrompt(
        message: String,
        predictLength: Int,
    ): Flow<String> = flow {
        require(message.isNotEmpty()) { "User prompt discarded due to being empty!" }
        check(_state.value is InferenceEngine.State.ModelReady) {
            "User prompt discarded due to: ${_state.value.javaClass.simpleName}"
        }

        try {
            Log.i(TAG, "Sending user prompt...")
            _readyForSystemPrompt = false
            _state.value = InferenceEngine.State.ProcessingUserPrompt

            processUserPrompt(message, predictLength).let { result ->
                if (result != 0) {
                    Log.e(TAG, "Failed to process user prompt: $result")
                    return@flow
                }
            }

            Log.i(TAG, "User prompt processed. Generating assistant prompt...")
            _state.value = InferenceEngine.State.Generating
            while (true) {
                generateNextToken()?.let { utf8token ->
                    if (utf8token.isNotEmpty()) emit(utf8token)
                } ?: break
            }
            Log.i(TAG, "Assistant generation complete. Awaiting user prompt...")
            _state.value = InferenceEngine.State.ModelReady
        } catch (e: CancellationException) {
            Log.i(TAG, "Generation cancelled by user.")
            _state.value = InferenceEngine.State.ModelReady
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Error during generation!", e)
            _state.value = InferenceEngine.State.Error(e.message ?: "Unknown error")
            throw e
        }
    }.flowOn(llamaDispatcher)

    /**
     * Benchmark the model
     */
    override suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int): String =
        withContext(llamaDispatcher) {
            check(_state.value is InferenceEngine.State.ModelReady) {
                "Benchmark request discarded due to: $state"
            }
            Log.i(TAG, "Start benchmark (pp: $pp, tg: $tg, pl: $pl, nr: $nr)")
            _readyForSystemPrompt = false   // Just to be safe
            _state.value = InferenceEngine.State.Benchmarking
            benchModel(pp, tg, pl, nr).also {
                _state.value = InferenceEngine.State.ModelReady
            }
        }

    /**
     * Unloads the model and frees resources
     */
    override suspend fun unloadModel() =
        withContext(llamaDispatcher) {
            when (val state = _state.value) {
                is InferenceEngine.State.ModelReady, is InferenceEngine.State.Error -> {
                    Log.i(TAG, "Unloading model and free resources...")
                    _readyForSystemPrompt = false
                    _state.value = InferenceEngine.State.UnloadingModel

                    unload()

                    _state.value = InferenceEngine.State.Initialized
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
            is InferenceEngine.State.Uninitialized -> {}
            is InferenceEngine.State.Initialized -> shutdown()
            else -> { unload(); shutdown() }
        }
    }
}
