package android.llama.cpp

import android.util.Log
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import kotlin.concurrent.thread

class LLamaAndroid {
    /**
     * JNI methods
     * @see llama-android.cpp
     */
    private external fun systemInfo(): String

    private external fun loadModel(filename: String): Int
    private external fun initContext(): Int
    private external fun cleanUp()

    private external fun benchModel(pp: Int, tg: Int, pl: Int, nr: Int): String

    private external fun processSystemPrompt(systemPrompt: String): Int
    private external fun processUserPrompt(userPrompt: String, nPredict: Int): Int
    private external fun predictLoop(): String?

    /**
     * Thread local state
     */
    private sealed interface State {
        data object NotInitialized: State
        data object EnvReady: State
        data object AwaitingUserPrompt: State
        data object Processing: State
    }
    private val threadLocalState: ThreadLocal<State> = ThreadLocal.withInitial { State.NotInitialized }

    private val runLoop: CoroutineDispatcher = Executors.newSingleThreadExecutor {
        thread(start = false, name = LLAMA_THREAD) {
            Log.d(TAG, "Dedicated thread for native code: ${Thread.currentThread().name}")

            // No-op if called more than once.
            System.loadLibrary(LIB_LLAMA_ANDROID)
            Log.d(TAG, systemInfo())

            it.run()
        }.apply {
            uncaughtExceptionHandler = Thread.UncaughtExceptionHandler { _, exception: Throwable ->
                Log.e(TAG, "Unhandled exception", exception)
            }
        }
    }.asCoroutineDispatcher()

    /**
     * Load the LLM, then process the formatted system prompt if provided
     */
    suspend fun load(pathToModel: String, systemPrompt: String? = null) =
        withContext(runLoop) {
            when (threadLocalState.get()) {
                is State.NotInitialized -> {
                    val modelResult = loadModel(pathToModel)
                    if (modelResult != 0)  throw IllegalStateException("Load model failed: $modelResult")

                    val initResult = initContext()
                    if (initResult != 0) throw IllegalStateException("Initialization failed with error code: $initResult")

                    Log.i(TAG, "Loaded model $pathToModel")
                    threadLocalState.set(State.EnvReady)

                    systemPrompt?.let {
                        initWithSystemPrompt(systemPrompt)
                    } ?: run {
                        Log.w(TAG, "No system prompt to process.")
                        threadLocalState.set(State.AwaitingUserPrompt)
                    }
                }
                else -> throw IllegalStateException("Model already loaded")
            }
        }

    /**
     * Helper method to process system prompt and update [State]
     */
    private suspend fun initWithSystemPrompt(formattedMessage: String) =
        withContext(runLoop) {
            when (threadLocalState.get()) {
                is State.EnvReady -> {
                    Log.i(TAG, "Process system prompt...")
                    threadLocalState.set(State.Processing)
                    processSystemPrompt(formattedMessage).let {
                        if (it != 0)
                            throw IllegalStateException("Failed to process system prompt: $it")
                    }

                    Log.i(TAG, "System prompt processed!")
                    threadLocalState.set(State.AwaitingUserPrompt)
                }
                else -> throw IllegalStateException(
                    "Failed to process system prompt: Model not loaded!"
                )
            }
        }

    /**
     * Send formatted user prompt to LLM
     */
    fun sendUserPrompt(
        message: String,
        predictLength: Int = DEFAULT_PREDICT_LENGTH,
    ): Flow<String> = flow {
        require(message.isNotEmpty()) {
            Log.w(TAG, "User prompt discarded due to being empty!")
        }

        when (val state = threadLocalState.get()) {
            is State.AwaitingUserPrompt -> {
                Log.i(TAG, "Sending user prompt...")
                threadLocalState.set(State.Processing)
                processUserPrompt(message, predictLength).let { result ->
                    if (result != 0) {
                        Log.e(TAG, "Failed to process user prompt: $result")
                        return@flow
                    }
                }

                Log.i(TAG, "User prompt processed! Generating assistant prompt...")
                while (true) {
                    predictLoop()?.let { utf8token ->
                        if (utf8token.isNotEmpty()) emit(utf8token)
                    } ?: break
                }

                Log.i(TAG, "Assistant generation complete!")
                threadLocalState.set(State.AwaitingUserPrompt)
            }
            else -> {
                Log.w(TAG, "User prompt discarded due to incorrect state: $state")
            }
        }
    }.flowOn(runLoop)

    /**
     * Benchmark the model
     */
    suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1): String =
        withContext(runLoop) {
            when (threadLocalState.get()) {
                is State.AwaitingUserPrompt -> {
                    threadLocalState.set(State.Processing)
                    Log.d(TAG, "Start benchmark (pp: $pp, tg: $tg, pl: $pl, nr: $nr)")
                    benchModel(pp, tg, pl, nr).also {
                        threadLocalState.set(State.AwaitingUserPrompt)
                    }
                }

                // TODO-hyin: disable button when state incorrect
                else -> throw IllegalStateException("No model loaded")
            }
        }

    /**
     * Unloads the model and frees resources.
     *
     * This is a no-op if there's no model loaded.
     */
    suspend fun unload() =
        withContext(runLoop) {
            when (val state = threadLocalState.get()) {
                is State.EnvReady, State.AwaitingUserPrompt -> {
                    cleanUp()
                    threadLocalState.set(State.NotInitialized)
                }
                else -> {
                    Log.w(TAG, "Cannot unload model due to incorrect state: $state")
                }
            }
        }

    companion object {
        private val TAG = LLamaAndroid::class.simpleName

        private const val LIB_LLAMA_ANDROID = "llama-android"
        private const val LLAMA_THREAD = "llama-thread"

        private const val DEFAULT_PREDICT_LENGTH = 64

        // Enforce only one instance of Llm.
        private val _instance: LLamaAndroid = LLamaAndroid()
        fun instance(): LLamaAndroid = _instance
    }
}
