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
    private external fun log_to_android()
    private external fun system_info(): String
    private external fun backend_init()

    private external fun load_model(filename: String): Int
    private external fun ctx_init(): Int
    private external fun clean_up()

    private external fun bench_model(pp: Int, tg: Int, pl: Int, nr: Int): String

    private external fun process_system_prompt(system_prompt: String): Int
    private external fun process_user_prompt(user_prompt: String, nLen: Int): Int
    private external fun predict_loop(): String?

    /**
     * Thread local state
     */
    private sealed interface State {
        data object Idle: State
        data object ModelLoaded: State
        data object ReadyForUserPrompt: State
    }
    private val threadLocalState: ThreadLocal<State> = ThreadLocal.withInitial { State.Idle }

    private val runLoop: CoroutineDispatcher = Executors.newSingleThreadExecutor {
        thread(start = false, name = "Llm-RunLoop") {
            Log.d(TAG, "Dedicated thread for native code: ${Thread.currentThread().name}")

            // No-op if called more than once.
            System.loadLibrary("llama-android")

            // Set llama log handler to Android
            log_to_android()
            backend_init()

            Log.d(TAG, system_info())

            it.run()
        }.apply {
            uncaughtExceptionHandler = Thread.UncaughtExceptionHandler { _, exception: Throwable ->
                Log.e(TAG, "Unhandled exception", exception)
            }
        }
    }.asCoroutineDispatcher()

    /**
     * Load the LLM, then process the system prompt if provided
     */
    suspend fun load(pathToModel: String, formattedSystemPrompt: String? = null) {
        withContext(runLoop) {
            when (threadLocalState.get()) {
                is State.Idle -> {
                    val model = load_model(pathToModel)
                    if (model != 0)  throw IllegalStateException("Load model failed")

                    val result = ctx_init()
                    if (result != 0) throw IllegalStateException("Initialization failed with error code: $result")

                    Log.i(TAG, "Loaded model $pathToModel")
                    threadLocalState.set(State.ModelLoaded)

                    formattedSystemPrompt?.let {
                        initWithSystemPrompt(formattedSystemPrompt)
                    } ?: {
                        Log.w(TAG, "No system prompt to process.")
                        threadLocalState.set(State.ReadyForUserPrompt)
                    }
                }
                else -> throw IllegalStateException("Model already loaded")
            }
        }
    }

    /**
     * Helper method to process system prompt and update [State]
     */
    private suspend fun initWithSystemPrompt(formattedMessage: String) {
        withContext(runLoop) {
            when (threadLocalState.get()) {
                is State.ModelLoaded -> {
                    Log.i(TAG, "Process system prompt...")
                    process_system_prompt(formattedMessage).let {
                        if (it != 0)
                            throw IllegalStateException("Failed to process system prompt: $it")
                    }

                    Log.i(TAG, "System prompt processed!")
                    threadLocalState.set(State.ReadyForUserPrompt)
                }
                else -> throw IllegalStateException(
                    "Failed to process system prompt: Model not loaded!"
                )
            }
        }
    }

    /**
     * Send plain text user prompt to LLM
     */
    fun sendUserPrompt(
        formattedMessage: String,
        nPredict: Int = DEFAULT_PREDICT_LENGTH,
    ): Flow<String> = flow {
        when (threadLocalState.get()) {
            is State.ReadyForUserPrompt -> {
                process_user_prompt(formattedMessage, nPredict).let {
                    if (it != 0) {
                        Log.e(TAG, "Failed to process user prompt: $it")
                        return@flow
                    }
                }

                Log.i(TAG, "User prompt processed! Generating assistant prompt...")
                while (true) {
                    val str = predict_loop() ?: break
                    if (str.isNotEmpty()) {
                        emit(str)
                    }
                }
                Log.i(TAG, "Assistant generation complete!")
            }
            else -> {}
        }
    }.flowOn(runLoop)

    /**
     * Benchmark the model
     */
    suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1): String {
        return withContext(runLoop) {
            when (val state = threadLocalState.get()) {
                is State.ModelLoaded -> {
                    Log.d(TAG, "bench(): $state")
                    bench_model(pp, tg, pl, nr)
                }

                // TODO-hyin: catch exception in ViewController; disable button when state incorrect
                else -> throw IllegalStateException("No model loaded")
            }
        }
    }

    /**
     * Unloads the model and frees resources.
     *
     * This is a no-op if there's no model loaded.
     */
    suspend fun unload() {
        withContext(runLoop) {
            when (threadLocalState.get()) {
                is State.ModelLoaded -> {
                    clean_up()
                    threadLocalState.set(State.Idle)
                }
                else -> {}
            }
        }
    }

    companion object {
        private val TAG = this::class.simpleName

        private const val DEFAULT_PREDICT_LENGTH = 128

        // Enforce only one instance of Llm.
        private val _instance: LLamaAndroid = LLamaAndroid()
        fun instance(): LLamaAndroid = _instance
    }
}
