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
    private val tag: String? = this::class.simpleName

    private val threadLocalState: ThreadLocal<State> = ThreadLocal.withInitial { State.Idle }

    private val runLoop: CoroutineDispatcher = Executors.newSingleThreadExecutor {
        thread(start = false, name = "Llm-RunLoop") {
            Log.d(tag, "Dedicated thread for native code: ${Thread.currentThread().name}")

            // No-op if called more than once.
            System.loadLibrary("llama-android")

            // Set llama log handler to Android
            log_to_android()
            backend_init()

            Log.d(tag, system_info())

            it.run()
        }.apply {
            uncaughtExceptionHandler = Thread.UncaughtExceptionHandler { _, exception: Throwable ->
                Log.e(tag, "Unhandled exception", exception)
            }
        }
    }.asCoroutineDispatcher()

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

    suspend fun load(pathToModel: String, formattedSystemPrompt: String? = null) {
        withContext(runLoop) {
            when (threadLocalState.get()) {
                is State.Idle -> {
                    val model = load_model(pathToModel)
                    if (model != 0)  throw IllegalStateException("Load model failed")

                    val result = ctx_init()
                    if (result != 0) throw IllegalStateException("Initialization failed with error code: $result")

                    Log.i(tag, "Loaded model $pathToModel")
                    threadLocalState.set(State.ModelLoaded)

                    formattedSystemPrompt?.let {
                        initWithSystemPrompt(formattedSystemPrompt)
                    } ?: {
                        threadLocalState.set(State.ReadyForUserPrompt)
                    }
                }
                else -> throw IllegalStateException("Model already loaded")
            }
        }
    }

    suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1): String {
        return withContext(runLoop) {
            when (val state = threadLocalState.get()) {
                is State.ModelLoaded -> {
                    Log.d(tag, "bench(): $state")
                    bench_model(pp, tg, pl, nr)
                }

                // TODO-hyin: catch exception in ViewController; disable button when state incorrect
                else -> throw IllegalStateException("No model loaded")
            }
        }
    }

    private suspend fun initWithSystemPrompt(systemPrompt: String) {
        withContext(runLoop) {
            when (threadLocalState.get()) {
                is State.ModelLoaded -> {
                    process_system_prompt(systemPrompt).let {
                        if (it != 0) {
                            throw IllegalStateException("Failed to process system prompt: $it")
                        }
                    }

                    Log.i(tag, "System prompt processed!")
                    threadLocalState.set(State.ReadyForUserPrompt)
                }
                else -> throw IllegalStateException("Model not loaded")
            }
        }
    }

    fun sendMessage(
        formattedUserPrompt: String,
        nPredict: Int = DEFAULT_PREDICT_LENGTH,
    ): Flow<String> = flow {
        when (threadLocalState.get()) {
            is State.ReadyForUserPrompt -> {
                process_user_prompt(formattedUserPrompt, nPredict).let {
                    if (it != 0) {
                        Log.e(tag, "Failed to process user prompt: $it")
                        return@flow
                    }
                }

                Log.i(tag, "User prompt processed! Generating assistant prompt...")
                while (true) {
                    val str = predict_loop() ?: break
                    if (str.isNotEmpty()) {
                        emit(str)
                    }
                }
                Log.i(tag, "Assistant generation complete!")
            }
            else -> {}
        }
    }.flowOn(runLoop)

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
        private const val DEFAULT_PREDICT_LENGTH = 128

        private sealed interface State {
            data object Idle: State
            data object ModelLoaded: State
            data object ReadyForUserPrompt: State
        }

        // Enforce only one instance of Llm.
        private val _instance: LLamaAndroid = LLamaAndroid()

        fun instance(): LLamaAndroid = _instance
    }
}
