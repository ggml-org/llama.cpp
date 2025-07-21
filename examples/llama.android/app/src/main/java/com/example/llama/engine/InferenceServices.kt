package com.example.llama.engine

import android.llama.cpp.InferenceEngine
import android.llama.cpp.InferenceEngine.State
import android.util.Log
import com.example.llama.data.model.ModelInfo
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flow
import javax.inject.Inject
import javax.inject.Singleton

interface InferenceService {
    /**
     * Expose engine state
     */
    val engineState: StateFlow<State>

    /**
     * Currently selected model
     */
    val currentSelectedModel: StateFlow<ModelInfo?>

    /**
     * Set current model
     */
    fun setCurrentModel(model: ModelInfo)

    /**
     * Unload current model and free resources
     */
    suspend fun unloadModel()
}

interface ModelLoadingService : InferenceService {
    /**
     * Load a model for benchmark
     */
    suspend fun loadModelForBenchmark(): ModelLoadingMetrics?

    /**
     * Load a model for conversation
     */
    suspend fun loadModelForConversation(systemPrompt: String?): ModelLoadingMetrics?
}

interface BenchmarkService : InferenceService {
    /**
     * Run benchmark
     *
     * @param pp: Prompt Processing size
     * @param tg: Token Generation size
     * @param pl: Parallel sequences
     * @param nr: Number of Runs, i.e. repetitions
     */
    suspend fun benchmark(pp: Int, tg: Int, pl: Int, nr: Int): String

    /**
     * Benchmark results
     */
    val benchmarkResults: StateFlow<String?>
}

interface ConversationService : InferenceService {
    /**
     * System prompt
     */
    val systemPrompt: StateFlow<String?>

    /**
     * Generate response from prompt
     */
    fun generateResponse(prompt: String): Flow<GenerationUpdate>

    /**
     * Create token metrics based on current state
     */
    fun createTokenMetrics(): TokenMetrics
}

/**
 * Metrics for model loading and system prompt processing
 */
data class ModelLoadingMetrics(
    val modelLoadingTimeMs: Long,
    val systemPromptProcessingTimeMs: Long? = null
) {
    val totalTimeMs: Long
        get() = modelLoadingTimeMs + (systemPromptProcessingTimeMs ?: 0)
}

/**
 * Represents an update during text generation
 */
data class GenerationUpdate(
    val text: String,
    val isComplete: Boolean,
    val metrics: TokenMetrics? = null
)

/**
 * Metrics for token generation performance
 */
data class TokenMetrics(
    val tokensCount: Int,
    val ttftMs: Long,
    val tpsMs: Float,
) {
    val text: String
        get() = "Tokens: $tokensCount, TTFT: ${ttftMs}ms, TPS: ${"%.1f".format(tpsMs)}"
}

/**
 * Internal implementation of the above [InferenceService]s
 */
@Singleton
internal class InferenceServiceImpl @Inject internal constructor(
    private val inferenceEngine: InferenceEngine
) : ModelLoadingService, BenchmarkService, ConversationService {

    /*
     *
     * InferenceService implementation
     *
     */

    override val engineState: StateFlow<State> = inferenceEngine.state

    private val _currentModel = MutableStateFlow<ModelInfo?>(null)
    override val currentSelectedModel: StateFlow<ModelInfo?> = _currentModel.asStateFlow()

    override fun setCurrentModel(model: ModelInfo) { _currentModel.value = model }

    override suspend fun unloadModel() = inferenceEngine.cleanUp()

    /**
     * Shut down inference engine
     */
    fun destroy() = inferenceEngine.destroy()

    /*
     *
     * ModelLoadingService implementation
     *
     */

    override suspend fun loadModelForBenchmark(): ModelLoadingMetrics? {
        checkNotNull(_currentModel.value) { "Attempt to load model for bench while none selected!" }

        return _currentModel.value?.let { model ->
            try {
                val modelLoadStartTs = System.currentTimeMillis()
                inferenceEngine.loadModel(model.path)
                val modelLoadEndTs = System.currentTimeMillis()
                ModelLoadingMetrics(modelLoadEndTs - modelLoadStartTs)
            } catch (e: Exception) {
                Log.e(TAG, "Error loading model", e)
                null
            }
        }
    }

    override suspend fun loadModelForConversation(systemPrompt: String?): ModelLoadingMetrics? {
        checkNotNull(_currentModel.value) { "Attempt to load model for chat while none selected!" }

        return _currentModel.value?.let { model ->
            try {
                _systemPrompt.value = systemPrompt

                val modelLoadStartTs = System.currentTimeMillis()
                inferenceEngine.loadModel(model.path)
                val modelLoadEndTs = System.currentTimeMillis()

                if (systemPrompt.isNullOrBlank()) {
                    ModelLoadingMetrics(modelLoadEndTs - modelLoadStartTs)
                } else {
                    val prompt: String = systemPrompt
                    val systemPromptStartTs = System.currentTimeMillis()
                    inferenceEngine.setSystemPrompt(prompt)
                    val systemPromptEndTs = System.currentTimeMillis()
                    ModelLoadingMetrics(
                        modelLoadingTimeMs = modelLoadEndTs - modelLoadStartTs,
                        systemPromptProcessingTimeMs = systemPromptEndTs - systemPromptStartTs
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error loading model", e)
                null
            }
        }
    }

    /*
     *
     * BenchmarkService implementation
     *
     */

    override suspend fun benchmark(pp: Int, tg: Int, pl: Int, nr: Int): String =
        inferenceEngine.bench(pp, tg, pl, nr).also {
            _benchmarkResults.value = it
        }

    /**
     * Benchmark results if available
     */
    private val _benchmarkResults = MutableStateFlow<String?>(null)
    override val benchmarkResults: StateFlow<String?> = _benchmarkResults


    /* ConversationService implementation */

    private val _systemPrompt = MutableStateFlow<String?>(null)
    override val systemPrompt: StateFlow<String?> = _systemPrompt.asStateFlow()

    // Token metrics tracking
    private var generationStartTime: Long = 0L
    private var firstTokenTime: Long = 0L
    private var tokenCount: Int = 0
    private var isFirstToken: Boolean = true

    override fun generateResponse(prompt: String): Flow<GenerationUpdate> = flow {
        val response = StringBuilder()

        try {
            // Reset metrics tracking
            generationStartTime = System.currentTimeMillis()
            firstTokenTime = 0L
            tokenCount = 0
            isFirstToken = true

            inferenceEngine.sendUserPrompt(prompt)
                .collect { token ->
                    // Track first token time
                    if (isFirstToken && token.isNotBlank()) {
                        firstTokenTime = System.currentTimeMillis()
                        isFirstToken = false
                    }

                    // Count tokens
                    if (token.isNotBlank()) {
                        tokenCount++
                    }

                    response.append(token)

                    // Emit ongoing response (not completed)
                    emit(GenerationUpdate(response.toString(), false))
                }

            // Calculate final metrics after completion
            val metrics = createTokenMetrics()

            // Emit final response with completion flag
            emit(GenerationUpdate(response.toString(), true, metrics))
        } catch (e: Exception) {
            // Emit error
            val metrics = createTokenMetrics()
            emit(GenerationUpdate(response.toString(), true, metrics))
            throw e
        }
    }

    override fun createTokenMetrics(): TokenMetrics {
        val endTime = System.currentTimeMillis()
        val totalTimeMs = endTime - generationStartTime

        return TokenMetrics(
            tokensCount = tokenCount,
            ttftMs = if (firstTokenTime > 0) firstTokenTime - generationStartTime else 0L,
            tpsMs = calculateTPS(tokenCount, totalTimeMs)
        )
    }

    /**
     * Calculate tokens per second
     */
    private fun calculateTPS(tokens: Int, timeMs: Long): Float {
        if (tokens <= 0 || timeMs <= 0) return 0f
        return (tokens.toFloat() * 1000f) / timeMs
    }

    companion object {
        private val TAG = InferenceServiceImpl::class.java.simpleName
    }
}
