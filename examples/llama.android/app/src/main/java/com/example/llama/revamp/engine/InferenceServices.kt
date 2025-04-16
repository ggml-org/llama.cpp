package com.example.llama.revamp.engine

import android.util.Log
import com.example.llama.revamp.data.model.ModelInfo
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
    val engineState: StateFlow<InferenceEngine.State>

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
    suspend fun loadModelForBenchmark(): Boolean

    /**
     * Load a model for conversation
     */
    suspend fun loadModelForConversation(systemPrompt: String?): Boolean
}

interface BenchmarkService : InferenceService {
    /**
     * Run benchmark
     *
     * @param pp: Prompt Processing size
     * @param tg: Token Generation size
     * @param pl: Parallel sequences
     * @param nr: repetitions (Number of Runs)
     */
    suspend fun benchmark(pp: Int, tg: Int, pl: Int, nr: Int): String

    /**
     * Benchmark results
     */
    val results: StateFlow<String?>
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
 * Represents an update during text generation
 */
data class GenerationUpdate(
    val text: String,
    val isComplete: Boolean
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

    /* InferenceService implementation */

    override val engineState: StateFlow<InferenceEngine.State> = inferenceEngine.state

    private val _currentModel = MutableStateFlow<ModelInfo?>(null)
    override val currentSelectedModel: StateFlow<ModelInfo?> = _currentModel.asStateFlow()

    override fun setCurrentModel(model: ModelInfo) {
        _currentModel.value = model
    }

    override suspend fun unloadModel() = inferenceEngine.unloadModel()

    /**
     * Shut down inference engine
     */
    fun destroy() = inferenceEngine.destroy()


    /* ModelLoadingService implementation */

    override suspend fun loadModelForBenchmark(): Boolean {
        return _currentModel.value?.let { model ->
            try {
                inferenceEngine.loadModel(model.path)
                true
            } catch (e: Exception) {
                Log.e("InferenceManager", "Error loading model", e)
                false
            }
        } == true
    }

    override suspend fun loadModelForConversation(systemPrompt: String?): Boolean {
        _systemPrompt.value = systemPrompt
        return _currentModel.value?.let { model ->
            try {
                inferenceEngine.loadModel(model.path, systemPrompt)
                true
            } catch (e: Exception) {
                Log.e("InferenceManager", "Error loading model", e)
                false
            }
        } == true
    }


    /* BenchmarkService implementation */

    override suspend fun benchmark(pp: Int, tg: Int, pl: Int, nr: Int): String =
        inferenceEngine.bench(pp, tg, pl, nr)

    override val results: StateFlow<String?> = inferenceEngine.benchmarkResults


    /* ConversationService implementation */

    private val _systemPrompt = MutableStateFlow<String?>(null)
    override val systemPrompt: StateFlow<String?> = _systemPrompt.asStateFlow()

    // Token metrics tracking
    private var generationStartTime: Long = 0L
    private var firstTokenTime: Long = 0L
    private var tokenCount: Int = 0
    private var isFirstToken: Boolean = true

    override fun generateResponse(prompt: String): Flow<GenerationUpdate> = flow {
        try {
            // Reset metrics tracking
            generationStartTime = System.currentTimeMillis()
            firstTokenTime = 0L
            tokenCount = 0
            isFirstToken = true

            val response = StringBuilder()

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
            emit(GenerationUpdate(response.toString(), true))
        } catch (e: Exception) {
            // Emit error
            val metrics = createTokenMetrics()
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
}
