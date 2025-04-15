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

@Singleton
class InferenceManager @Inject constructor(
    private val inferenceEngine: InferenceEngine
) {
    // Expose engine state
    val engineState: StateFlow<InferenceEngine.State> = inferenceEngine.state

    // Benchmark results
    val benchmarkResults: StateFlow<String?> = inferenceEngine.benchmarkResults

    // Currently loaded model
    private val _currentModel = MutableStateFlow<ModelInfo?>(null)
    val currentModel: StateFlow<ModelInfo?> = _currentModel.asStateFlow()

    // System prompt
    private val _systemPrompt = MutableStateFlow<String?>(null)
    val systemPrompt: StateFlow<String?> = _systemPrompt.asStateFlow()

    // Token metrics tracking
    private var generationStartTime: Long = 0L
    private var firstTokenTime: Long = 0L
    private var tokenCount: Int = 0
    private var isFirstToken: Boolean = true

    /**
     * Set current model
     */
    fun setCurrentModel(model: ModelInfo) {
        _currentModel.value = model
    }

    /**
     * Load a model for benchmark
     */
    suspend fun loadModelForBenchmark(): Boolean {
        return _currentModel.value?.let { model ->
            try {
                inferenceEngine.loadModel(model.path)
                true
            } catch (e: Exception) {
                Log.e("InferenceManager", "Error loading model", e)
                false
            }
        } ?: false
    }

    /**
     * Load a model for conversation
     */
    suspend fun loadModelForConversation(systemPrompt: String? = null): Boolean {
        _systemPrompt.value = systemPrompt
        return _currentModel.value?.let { model ->
            try {
                inferenceEngine.loadModel(model.path, systemPrompt)
                true
            } catch (e: Exception) {
                Log.e("InferenceManager", "Error loading model", e)
                false
            }
        } ?: false
    }

    /**
     * Run benchmark
     */
    suspend fun benchmark(
        pp: Int = 512,
        tg: Int = 128,
        pl: Int = 1,
        nr: Int = 3
    ): String = inferenceEngine.bench(pp, tg, pl, nr)

    /**
     * Generate response from prompt
     */
    fun generateResponse(prompt: String): Flow<Pair<String, Boolean>> = flow {
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
                    emit(Pair(response.toString(), false))
                }

            // Calculate final metrics after completion
            val metrics = createTokenMetrics()

            // Emit final response with completion flag
            emit(Pair(response.toString(), true))
        } catch (e: Exception) {
            // Emit error
            val metrics = createTokenMetrics()
            throw e
        }
    }

    /**
     * Create token metrics based on current state
     */
    fun createTokenMetrics(): TokenMetrics {
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

    /**
     * Unload current model
     */
    suspend fun unloadModel() = inferenceEngine.unloadModel()

    /**
     * Cleanup resources
     */
    fun destroy() = inferenceEngine.destroy()
}

data class TokenMetrics(
    val tokensCount: Int,
    val ttftMs: Long,
    val tpsMs: Float,
) {
    val text: String
        get() = "Tokens: $tokensCount, TTFT: ${ttftMs}ms, TPS: ${"%.1f".format(tpsMs)}"
}
