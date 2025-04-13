package com.example.llama.revamp.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.engine.InferenceEngine
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Main ViewModel that handles the LLM engine state and operations.
 */
class MainViewModel(
    private val inferenceEngine: InferenceEngine = InferenceEngine()
) : ViewModel() {

    // Expose the engine state
    val engineState: StateFlow<InferenceEngine.State> = inferenceEngine.state

    // Benchmark results
    val benchmarkResults: StateFlow<String?> = inferenceEngine.benchmarkResults

    // Selected model information
    private val _selectedModel = MutableStateFlow<ModelInfo?>(null)
    val selectedModel: StateFlow<ModelInfo?> = _selectedModel.asStateFlow()

    // Messages in the conversation
    private val _messages = MutableStateFlow<List<Message>>(emptyList())
    val messages: StateFlow<List<Message>> = _messages.asStateFlow()

    // System prompt for the conversation
    private val _systemPrompt = MutableStateFlow<String?>(null)
    val systemPrompt: StateFlow<String?> = _systemPrompt.asStateFlow()

    // Flag to track if token collection is active
    private var tokenCollectionJob: Job? = null

    /**
     * Selects a model for use.
     */
    fun selectModel(modelInfo: ModelInfo) {
        _selectedModel.value = modelInfo
    }

    /**
     * Prepares the engine for benchmark mode.
     */
    fun prepareForBenchmark() {
        viewModelScope.launch {
            _selectedModel.value?.let { model ->
                inferenceEngine.loadModel(model.path)
                runBenchmark()
            }
        }
    }

    /**
     * Runs the benchmark with current parameters.
     */
    private suspend fun runBenchmark() {
        inferenceEngine.bench(512, 128, 1, 3)
    }

    /**
     * Reruns the benchmark.
     */
    fun rerunBenchmark() {
        viewModelScope.launch {
            runBenchmark()
        }
    }

    /**
     * Prepares the engine for conversation mode.
     */
    suspend fun prepareForConversation(systemPrompt: String? = null) {
        _systemPrompt.value = systemPrompt
        _selectedModel.value?.let { model ->
            inferenceEngine.loadModel(model.path, systemPrompt)
        }
    }

    /**
     * Tracks token generation metrics
     */
    private var generationStartTime: Long = 0L
    private var firstTokenTime: Long = 0L
    private var tokenCount: Int = 0
    private var isFirstToken: Boolean = true

    /**
     * Sends a user message and collects the response.
     */
    fun sendMessage(content: String) {
        if (content.isBlank()) return

        // Cancel any ongoing token collection
        tokenCollectionJob?.cancel()

        // Add user message
        val userMessage = Message.User(
            content = content,
            timestamp = System.currentTimeMillis()
        )
        _messages.value = _messages.value + userMessage

        // Create placeholder for assistant message
        val assistantMessage = Message.Assistant.Ongoing(
            content = "",
            timestamp = System.currentTimeMillis()
        )
        _messages.value = _messages.value + assistantMessage

        // Reset metrics tracking
        generationStartTime = System.currentTimeMillis()
        firstTokenTime = 0L
        tokenCount = 0
        isFirstToken = true

        // Get response from engine
        tokenCollectionJob = viewModelScope.launch {
            val response = StringBuilder()

            try {
                inferenceEngine.sendUserPrompt(content)
                    .catch { e ->
                        // Handle errors during token collection
                        val currentMessages = _messages.value.toMutableList()
                        if (currentMessages.size >= 2) {
                            val messageIndex = currentMessages.size - 1
                            val currentAssistantMessage = currentMessages[messageIndex] as? Message.Assistant.Ongoing
                            if (currentAssistantMessage != null) {
                                // Create metrics with error indication
                                val errorMetrics = TokenMetrics(
                                    tokensCount = tokenCount,
                                    ttftMs = if (firstTokenTime > 0) firstTokenTime - generationStartTime else 0L,
                                    tpsMs = calculateTPS(tokenCount, System.currentTimeMillis() - generationStartTime)
                                )

                                currentMessages[messageIndex] = Message.Assistant.Completed(
                                    content = "${response}[Error: ${e.message}]",
                                    timestamp = currentAssistantMessage.timestamp,
                                    metrics = errorMetrics
                                )
                                _messages.value = currentMessages
                            }
                        }
                    }
                    .onCompletion { cause ->
                        // Handle completion (normal or cancelled)
                        val currentMessages = _messages.value.toMutableList()
                        if (currentMessages.isNotEmpty()) {
                            val messageIndex = currentMessages.size - 1
                            val currentAssistantMessage = currentMessages.getOrNull(messageIndex) as? Message.Assistant.Ongoing
                            if (currentAssistantMessage != null) {
                                // Calculate final metrics
                                val endTime = System.currentTimeMillis()
                                val totalTimeMs = endTime - generationStartTime

                                val metrics = TokenMetrics(
                                    tokensCount = tokenCount,
                                    ttftMs = if (firstTokenTime > 0) firstTokenTime - generationStartTime else 0L,
                                    tpsMs = calculateTPS(tokenCount, totalTimeMs)
                                )

                                currentMessages[messageIndex] = Message.Assistant.Completed(
                                    content = response.toString(),
                                    timestamp = currentAssistantMessage.timestamp,
                                    metrics = metrics
                                )
                                _messages.value = currentMessages
                            }
                        }
                    }
                    .collect { token ->
                        // Track first token time
                        if (isFirstToken && token.isNotBlank()) {
                            firstTokenTime = System.currentTimeMillis()
                            isFirstToken = false
                        }

                        // Count tokens - each non-empty emission is at least one token
                        if (token.isNotBlank()) {
                            tokenCount++
                        }

                        response.append(token)

                        // Safely update the assistant message with the generated text
                        val currentMessages = _messages.value.toMutableList()
                        if (currentMessages.isNotEmpty()) {
                            val messageIndex = currentMessages.size - 1
                            val currentAssistantMessage = currentMessages.getOrNull(messageIndex) as? Message.Assistant.Ongoing
                            if (currentAssistantMessage != null) {
                                currentMessages[messageIndex] = Message.Assistant.Ongoing(
                                    content = response.toString(),
                                    timestamp = currentAssistantMessage.timestamp
                                )
                                _messages.value = currentMessages
                            }
                        }
                    }
            } catch (e: Exception) {
                // Handle any unexpected exceptions
                val currentMessages = _messages.value.toMutableList()
                if (currentMessages.isNotEmpty()) {
                    val messageIndex = currentMessages.size - 1
                    val currentAssistantMessage = currentMessages.getOrNull(messageIndex) as? Message.Assistant.Ongoing
                    if (currentAssistantMessage != null) {
                        // Create metrics with error indication
                        val errorMetrics = TokenMetrics(
                            tokensCount = tokenCount,
                            ttftMs = if (firstTokenTime > 0) firstTokenTime - generationStartTime else 0L,
                            tpsMs = calculateTPS(tokenCount, System.currentTimeMillis() - generationStartTime)
                        )

                        currentMessages[messageIndex] = Message.Assistant.Completed(
                            content = "${response}[Error: ${e.message}]",
                            timestamp = currentAssistantMessage.timestamp,
                            metrics = errorMetrics
                        )
                        _messages.value = currentMessages
                    }
                }
            }
        }
    }

    /**
     * Calculate tokens per second.
     */
    private fun calculateTPS(tokens: Int, timeMs: Long): Float {
        if (tokens <= 0 || timeMs <= 0) return 0f
        return (tokens.toFloat() * 1000f) / timeMs
    }

    /**
     * Unloads the currently loaded model.
     */
    suspend fun unloadModel() {
        // Cancel any ongoing token collection
        tokenCollectionJob?.cancel()

        // Clear messages
        _messages.value = emptyList()

        // Unload model
        inferenceEngine.unloadModel()
    }

    /**
     * Checks if a model is currently being loaded.
     */
    fun isModelLoading() =
        engineState.value.let {
            it is InferenceEngine.State.LoadingModel
                || it is InferenceEngine.State.ProcessingSystemPrompt
        }

    /**
     * Checks if a model has already been loaded.
     */
    fun isModelLoaded() =
        engineState.value.let {
            it !is InferenceEngine.State.Uninitialized
                && it !is InferenceEngine.State.LibraryLoaded
        }

    /**
     * Clean up resources when ViewModel is cleared.
     */
    override fun onCleared() {
        inferenceEngine.destroy()
        super.onCleared()
    }

    /**
     * Factory for creating MainViewModel instances.
     */
    class Factory(private val inferenceEngine: InferenceEngine) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            if (modelClass.isAssignableFrom(MainViewModel::class.java)) {
                return MainViewModel(inferenceEngine) as T
            }
            throw IllegalArgumentException("Unknown ViewModel class")
        }
    }
}

/**
 * Sealed class representing messages in a conversation.
 */
sealed class Message {
    abstract val timestamp: Long
    abstract val content: String

    val formattedTime: String
        get() = datetimeFormatter.format(Date(timestamp))

    data class User(
        override val timestamp: Long,
        override val content: String
    ) : Message()

    sealed class Assistant : Message() {
        data class Ongoing(
            override val timestamp: Long,
            override val content: String,
        ) : Assistant()

        data class Completed(
            override val timestamp: Long,
            override val content: String,
            val metrics: TokenMetrics
        ) : Assistant()
    }

    companion object {
        private val datetimeFormatter by lazy { SimpleDateFormat("h:mm a", Locale.getDefault()) }
    }
}

data class TokenMetrics(
    val tokensCount: Int,
    val ttftMs: Long,
    val tpsMs: Float,
) {
    val text: String
        get() = "Tokens: $tokensCount, TTFT: ${ttftMs}ms, TPS: ${"%.1f".format(tpsMs)}"
}
