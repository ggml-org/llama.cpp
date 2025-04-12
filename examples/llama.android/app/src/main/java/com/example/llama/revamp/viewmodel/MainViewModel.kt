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

    // Benchmark parameters
    private var pp: Int = 32
    private var tg: Int = 32
    private var pl: Int = 512

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
        inferenceEngine.bench(pp, tg, pl)
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
    fun prepareForConversation(systemPrompt: String? = null) {
        _systemPrompt.value = systemPrompt
        viewModelScope.launch {
            _selectedModel.value?.let { model ->
                inferenceEngine.loadModel(model.path, systemPrompt)
            }
        }
    }

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
        val assistantMessage = Message.Assistant(
            content = "",
            timestamp = System.currentTimeMillis(),
            isComplete = false
        )
        _messages.value = _messages.value + assistantMessage

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
                            val currentAssistantMessage = currentMessages[messageIndex] as? Message.Assistant
                            if (currentAssistantMessage != null) {
                                currentMessages[messageIndex] = currentAssistantMessage.copy(
                                    content = "${response}[Error: ${e.message}]",
                                    isComplete = true
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
                            val currentAssistantMessage = currentMessages.getOrNull(messageIndex) as? Message.Assistant
                            if (currentAssistantMessage != null) {
                                currentMessages[messageIndex] = currentAssistantMessage.copy(
                                    isComplete = true
                                )
                                _messages.value = currentMessages
                            }
                        }
                    }
                    .collect { token ->
                        response.append(token)

                        // Safely update the assistant message with the generated text
                        val currentMessages = _messages.value.toMutableList()
                        if (currentMessages.isNotEmpty()) {
                            val messageIndex = currentMessages.size - 1
                            val currentAssistantMessage = currentMessages.getOrNull(messageIndex) as? Message.Assistant
                            if (currentAssistantMessage != null) {
                                currentMessages[messageIndex] = currentAssistantMessage.copy(
                                    content = response.toString(),
                                    isComplete = false
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
                    val currentAssistantMessage = currentMessages.getOrNull(messageIndex) as? Message.Assistant
                    if (currentAssistantMessage != null) {
                        currentMessages[messageIndex] = currentAssistantMessage.copy(
                            content = "${response}[Error: ${e.message}]",
                            isComplete = true
                        )
                        _messages.value = currentMessages
                    }
                }
            }
        }
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
     * Checks if a model is currently loaded.
     */
    fun isModelLoaded(): Boolean {
        return engineState.value !is InferenceEngine.State.Uninitialized &&
            engineState.value !is InferenceEngine.State.LibraryLoaded
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
    abstract val content: String
    abstract val timestamp: Long

    val formattedTime: String
        get() {
            val formatter = SimpleDateFormat("h:mm a", Locale.getDefault())
            return formatter.format(Date(timestamp))
        }

    data class User(
        override val content: String,
        override val timestamp: Long
    ) : Message()

    data class Assistant(
        override val content: String,
        override val timestamp: Long,
        val isComplete: Boolean = true
    ) : Message()
}
