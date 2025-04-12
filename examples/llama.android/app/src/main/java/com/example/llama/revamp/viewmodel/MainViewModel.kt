package com.example.llama.revamp.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.engine.InferenceEngine
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
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
        val messageIndex = _messages.value.size - 1

        viewModelScope.launch {
            val response = StringBuilder()

            inferenceEngine.sendUserPrompt(content).collect { token ->
                response.append(token)

                // Update the assistant message with the generated text
                val currentMessages = _messages.value.toMutableList()
                val currentAssistantMessage = currentMessages[messageIndex] as Message.Assistant
                currentMessages[messageIndex] = currentAssistantMessage.copy(
                    content = response.toString(),
                    isComplete = false
                )
                _messages.value = currentMessages
            }

            // Mark message as complete when generation finishes
            val finalMessages = _messages.value.toMutableList()
            val finalAssistantMessage = finalMessages[messageIndex] as Message.Assistant
            finalMessages[messageIndex] = finalAssistantMessage.copy(
                isComplete = true
            )
            _messages.value = finalMessages
        }
    }

    /**
     * Unloads the currently loaded model.
     */
    suspend fun unloadModel() {
        inferenceEngine.unloadModel()
        _messages.value = emptyList()
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
