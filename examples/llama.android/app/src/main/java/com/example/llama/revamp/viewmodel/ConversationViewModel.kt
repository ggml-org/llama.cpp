package com.example.llama.revamp.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.engine.InferenceManager
import com.example.llama.revamp.engine.TokenMetrics
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.inject.Inject
import kotlin.getValue


@HiltViewModel
class ConversationViewModel @Inject constructor(
    private val inferenceManager: InferenceManager
) : ViewModel() {

    val engineState = inferenceManager.engineState
    val selectedModel = inferenceManager.currentModel
    val systemPrompt = inferenceManager.systemPrompt

    // Messages in conversation
    private val _messages = MutableStateFlow<List<Message>>(emptyList())
    val messages: StateFlow<List<Message>> = _messages.asStateFlow()

    // Token generation job
    private var tokenCollectionJob: Job? = null

    /**
     * Send a message with the provided content.
     * Note: This matches the existing UI which manages input state outside the ViewModel.
     */
    fun sendMessage(content: String) {
        if (content.isBlank()) return

        // Cancel ongoing collection
        tokenCollectionJob?.cancel()

        // Add user message
        val userMessage = Message.User(
            content = content,
            timestamp = System.currentTimeMillis()
        )
        _messages.value = _messages.value + userMessage

        // Add placeholder for assistant response
        val assistantMessage = Message.Assistant.Ongoing(
            content = "",
            timestamp = System.currentTimeMillis()
        )
        _messages.value = _messages.value + assistantMessage

        // Collect response
        tokenCollectionJob = viewModelScope.launch {
            try {
                inferenceManager.generateResponse(content)
                    .collect { (text, isComplete) ->
                        updateAssistantMessage(text, isComplete)
                    }
            } catch (e: Exception) {
                // Handle error
                handleResponseError(e)
            }
        }
    }

    /**
     * Handle updating the assistant message
     */
    private fun updateAssistantMessage(text: String, isComplete: Boolean) {
        val currentMessages = _messages.value.toMutableList()
        val lastIndex = currentMessages.size - 1
        val currentAssistantMessage = currentMessages.getOrNull(lastIndex) as? Message.Assistant.Ongoing

        if (currentAssistantMessage != null) {
            if (isComplete) {
                // Final message with metrics
                currentMessages[lastIndex] = Message.Assistant.Completed(
                    content = text,
                    timestamp = currentAssistantMessage.timestamp,
                    metrics = inferenceManager.createTokenMetrics()
                )
            } else {
                // Ongoing message update
                currentMessages[lastIndex] = Message.Assistant.Ongoing(
                    content = text,
                    timestamp = currentAssistantMessage.timestamp
                )
            }
            _messages.value = currentMessages
        }
    }

    /**
     * Handle response error
     */
    private fun handleResponseError(e: Exception) {
        val currentMessages = _messages.value.toMutableList()
        val lastIndex = currentMessages.size - 1
        val currentAssistantMessage = currentMessages.getOrNull(lastIndex) as? Message.Assistant.Ongoing

        if (currentAssistantMessage != null) {
            currentMessages[lastIndex] = Message.Assistant.Completed(
                content = "${currentAssistantMessage.content}[Error: ${e.message}]",
                timestamp = currentAssistantMessage.timestamp,
                metrics = inferenceManager.createTokenMetrics()
            )
            _messages.value = currentMessages
        }
    }

    /**
     * Clear conversation
     */
    fun clearConversation() {
        tokenCollectionJob?.cancel()
        _messages.value = emptyList()
    }

    override fun onCleared() {
        tokenCollectionJob?.cancel()
        super.onCleared()
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

