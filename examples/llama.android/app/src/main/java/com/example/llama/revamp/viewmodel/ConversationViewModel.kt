package com.example.llama.revamp.viewmodel

import androidx.compose.foundation.text.input.TextFieldState
import androidx.compose.foundation.text.input.clearText
import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.engine.ConversationService
import com.example.llama.revamp.engine.GenerationUpdate
import com.example.llama.revamp.engine.TokenMetrics
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.inject.Inject


@HiltViewModel
class ConversationViewModel @Inject constructor(
    private val conversationService: ConversationService
) : ModelUnloadingViewModel(conversationService) {
    // Data
    val selectedModel = conversationService.currentSelectedModel
    val systemPrompt = conversationService.systemPrompt

    // Messages state
    private val _messages = MutableStateFlow<List<Message>>(emptyList())
    val messages: StateFlow<List<Message>> = _messages.asStateFlow()

    // Input text field state
    val inputFieldState = TextFieldState()

    // Token generation job
    private var tokenCollectionJob: Job? = null

    /**
     * Send a message with the provided content
     */
    fun sendMessage() {
        val content = inputFieldState.text.toString()
        if (content.isBlank()) return

        // Cancel ongoing collection
        stopGeneration()

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

        // Clear input field
        inputFieldState.clearText()

        // Collect response
        tokenCollectionJob = viewModelScope.launch {
            try {
                conversationService.generateResponse(content)
                    .onCompletion { tokenCollectionJob = null }
                    .collect(::updateAssistantMessage)

            } catch (_: CancellationException) {
                handleCancellation()
                tokenCollectionJob = null

            } catch (e: Exception) {
                handleResponseError(e)
                tokenCollectionJob = null
            }
        }
    }

    /**
     * Stop ongoing generation
     */
    fun stopGeneration() {
        tokenCollectionJob?.let { job ->
            // handled by the catch blocks
            if (job.isActive) { job.cancel() }
        }
    }

    /**
     * Handle the case when generation is explicitly cancelled
     */
    private fun handleCancellation() {
        val currentMessages = _messages.value.toMutableList()
        val lastIndex = currentMessages.size - 1
        val currentAssistantMessage = currentMessages.getOrNull(lastIndex) as? Message.Assistant.Ongoing

        if (currentAssistantMessage != null) {
            // Replace with completed message, adding note that it was interrupted
            currentMessages[lastIndex] = Message.Assistant.Completed(
                content = currentAssistantMessage.content + " [Generation stopped]",
                timestamp = currentAssistantMessage.timestamp,
                metrics = conversationService.createTokenMetrics()
            )
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
                content = currentAssistantMessage.content + " [Error: ${e.message}]",
                timestamp = currentAssistantMessage.timestamp,
                metrics = conversationService.createTokenMetrics()
            )
            _messages.value = currentMessages
        }
    }

    /**
     * Handle updating the assistant message
     */
    private fun updateAssistantMessage(update: GenerationUpdate) {
        val currentMessages = _messages.value.toMutableList()
        val lastIndex = currentMessages.size - 1
        val currentAssistantMessage = currentMessages.getOrNull(lastIndex) as? Message.Assistant.Ongoing

        if (currentAssistantMessage != null) {
            if (update.isComplete) {
                // Final message with metrics
                currentMessages[lastIndex] = Message.Assistant.Completed(
                    content = update.text,
                    timestamp = currentAssistantMessage.timestamp,
                    metrics = conversationService.createTokenMetrics()
                )
            } else {
                // Ongoing message update
                currentMessages[lastIndex] = Message.Assistant.Ongoing(
                    content = update.text,
                    timestamp = currentAssistantMessage.timestamp
                )
            }
            _messages.value = currentMessages
        }
    }

    override suspend fun performCleanup() = clearConversation()

    /**
     * Stop ongoing generation if any, then clean up all messages in the current conversation
     */
    fun clearConversation() {
        stopGeneration()
        _messages.value = emptyList()
    }

    override fun onCleared() {
        stopGeneration()
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

