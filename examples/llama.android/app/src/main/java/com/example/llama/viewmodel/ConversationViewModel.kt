package com.example.llama.viewmodel

import androidx.compose.foundation.text.input.TextFieldState
import androidx.compose.foundation.text.input.clearText
import androidx.lifecycle.viewModelScope
import com.example.llama.engine.ConversationService
import com.example.llama.engine.GenerationUpdate
import com.example.llama.engine.TokenMetrics
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.launch
import okhttp3.internal.toImmutableList
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

    // UI state: Model card
    private val _showModelCard = MutableStateFlow(false)
    val showModelCard = _showModelCard.asStateFlow()

    fun toggleModelCard(show: Boolean) {
        _showModelCard.value = show
    }

    // UI state: conversation messages
    private val _messages = MutableStateFlow<List<Message>>(emptyList())
    val messages: StateFlow<List<Message>> = _messages.asStateFlow()

    // UI state: Input text field
    val inputFieldState = TextFieldState()

    // Ongoing coroutine jobs
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
    fun stopGeneration() =
        tokenCollectionJob?.let { job ->
            // handled by the catch blocks
            if (job.isActive) { job.cancel() }
        }

    /**
     * Handle the case when generation is explicitly cancelled by adding a stopping suffix
     */
    private fun handleCancellation() =
        _messages.value.toMutableList().apply {
            (removeLastOrNull() as? Message.Assistant.Stopped)?.let {
                add(it.copy(content = it.content + SUFFIX_GENERATION_STOPPED))
                _messages.value = toImmutableList()
            }
        }

    /**
     * Handle response error by appending an error suffix
     */
    private fun handleResponseError(e: Exception) =
        _messages.value.toMutableList().apply {
            (removeLastOrNull() as? Message.Assistant.Stopped)?.let {
                add(it.copy(content = it.content + SUFFIX_GENERATION_ERROR.format(e.message)))
                _messages.value = toImmutableList()
            }
        }

    /**
     * Handle updating the assistant message
     */
    private fun updateAssistantMessage(update: GenerationUpdate) =
        _messages.value.toMutableList().apply {
            (removeLastOrNull() as? Message.Assistant.Ongoing)?.let {
                if (update.metrics != null) {
                    // Finalized message (partial or complete) with metrics
                    add(Message.Assistant.Stopped(
                        content = update.text,
                        timestamp = it.timestamp,
                        metrics = update.metrics
                    ))
                } else if (!update.isComplete) {
                    // Ongoing message update
                    add(Message.Assistant.Ongoing(
                        content = update.text,
                        timestamp = it.timestamp
                    ))
                }
                _messages.value = toImmutableList()
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

    companion object {
        private const val SUFFIX_GENERATION_STOPPED = " [Generation stopped]"
        private const val SUFFIX_GENERATION_ERROR = " [Error: %s]"
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

        data class Stopped(
            override val timestamp: Long,
            override val content: String,
            val metrics: TokenMetrics
        ) : Assistant()
    }

    companion object {
        private val datetimeFormatter by lazy { SimpleDateFormat("h:mm a", Locale.getDefault()) }
    }
}

