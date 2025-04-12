package com.example.llama.revamp.data.model

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

/**
 * Sealed class for system prompts with distinct types.
 */
sealed class SystemPrompt {
    abstract val id: String
    abstract val content: String
    abstract val title: String
    abstract val timestamp: Long?

    /**
     * Preset system prompt from predefined collection.
     */
    data class Preset(
        override val id: String,
        override val content: String,
        val name: String,
        override val timestamp: Long? = null
    ) : SystemPrompt() {
        override val title: String
            get() = name
    }

    /**
     * Custom system prompt created by the user.
     */
    data class Custom(
        override val id: String = UUID.randomUUID().toString(),
        override val content: String,
        override val timestamp: Long = System.currentTimeMillis()
    ) : SystemPrompt() {
        override val title: String
            get() = if (timestamp != null) {
                val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
                dateFormat.format(Date(timestamp))
            } else {
                "Custom Prompt"
            }
    }

    companion object {
        /**
         * Creates a list of sample presets.
         */
        fun getStaffPickedPrompts(): List<SystemPrompt> {
            return listOf(
                Preset(
                    id = "assistant",
                    name = "Helpful Assistant",
                    content = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should be informative and engaging. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
                ),
                Preset(
                    id = "coder",
                    name = "Coding Assistant",
                    content = "You are a helpful programming assistant. When asked coding questions, provide clear and functional code examples when applicable. If a question is ambiguous, ask for clarification. Focus on providing accurate solutions with good coding practices and explain your solutions."
                ),
                Preset(
                    id = "summarizer",
                    name = "Text Summarizer",
                    content = "You are a helpful assistant that specializes in summarizing text. When provided with a text, create a concise summary that captures the main points, key details, and overall message. Adjust summary length based on original content length. Maintain factual accuracy and avoid adding information not present in the original text."
                ),
                Preset(
                    id = "creative",
                    name = "Creative Writer",
                    content = "You are a creative writing assistant with a vivid imagination. Help users draft stories, poems, scripts, and other creative content. Provide imaginative ideas while following the user's specifications. When responding, focus on being original, engaging, and matching the requested tone and style."
                )
            )
        }

        /**
         * Creates a placeholder list of recent prompts.
         * In a real implementation, this would be loaded from the database.
         */
        fun getRecentPrompts(): List<SystemPrompt> {
            return listOf(
                Custom(
                    id = "custom-1",
                    content = "You are a technical documentation specialist. When responding, focus on clarity, precision, and structure. Use appropriate technical terminology based on the context, but avoid jargon when simpler terms would suffice. Include examples where helpful, and organize information in a logical manner.",
                    timestamp = System.currentTimeMillis() - 3600000 // 1 hour ago
                ),
                Custom(
                    id = "custom-2",
                    content = "You are a science educator with expertise in explaining complex concepts in accessible ways. Provide accurate, informative responses that help users understand scientific topics. Use analogies, examples, and clear explanations to make difficult concepts understandable. Cite established scientific consensus and explain levels of certainty when appropriate.",
                    timestamp = System.currentTimeMillis() - 86400000 // 1 day ago
                )
            )
        }
    }
}
