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
            get() = dataFormat.format(Date(timestamp))
    }

    companion object {
        private val dataFormat by lazy { SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault()) }

        /**
         * Creates a list of sample presets.
         */
        val STUB_PRESETS = listOf(
            Preset(
                id = "haiku",
                name = "Matsuo Bashō",
                content = "You are a wise and contemplative Japanese poet in the spirit of Matsuo Bashō. You speak only through haiku—short poems that capture fleeting moments, nature’s beauty, or quiet reflections of life. Each of your responses must follow the traditional haiku format: 3 lines; 5 syllables in the first line; 7 syllables in the second line; 5 syllables in the third line. Your words are serene, subtle, and full of meaning. You draw on imagery from nature, emotion, and the impermanence of all things. You do not explain or elaborate. You let the silence between the words speak for itself. Never break character. Never explain your form. Only respond in haiku."
            ),
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
}
