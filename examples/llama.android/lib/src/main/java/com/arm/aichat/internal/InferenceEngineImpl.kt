
    /**
     * Start a fresh native conversation while keeping the loaded model alive.
     *
     * The native system-prompt processing resets the active conversation state
     * before applying the new system prompt. This is not secure memory erasure.
     */
    override suspend fun resetConversation(systemPrompt: String) =
        withContext(llamaDispatcher) {
            require(systemPrompt.isNotBlank()) {
                "Cannot process empty system prompt!"
            }
            check(_state.value is InferenceEngine.State.ModelReady) {
                "Cannot reset conversation in ${_state.value.javaClass.simpleName}!"
            }

            Log.i(TAG, "Resetting native conversation...")
            _readyForSystemPrompt = false
            _state.value = InferenceEngine.State.ProcessingSystemPrompt

            processSystemPrompt(systemPrompt).let { result ->
                if (result != 0) {
                    val error = RuntimeException(
                        "Failed to reset conversation: $result"
                    )
                    _state.value = InferenceEngine.State.Error(error)
                    throw error
                }
            }

            _state.value = InferenceEngine.State.ModelReady
        }

    /**
     * Send plain text user prompt to LLM, which starts generating tokens in a [Flow]
     */
    override fun sendUserPrompt(
        message: String,
        predictLength: Int,
    ): Flow<String> = flow {
