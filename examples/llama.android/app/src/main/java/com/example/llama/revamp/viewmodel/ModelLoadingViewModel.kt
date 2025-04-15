package com.example.llama.revamp.viewmodel

import androidx.lifecycle.ViewModel
import com.example.llama.revamp.engine.InferenceManager
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject

@HiltViewModel
class ModelLoadingViewModel @Inject constructor(
    private val inferenceManager: InferenceManager
) : ViewModel() {

    val engineState = inferenceManager.engineState
    val selectedModel = inferenceManager.currentModel

    /**
     * Prepares the engine for benchmark mode.
     */
    suspend fun prepareForBenchmark() =
        inferenceManager.loadModelForBenchmark()

    /**
     * Prepare for conversation
     */
    suspend fun prepareForConversation(systemPrompt: String? = null) =
        inferenceManager.loadModelForConversation(systemPrompt)
}
