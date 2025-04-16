package com.example.llama.revamp.viewmodel

import androidx.lifecycle.ViewModel
import com.example.llama.revamp.engine.InferenceService
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject

@HiltViewModel
/**
 * Main ViewModel that expose the core states of [InferenceEngine]
 */
class MainViewModel @Inject constructor (
    private val inferenceService: InferenceService,
) : ViewModel() {

    val engineState = inferenceService.engineState

    /**
     * Unload the current model and release the resources
     */
    suspend fun unloadModel() = inferenceService.unloadModel()
}

