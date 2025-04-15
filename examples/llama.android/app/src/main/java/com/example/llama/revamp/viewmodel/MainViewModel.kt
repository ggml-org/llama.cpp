package com.example.llama.revamp.viewmodel

import androidx.lifecycle.ViewModel
import com.example.llama.revamp.engine.InferenceEngine
import com.example.llama.revamp.engine.InferenceManager
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject

@HiltViewModel
/**
 * Main ViewModel that expose the core states of [InferenceEngine]
 */
class MainViewModel @Inject constructor (
    private val inferenceManager: InferenceManager,
) : ViewModel() {

    val engineState = inferenceManager.engineState

    /**
     * Unload the current model and release the resources
     */
    suspend fun unloadModel() = inferenceManager.unloadModel()

    companion object {
        private val TAG = MainViewModel::class.java.simpleName

        private const val SUBSCRIPTION_TIMEOUT_MS = 5000L
    }
}

