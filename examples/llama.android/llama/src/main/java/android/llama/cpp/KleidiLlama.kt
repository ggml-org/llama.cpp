package android.llama.cpp

import android.content.Context
import android.llama.cpp.internal.InferenceEngineFactory
import android.llama.cpp.internal.TierDetectionImpl

/**
 * Main entry point for the Llama Android library.
 * This is the only class that should be used by library consumers.
 */
object KleidiLlama {
    /**
     * Create an inference engine instance with automatic tier detection.
     */
    fun createInferenceEngine(context: Context) = InferenceEngineFactory.getInstance(context)

    /**
     * Get tier detection information for debugging/settings.
     */
    fun getTierDetection(context: Context): TierDetection = TierDetectionImpl(context)
}
