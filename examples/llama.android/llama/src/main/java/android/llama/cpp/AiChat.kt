package android.llama.cpp

import android.content.Context
import android.llama.cpp.internal.InferenceEngineImpl
import android.llama.cpp.internal.TierDetectionImpl

/**
 * Main entry point for the Ai Chat library.
 * This is the only class that should be used by library consumers.
 */
object AiChat {
    /**
     * Get the inference engine single instance.
     */
    fun getInferenceEngine(context: Context) = InferenceEngineImpl.getInstance(context)

    /**
     * Get tier detection single instance.
     */
    fun getTierDetection(context: Context): TierDetection = TierDetectionImpl.getInstance(context)
}
