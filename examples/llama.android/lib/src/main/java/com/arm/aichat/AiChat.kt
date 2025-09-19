package com.arm.aichat

import android.content.Context
import com.arm.aichat.internal.InferenceEngineImpl
import com.arm.aichat.internal.TierDetectionImpl

/**
 * Main entry point for Arm's AI Chat library.
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
