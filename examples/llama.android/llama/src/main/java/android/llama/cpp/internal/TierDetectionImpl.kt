package android.llama.cpp.internal

import android.content.Context
import android.llama.cpp.LLamaTier
import android.llama.cpp.TierDetection

/**
 * Internal tier detection implementation
 */
internal class TierDetectionImpl(private val context: Context) : TierDetection {
    override val detectedTier: LLamaTier?
        get() = InferenceEngineLoader.detectedTier

    override fun clearCache() = InferenceEngineLoader.clearCache(context)
}
