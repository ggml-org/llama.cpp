package android.llama.cpp.internal

import android.content.Context
import android.llama.cpp.TierDetection

/**
 * Internal factory to create [InferenceEngine] and [TierDetection]
 */
internal object InferenceEngineFactory {
    fun getInstance(context: Context) = InferenceEngineLoader.getInstance(context)

    fun getTierDetection(context: Context): TierDetection = TierDetectionImpl(context)
}
