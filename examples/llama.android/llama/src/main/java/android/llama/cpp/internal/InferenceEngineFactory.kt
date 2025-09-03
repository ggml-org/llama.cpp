package android.llama.cpp.internal

import android.content.Context
import android.llama.cpp.InferenceEngine
import android.llama.cpp.TierDetection
import android.util.Log
import kotlinx.coroutines.runBlocking

/**
 * Internal factory to create [InferenceEngine] and [TierDetection]
 */
internal object InferenceEngineFactory {
    private val TAG = InferenceEngineFactory::class.simpleName

    private var _cachedInstance: InferenceEngineImpl? = null

    /**
     * Factory method to get a configured [InferenceEngineImpl] instance.
     * Handles tier detection, caching, and library loading automatically.
     */
    @Synchronized
    fun getInstance(context: Context): InferenceEngine? {
        // Return cached instance if available
        _cachedInstance?.let { return it }

        return runBlocking {
            try {
                // Create and cache the inference engine instance
                InferenceEngineImpl.create(context).also {
                    _cachedInstance = it
                    Log.i(TAG, "Successfully instantiated Inference Engine")
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error instantiating Inference Engine", e)
                null
            }
        }
    }

    fun clearCache() {
        _cachedInstance = null
        Log.i(TAG, "Cleared cached instance of InferenceEngine")
    }
}
