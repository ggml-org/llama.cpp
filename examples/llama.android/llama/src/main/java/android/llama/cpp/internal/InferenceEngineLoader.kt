package android.llama.cpp.internal

import android.content.Context
import android.llama.cpp.InferenceEngine
import android.llama.cpp.LLamaTier
import android.util.Log
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking

/**
 * Internal [android.llama.cpp.InferenceEngine] loader implementation
 */
internal object InferenceEngineLoader {
    private val TAG = InferenceEngineLoader::class.simpleName

    // CPU feature detection preferences
    private const val DATASTORE_CPU_DETECTION = "llama_cpu_detection"
    private val Context.llamaTierDataStore: DataStore<Preferences>
        by preferencesDataStore(name = DATASTORE_CPU_DETECTION)

    private val DETECTION_VERSION = intPreferencesKey("detection_version")
    private val DETECTED_TIER = intPreferencesKey("detected_tier")

    // Constants
    private const val DATASTORE_VERSION = 1

    @JvmStatic
    private external fun getOptimalTier(): Int

    @JvmStatic
    private external fun getCpuFeaturesString(): String

    private var _cachedInstance: InferenceEngineImpl? = null
    private var _detectedTier: LLamaTier? = null

    /**
     * Get the detected tier, loading from cache if needed
     */
    fun getDetectedTier(context: Context): LLamaTier? =
        _detectedTier ?: runBlocking {
            loadDetectedTierFromDataStore(context)
        }

    /**
     * Factory method to get a configured [InferenceEngineImpl] instance.
     * Handles tier detection, caching, and library loading automatically.
     */
    @Synchronized
    fun getInstance(context: Context): InferenceEngine? {
        // Return cached instance if available
        _cachedInstance?.let { return it }

        return runBlocking {
            // Obtain the optimal tier from cache if available
            val tier = loadDetectedTierFromDataStore(context) ?: run {
                Log.i(TAG, "Performing fresh tier detection")
                detectAndSaveOptimalTier(context)
            }

            if (tier == null || tier == LLamaTier.NONE) {
                Log.e(TAG, "Aborted instantiating Inference Engine due to invalid tier")
                return@runBlocking null
            }

            try {
                // Create and cache the inference engine instance
                Log.i(TAG, "Using tier: ${tier.name} (${tier.description})")
                InferenceEngineImpl.createWithTier(tier).also {
                    _cachedInstance = it
                    Log.i(TAG, "Successfully instantiated Inference Engine w/ ${tier.name}")
                }

            } catch (e: Exception) {
                Log.e(TAG, "Error instantiating Inference Engine", e)
                null
            }
        }
    }

    /**
     * Clear cached detection results (for testing/debugging)
     */
    fun clearCache(context: Context) {
        runBlocking { context.llamaTierDataStore.edit { it.clear() } }
        _cachedInstance = null
        _detectedTier = null
        Log.i(TAG, "Cleared detection results and cached instance")
    }

    /**
     * Load cached tier from datastore without performing detection
     */
    private suspend fun loadDetectedTierFromDataStore(context: Context): LLamaTier? {
        val preferences = context.llamaTierDataStore.data.first()
        val cachedVersion = preferences[DETECTION_VERSION] ?: -1
        val cachedTierValue = preferences[DETECTED_TIER] ?: -1

        return if (cachedVersion == DATASTORE_VERSION && cachedTierValue >= 0) {
            LLamaTier.fromRawValue(cachedTierValue)?.also {
                Log.i(TAG, "Loaded cached tier: ${it.name}")
                _detectedTier = it
            }
        } else {
            Log.i(TAG, "No valid cached tier found")
            null
        }
    }

    /**
     * Detect optimal tier and save to cache
     */
    private suspend fun detectAndSaveOptimalTier(context: Context): LLamaTier? =
        detectOptimalTier().also { tier ->
            tier?.saveToDataStore(context)
            _detectedTier = tier
        }

    /**
     * Detect optimal tier and save to cache
     */
    private fun detectOptimalTier(): LLamaTier? {
        try {
            // Load CPU detection library
            System.loadLibrary("llama_cpu_detector")
            Log.i(TAG, "CPU feature detector loaded successfully")

            // Detect optimal tier
            val tierValue = getOptimalTier()
            val features = getCpuFeaturesString()
            Log.i(TAG, "Raw tier $tierValue w/ CPU features: $features")

            // Convert to enum and validate
            val tier = LLamaTier.fromRawValue(tierValue) ?: run {
                Log.e(TAG, "Invalid tier value $tierValue")
                return LLamaTier.NONE
            }

            // Ensure we don't exceed maximum supported tier
            return if (tier.rawValue > LLamaTier.maxSupportedTier.rawValue) {
                Log.w(TAG, "Detected tier ${tier.name} exceeds max supported, using ${LLamaTier.maxSupportedTier.name}")
                LLamaTier.maxSupportedTier
            } else {
                tier
            }

        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Failed to load CPU detection library", e)
            return null

        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error during tier detection", e)
            return null
        }
    }

    private suspend fun LLamaTier.saveToDataStore(context: Context) {
        context.llamaTierDataStore.edit { prefs ->
            prefs[DETECTED_TIER] = this.rawValue
            prefs[DETECTION_VERSION] = DATASTORE_VERSION
        }
        Log.i(TAG, "Saved ${this.name} to data store")
    }
}
