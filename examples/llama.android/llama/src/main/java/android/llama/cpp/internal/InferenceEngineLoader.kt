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
    val detectedTier: LLamaTier? get() = _detectedTier

    /**
     * Factory method to get a configured [InferenceEngineImpl] instance.
     * Handles tier detection, caching, and library loading automatically.
     */
    @Synchronized
    fun createInstance(context: Context): InferenceEngine? {
        // Return cached instance if available
        _cachedInstance?.let { return it }

        return runBlocking {
            try {
                // Obtain the optimal tier from cache if available
                val tier = getOrDetectOptimalTier(context) ?: run {
                    Log.e(TAG, "Failed to determine optimal tier")
                    return@runBlocking null
                }
                _detectedTier = tier
                Log.i(TAG, "Using tier: ${tier.name} (${tier.description})")

                // Create and cache the inference engine instance
                val instance = InferenceEngineImpl.createWithTier(tier) ?: run {
                    Log.e(TAG, "Failed to instantiate InferenceEngineImpl")
                    return@runBlocking null
                }
                _cachedInstance = instance
                Log.i(TAG, "Successfully created InferenceEngineImpl instance with ${tier.name}")

                instance

            } catch (e: Exception) {
                Log.e(TAG, "Error creating InferenceEngineImpl instance", e)
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
     * Get optimal tier from cache or detect it fresh
     */
    private suspend fun getOrDetectOptimalTier(context: Context): LLamaTier? {
        val preferences = context.llamaTierDataStore.data.first()

        // Check if we have a cached result with the current detection version
        val cachedVersion = preferences[DETECTION_VERSION] ?: -1
        val cachedTierValue = preferences[DETECTED_TIER] ?: -1
        if (cachedVersion == DATASTORE_VERSION && cachedTierValue >= 0) {
            val cachedTier = LLamaTier.Companion.fromRawValue(cachedTierValue)
            if (cachedTier != null) {
                Log.i(TAG, "Using cached tier detection: ${cachedTier.name}")
                return cachedTier
            }
        }

        // No valid cache, detect fresh
        Log.i(TAG, "Performing fresh tier detection")
        return detectAndCacheOptimalTier(context)
    }

    /**
     * Detect optimal tier and save to cache
     */
    private suspend fun detectAndCacheOptimalTier(context: Context): LLamaTier? {
        try {
            // Load CPU detection library
            System.loadLibrary("llama_cpu_detector")
            Log.i(TAG, "CPU feature detector loaded successfully")

            // Detect optimal tier
            val tierValue = getOptimalTier()
            val features = getCpuFeaturesString()
            Log.i(TAG, "Raw tier $tierValue w/ CPU features: $features")

            // Convert to enum and validate
            val tier = LLamaTier.Companion.fromRawValue(tierValue) ?: run {
                Log.w(TAG, "Invalid tier value $tierValue")
                return null
            }

            // Ensure we don't exceed maximum supported tier
            val finalTier = if (tier.rawValue > LLamaTier.Companion.maxSupportedTier.rawValue) {
                Log.w(TAG, "Detected tier ${tier.name} exceeds max supported, using ${LLamaTier.Companion.maxSupportedTier.name}")
                LLamaTier.Companion.maxSupportedTier
            } else {
                tier
            }

            // Cache the result
            context.llamaTierDataStore.edit {
                it[DETECTED_TIER] = finalTier.rawValue
                it[DETECTION_VERSION] = DATASTORE_VERSION
            }

            Log.i(TAG, "Detected and cached optimal tier: ${finalTier.name}")
            return finalTier

        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Failed to load CPU detection library", e)

            // Fallback to T0 and cache it
            val fallbackTier = LLamaTier.T0
            context.llamaTierDataStore.edit {
                it[DETECTED_TIER] = fallbackTier.rawValue
                it[DETECTION_VERSION] = DATASTORE_VERSION
            }

            Log.i(TAG, "Using fallback tier: ${fallbackTier.name}")
            return fallbackTier

        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error during tier detection", e)
            return null
        }
    }
}
