package android.llama.cpp

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import androidx.core.content.edit

enum class LLamaTier(val rawValue: Int, val libraryName: String, val description: String) {
    T0(0, "llama_android_t0", "ARMv8-a baseline with SIMD"),
    T1(1, "llama_android_t1", "ARMv8.2-a with DotProd"),
    T2(2, "llama_android_t2", "ARMv8.6-a with DotProd + I8MM"),
    T3(3, "llama_android_t3", "ARMv9-a with DotProd + I8MM + SVE/SVE2");
    // TODO-han.yin: implement T4 once obtaining an Android device with SME!

    companion object {
        fun fromRawValue(value: Int): LLamaTier? {
            return entries.find { it.rawValue == value }
        }

        fun getMaxSupportedTier(): LLamaTier = T3
    }
}

class InferenceEngineLoader private constructor() {

    companion object {
        private val TAG = InferenceEngineLoader::class.simpleName

        private const val DETECTION_VERSION = 1
        private const val PREFS_NAME = "llama_cpu_detection"
        private const val KEY_DETECTED_TIER = "detected_tier"
        private const val KEY_DETECTION_VERSION = "detection_version"

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

            try {
                // Obtain the optimal tier from cache if available
                val tier = getOrDetectOptimalTier(context) ?: run {
                    Log.e(TAG, "Failed to determine optimal tier")
                    return null
                }
                _detectedTier = tier
                Log.i(TAG, "Using tier: ${tier.name} (${tier.description})")

                // Create and cache the inference engine instance
                val instance = InferenceEngineImpl.createWithTier(tier) ?: run {
                    Log.e(TAG, "Failed to instantiate InferenceEngineImpl")
                    return null
                }
                _cachedInstance = instance
                Log.i(TAG, "Successfully created InferenceEngineImpl instance with ${tier.name}")

                return instance

            } catch (e: Exception) {
                Log.e(TAG, "Error creating InferenceEngineImpl instance", e)
                return null
            }
        }

        /**
         * Clear cached detection results (for testing/debugging)
         */
        fun clearCache(context: Context) {
            getSharedPrefs(context).edit { clear() }
            _cachedInstance = null
            _detectedTier = null
            Log.i(TAG, "Cleared detection results and cached instance")
        }

        /**
         * Get optimal tier from cache or detect it fresh
         */
        private fun getOrDetectOptimalTier(context: Context): LLamaTier? {
            val prefs = getSharedPrefs(context)

            // Check if we have a cached result with the current detection version
            val cachedVersion = prefs.getInt(KEY_DETECTION_VERSION, -1)
            val cachedTierValue = prefs.getInt(KEY_DETECTED_TIER, -1)
            if (cachedVersion == DETECTION_VERSION && cachedTierValue >= 0) {
                val cachedTier = LLamaTier.fromRawValue(cachedTierValue)
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
        private fun detectAndCacheOptimalTier(context: Context): LLamaTier? {
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
                    Log.w(TAG, "Invalid tier value $tierValue")
                    return null
                }

                // Ensure we don't exceed maximum supported tier
                val finalTier = if (tier.rawValue > LLamaTier.getMaxSupportedTier().rawValue) {
                    Log.w(TAG, "Detected tier ${tier.name} exceeds max supported, using ${LLamaTier.getMaxSupportedTier().name}")
                    LLamaTier.getMaxSupportedTier()
                } else {
                    tier
                }

                // Cache the result
                getSharedPrefs(context).edit {
                    putInt(KEY_DETECTED_TIER, finalTier.rawValue)
                    putInt(KEY_DETECTION_VERSION, DETECTION_VERSION)
                }

                Log.i(TAG, "Detected and cached optimal tier: ${finalTier.name}")
                return finalTier

            } catch (e: UnsatisfiedLinkError) {
                Log.e(TAG, "Failed to load CPU detection library", e)

                // Fallback to T0 and cache it
                val fallbackTier = LLamaTier.T0
                getSharedPrefs(context).edit {
                    putInt(KEY_DETECTED_TIER, fallbackTier.rawValue)
                    putInt(KEY_DETECTION_VERSION, DETECTION_VERSION)
                }

                Log.i(TAG, "Using fallback tier: ${fallbackTier.name}")
                return fallbackTier

            } catch (e: Exception) {
                Log.e(TAG, "Unexpected error during tier detection", e)
                return null
            }
        }

        private fun getSharedPrefs(context: Context): SharedPreferences {
            return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
        }
    }
}
