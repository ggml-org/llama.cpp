package com.arm.aichat.internal

import android.content.Context
import android.util.Log
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.arm.aichat.ArmCpuTier
import com.arm.aichat.TierDetection
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking

/**
 * Internal [ArmCpuTier] detection implementation
 */
internal class TierDetectionImpl private constructor(
    private val context: Context
): TierDetection {

    companion object {
        private val TAG = TierDetectionImpl::class.simpleName

        // CPU feature detection preferences
        private const val DATASTORE_CPU_DETECTION = "cpu-detection"
        private const val DATASTORE_VERSION = 1
        private val Context.armCpuTierDataStore: DataStore<Preferences>
            by preferencesDataStore(name = DATASTORE_CPU_DETECTION)

        private val DETECTION_VERSION = intPreferencesKey("detection_version")
        private val DETECTED_TIER = intPreferencesKey("detected_tier")

        @Volatile
        private var instance: TierDetection? = null

        /**
         * Create or obtain [TierDetectionImpl]'s single instance.
         *
         * @param Context for obtaining the data store
         */
        internal fun getInstance(context: Context) =
            instance ?: synchronized(this) {
                instance ?: TierDetectionImpl(context).also { instance = it }
            }
    }

    private external fun getOptimalTier(): Int

    private external fun getCpuFeaturesString(): String

    private var _detectedTier: ArmCpuTier? = null

    /**
     * Get the detected tier, loading from cache if needed
     */
    override fun getDetectedTier(): ArmCpuTier? =
        _detectedTier ?: runBlocking { obtainTier() }

    /**
     * First attempt to load detected tier from storage, if available;
     * Otherwise, perform a fresh detection, then save to storage and cache.
     */
    private suspend fun obtainTier() =
        loadDetectedTierFromDataStore() ?: run {
            Log.i(TAG, "Performing fresh tier detection")
            performOptimalTierDetection().also { tier ->
                tier?.saveToDataStore()
                _detectedTier = tier
            }
        }

    /**
     * Load cached tier from datastore without performing detection
     */
    private suspend fun loadDetectedTierFromDataStore(): ArmCpuTier? {
        val preferences = context.armCpuTierDataStore.data.first()
        val cachedVersion = preferences[DETECTION_VERSION] ?: -1
        val cachedTierValue = preferences[DETECTED_TIER] ?: -1

        return if (cachedVersion == DATASTORE_VERSION && cachedTierValue >= 0) {
            ArmCpuTier.fromRawValue(cachedTierValue)?.also {
                Log.i(TAG, "Loaded cached tier: ${it.name}")
                _detectedTier = it
            }
        } else {
            Log.i(TAG, "No valid cached tier found")
            null
        }
    }

    /**
     * Actual implementation of optimal tier detection via native methods
     */
    private fun performOptimalTierDetection(): ArmCpuTier? {
        try {
            // Load CPU detection library
            System.loadLibrary("cpu-detector")
            Log.i(TAG, "CPU feature detector loaded successfully")

            // Detect optimal tier
            val tierValue = getOptimalTier()
            val features = getCpuFeaturesString()
            Log.i(TAG, "Raw tier $tierValue w/ CPU features: $features")

            // Convert to enum and validate
            val tier = ArmCpuTier.fromRawValue(tierValue) ?: run {
                Log.e(TAG, "Invalid tier value $tierValue")
                return ArmCpuTier.NONE
            }

            // Ensure we don't exceed maximum supported tier
            val maxTier = ArmCpuTier.maxSupportedTier
            return if (tier.rawValue > maxTier.rawValue) {
                Log.w(TAG, "Detected tier ${tier.name} exceeds max supported, using ${maxTier.name}")
                maxTier
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

    /**
     * Clear cached detection results (for testing/debugging)
     */
    override fun clearCache() {
        runBlocking { context.armCpuTierDataStore.edit { it.clear() } }
        _detectedTier = null
        Log.i(TAG, "Cleared CPU detection results")
    }

    private suspend fun ArmCpuTier.saveToDataStore() {
        context.armCpuTierDataStore.edit { prefs ->
            prefs[DETECTED_TIER] = this.rawValue
            prefs[DETECTION_VERSION] = DATASTORE_VERSION
        }
        Log.i(TAG, "Saved ${this.name} to data store")
    }
}
