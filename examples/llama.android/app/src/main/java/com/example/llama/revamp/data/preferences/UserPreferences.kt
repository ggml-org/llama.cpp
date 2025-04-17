package com.example.llama.revamp.data.preferences

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Manages user preferences for the application.
 */
@Singleton
class UserPreferences @Inject constructor (
    @ApplicationContext private val context: Context
) {

    companion object {
        private val Context.dataStore: DataStore<Preferences> by preferencesDataStore(name = "settings")

        // Performance monitoring preferences
        val PERFORMANCE_MONITORING_ENABLED = booleanPreferencesKey("performance_monitoring_enabled")
        val USE_FAHRENHEIT_TEMPERATURE = booleanPreferencesKey("use_fahrenheit_temperature")
        val MONITORING_INTERVAL_MS = longPreferencesKey("monitoring_interval_ms")

        // Default values
        const val DEFAULT_MONITORING_INTERVAL_MS = 5000L
    }

    /**
     * Gets whether performance monitoring is enabled.
     */
    fun isPerformanceMonitoringEnabled(): Flow<Boolean> {
        return context.dataStore.data.map { preferences ->
            preferences[PERFORMANCE_MONITORING_ENABLED] ?: true
        }
    }

    /**
     * Sets whether performance monitoring is enabled.
     */
    suspend fun setPerformanceMonitoringEnabled(enabled: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[PERFORMANCE_MONITORING_ENABLED] = enabled
        }
    }

    /**
     * Gets whether temperature should be displayed in Fahrenheit.
     */
    fun usesFahrenheitTemperature(): Flow<Boolean> {
        return context.dataStore.data.map { preferences ->
            preferences[USE_FAHRENHEIT_TEMPERATURE] ?: false
        }
    }

    /**
     * Sets whether temperature should be displayed in Fahrenheit.
     */
    suspend fun setUseFahrenheitTemperature(useFahrenheit: Boolean) {
        context.dataStore.edit { preferences ->
            preferences[USE_FAHRENHEIT_TEMPERATURE] = useFahrenheit
        }
    }

    /**
     * Gets the monitoring interval in milliseconds.
     *
     * TODO-han.yin: replace with Enum value instead of millisecond value
     */
    fun getMonitoringInterval(): Flow<Long> {
        return context.dataStore.data.map { preferences ->
            preferences[MONITORING_INTERVAL_MS] ?: DEFAULT_MONITORING_INTERVAL_MS
        }
    }

    /**
     * Sets the monitoring interval in milliseconds.
     */
    suspend fun setMonitoringInterval(intervalMs: Long) {
        context.dataStore.edit { preferences ->
            preferences[MONITORING_INTERVAL_MS] = intervalMs
        }
    }
}
