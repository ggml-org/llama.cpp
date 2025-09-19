package com.arm.aiplayground.data.source.prefs

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.longPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
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
        private const val DATASTORE_SETTINGS = "settings"
        private val Context.settingsDataStore: DataStore<Preferences>
            by preferencesDataStore(name = DATASTORE_SETTINGS)

        // Preferences keys
        private val PERFORMANCE_MONITORING_ENABLED = booleanPreferencesKey("performance_monitoring_enabled")
        private val USE_FAHRENHEIT_TEMPERATURE = booleanPreferencesKey("use_fahrenheit_temperature")
        private val MONITORING_INTERVAL_MS = longPreferencesKey("monitoring_interval_ms")
        private val COLOR_THEME_MODE = intPreferencesKey("color_theme_mode")
        private val DARK_THEME_MODE = intPreferencesKey("dark_theme_mode")

        // Constants
        private const val DEFAULT_MONITORING_INTERVAL_MS = 5000L
    }

    /**
     * Gets whether performance monitoring is enabled.
     */
    fun isPerformanceMonitoringEnabled(): Flow<Boolean> =
        context.settingsDataStore.data.map { preferences ->
            preferences[PERFORMANCE_MONITORING_ENABLED] != false
        }

    /**
     * Sets whether performance monitoring is enabled.
     */
    suspend fun setPerformanceMonitoringEnabled(enabled: Boolean) = withContext(Dispatchers.IO) {
        context.settingsDataStore.edit { preferences ->
            preferences[PERFORMANCE_MONITORING_ENABLED] = enabled
        }
    }

    /**
     * Gets whether temperature should be displayed in Fahrenheit.
     */
    fun usesFahrenheitTemperature(): Flow<Boolean> =
        context.settingsDataStore.data.map { preferences ->
            preferences[USE_FAHRENHEIT_TEMPERATURE] == true
        }

    /**
     * Sets whether temperature should be displayed in Fahrenheit.
     */
    suspend fun setUseFahrenheitTemperature(useFahrenheit: Boolean) = withContext(Dispatchers.IO) {
        context.settingsDataStore.edit { preferences ->
            preferences[USE_FAHRENHEIT_TEMPERATURE] = useFahrenheit
        }
    }

    /**
     * Gets the monitoring interval in milliseconds.
     *
     * TODO-han.yin: replace with Enum value instead of millisecond value
     */
    fun getMonitoringInterval(): Flow<Long> =
        context.settingsDataStore.data.map { preferences ->
            preferences[MONITORING_INTERVAL_MS] ?: DEFAULT_MONITORING_INTERVAL_MS
        }

    /**
     * Sets the monitoring interval in milliseconds.
     */
    suspend fun setMonitoringInterval(intervalMs: Long) = withContext(Dispatchers.IO) {
        context.settingsDataStore.edit { preferences ->
            preferences[MONITORING_INTERVAL_MS] = intervalMs
        }
    }

    /**
     * Gets the current color theme mode.
     */
    fun getColorThemeMode(): Flow<ColorThemeMode> =
        context.settingsDataStore.data.map { preferences ->
            ColorThemeMode.fromInt(preferences[COLOR_THEME_MODE]) ?: ColorThemeMode.ARM
        }

    /**
     * Sets the color theme mode.
     */
    suspend fun setColorThemeMode(mode: ColorThemeMode) = withContext(Dispatchers.IO) {
        context.settingsDataStore.edit { preferences ->
            preferences[COLOR_THEME_MODE] = mode.value
        }
    }

    /**
     * Gets the current dark theme mode.
     */
    fun getDarkThemeMode(): Flow<DarkThemeMode> =
        context.settingsDataStore.data.map { preferences ->
            DarkThemeMode.fromInt(preferences[DARK_THEME_MODE]) ?: DarkThemeMode.AUTO
        }

    /**
     * Sets the dark theme mode.
     */
    suspend fun setDarkThemeMode(mode: DarkThemeMode) = withContext(Dispatchers.IO) {
        context.settingsDataStore.edit { preferences ->
            preferences[DARK_THEME_MODE] = mode.value
        }
    }
}

enum class ColorThemeMode(val value: Int, val label: String) {
    ARM(0, "Arm®"),
    MATERIAL(1, "Material Design");

    companion object {
        fun fromInt(value: Int?) = entries.find { it.value == value }
    }
}

enum class DarkThemeMode(val value: Int, val label: String) {
    AUTO(0, "Auto"),
    LIGHT(1, "Light"),
    DARK(2, "Dark");

    companion object {
        fun fromInt(value: Int?) = entries.find { it.value == value }
    }
}
