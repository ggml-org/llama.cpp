package com.example.llama.viewmodel

import android.llama.cpp.LLamaTier
import android.llama.cpp.TierDetection
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.data.repo.ModelRepository
import com.example.llama.data.source.prefs.UserPreferences
import com.example.llama.monitoring.BatteryMetrics
import com.example.llama.monitoring.MemoryMetrics
import com.example.llama.monitoring.PerformanceMonitor
import com.example.llama.monitoring.StorageMetrics
import com.example.llama.monitoring.TemperatureMetrics
import com.example.llama.monitoring.TemperatureWarningLevel
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import javax.inject.Inject

/**
 * ViewModel that manages performance monitoring for the app.
 */
@HiltViewModel
class SettingsViewModel @Inject constructor(
    private val userPreferences: UserPreferences,
    private val performanceMonitor: PerformanceMonitor,
    private val modelRepository: ModelRepository,
    private val tierDetection: TierDetection,
) : ViewModel() {

    // Storage usage metrics
    private val _storageMetrics = MutableStateFlow<StorageMetrics?>(null)
    val storageMetrics: StateFlow<StorageMetrics?> = _storageMetrics.asStateFlow()

    // Memory usage metrics
    private val _memoryUsage = MutableStateFlow(MemoryMetrics(0, 0, 0, 0f, 0f))
    val memoryUsage: StateFlow<MemoryMetrics> = _memoryUsage.asStateFlow()

    // Battery information
    private val _batteryInfo = MutableStateFlow(BatteryMetrics(0, false))
    val batteryInfo: StateFlow<BatteryMetrics> = _batteryInfo.asStateFlow()

    // Temperature information
    private val _temperatureMetrics = MutableStateFlow(TemperatureMetrics(0f, TemperatureWarningLevel.NORMAL))
    val temperatureMetrics: StateFlow<TemperatureMetrics> = _temperatureMetrics.asStateFlow()

    // User preferences
    private val _isMonitoringEnabled = MutableStateFlow(true)
    val isMonitoringEnabled: StateFlow<Boolean> = _isMonitoringEnabled.asStateFlow()

    private val _useFahrenheitUnit = MutableStateFlow(false)
    val useFahrenheitUnit: StateFlow<Boolean> = _useFahrenheitUnit.asStateFlow()

    private val _monitoringInterval = MutableStateFlow(5000L)
    val monitoringInterval: StateFlow<Long> = _monitoringInterval.asStateFlow()

    private val _themeMode = MutableStateFlow(UserPreferences.THEME_MODE_AUTO)
    val themeMode: StateFlow<Int> = _themeMode.asStateFlow()

    val detectedTier: LLamaTier?
        get() = tierDetection.detectedTier

    init {
        viewModelScope.launch {
            // Load user preferences
            _isMonitoringEnabled.value = userPreferences.isPerformanceMonitoringEnabled().first()
            _useFahrenheitUnit.value = userPreferences.usesFahrenheitTemperature().first()
            _monitoringInterval.value = userPreferences.getMonitoringInterval().first()
            _themeMode.value = userPreferences.getThemeMode().first()

            // Start monitoring if enabled
            if (_isMonitoringEnabled.value) {
                startMonitoring()
            }

            viewModelScope.launch {
                userPreferences.getThemeMode().collect { mode ->
                    _themeMode.value = mode
                }
            }
        }
    }

    /**
     * Starts monitoring device performance.
     */
    private var monitoringJob: Job? = null
    private fun startMonitoring() {
        val interval = _monitoringInterval.value

        monitoringJob?.cancel()
        viewModelScope.launch {
            launch {
                modelRepository.getStorageMetrics().collect { metrics ->
                    _storageMetrics.value = metrics
                }
            }

            launch {
                performanceMonitor.monitorMemoryUsage(interval).collect { metrics ->
                    _memoryUsage.value = metrics
                }
            }

            launch {
                performanceMonitor.monitorBattery(interval * 2).collect { metrics ->
                    _batteryInfo.value = metrics
                }
            }

            launch {
                performanceMonitor.monitorTemperature(interval * 2).collect { metrics ->
                    _temperatureMetrics.value = metrics
                }
            }
        }
    }

    /**
     * Sets whether performance monitoring is enabled.
     */
    fun setMonitoringEnabled(enabled: Boolean) {
        viewModelScope.launch {
            if (enabled && !_isMonitoringEnabled.value) {
                startMonitoring()
            }
            _isMonitoringEnabled.value = enabled
            userPreferences.setPerformanceMonitoringEnabled(enabled)
        }
    }

    /**
     * Sets the temperature unit preference.
     */
    fun setUseFahrenheitUnit(useFahrenheit: Boolean) {
        viewModelScope.launch {
            userPreferences.setUseFahrenheitTemperature(useFahrenheit)
            _useFahrenheitUnit.value = useFahrenheit
        }
    }

    /**
     * Sets the monitoring interval.
     */
    fun setMonitoringInterval(intervalMs: Long) {
        viewModelScope.launch {
            // Restart monitoring with new interval if active
            if (_isMonitoringEnabled.value) {
                startMonitoring()
            }
            userPreferences.setMonitoringInterval(intervalMs)
            _monitoringInterval.value = intervalMs
        }
    }

    /**
     * Sets the theme mode.
     */
    fun setThemeMode(mode: Int) {
        viewModelScope.launch {
            userPreferences.setThemeMode(mode)
            _themeMode.value = mode
        }
    }
}
