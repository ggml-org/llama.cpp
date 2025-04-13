package com.example.llama.revamp.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.example.llama.revamp.data.preferences.UserPreferences
import com.example.llama.revamp.monitoring.BatteryMetrics
import com.example.llama.revamp.monitoring.MemoryMetrics
import com.example.llama.revamp.monitoring.PerformanceMonitor
import com.example.llama.revamp.monitoring.TemperatureMetrics
import com.example.llama.revamp.monitoring.TemperatureWarningLevel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

/**
 * ViewModel that manages performance monitoring for the app.
 */
class PerformanceViewModel(
    private val performanceMonitor: PerformanceMonitor,
    private val userPreferences: UserPreferences
) : ViewModel() {

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

    init {
        viewModelScope.launch {
            // Load user preferences
            _isMonitoringEnabled.value = userPreferences.isPerformanceMonitoringEnabled().first()
            _useFahrenheitUnit.value = userPreferences.usesFahrenheitTemperature().first()
            _monitoringInterval.value = userPreferences.getMonitoringInterval().first()

            // Start monitoring if enabled
            if (_isMonitoringEnabled.value) {
                startMonitoring()
            }
        }
    }

    /**
     * Starts monitoring device performance.
     */
    private fun startMonitoring() {
        val interval = _monitoringInterval.value

        viewModelScope.launch {
            performanceMonitor.monitorMemoryUsage(interval).collect { metrics ->
                _memoryUsage.value = metrics
            }
        }

        viewModelScope.launch {
            performanceMonitor.monitorBattery(interval * 2).collect { metrics ->
                _batteryInfo.value = metrics
            }
        }

        viewModelScope.launch {
            performanceMonitor.monitorTemperature(interval * 2).collect { metrics ->
                _temperatureMetrics.value = metrics
            }
        }
    }

    /**
     * Sets whether performance monitoring is enabled.
     */
    fun setMonitoringEnabled(enabled: Boolean) {
        viewModelScope.launch {
            userPreferences.setPerformanceMonitoringEnabled(enabled)
            _isMonitoringEnabled.value = enabled

            if (enabled && !isMonitoringActive()) {
                startMonitoring()
            }
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
            userPreferences.setMonitoringInterval(intervalMs)
            _monitoringInterval.value = intervalMs

            // Restart monitoring with new interval if active
            if (isMonitoringActive()) {
                startMonitoring()
            }
        }
    }

    /**
     * Checks if monitoring is currently active.
     */
    private fun isMonitoringActive(): Boolean {
        return _isMonitoringEnabled.value
    }

    /**
     * Factory for creating PerformanceViewModel instances.
     */
    class Factory(
        private val performanceMonitor: PerformanceMonitor,
        private val userPreferences: UserPreferences
    ) : ViewModelProvider.Factory {
        @Suppress("UNCHECKED_CAST")
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            if (modelClass.isAssignableFrom(PerformanceViewModel::class.java)) {
                return PerformanceViewModel(performanceMonitor, userPreferences) as T
            }
            throw IllegalArgumentException("Unknown ViewModel class")
        }
    }
}
