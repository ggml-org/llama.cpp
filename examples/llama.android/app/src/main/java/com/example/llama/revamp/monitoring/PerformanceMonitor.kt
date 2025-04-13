package com.example.llama.revamp.monitoring

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

/**
 * Service that monitors device performance metrics such as memory usage,
 * battery level, and temperature.
 */
class PerformanceMonitor(private val context: Context) {

    /**
     * Provides a flow of memory usage information that updates at the specified interval.
     */
    fun monitorMemoryUsage(intervalMs: Long = MEMORY_POLLING_INTERVAL): Flow<MemoryMetrics> = flow {
        while(true) {
            emit(getMemoryInfo())
            delay(intervalMs)
        }
    }

    /**
     * Provides a flow of battery information that updates at the specified interval.
     */
    fun monitorBattery(intervalMs: Long = BATTERY_POLLING_INTERVAL): Flow<BatteryMetrics> = flow {
        while(true) {
            emit(getBatteryInfo())
            delay(intervalMs)
        }
    }

    /**
     * Provides a flow of temperature information that updates at the specified interval.
     */
    fun monitorTemperature(intervalMs: Long = TEMP_POLLING_INTERVAL): Flow<TemperatureMetrics> = flow {
        while(true) {
            emit(getTemperatureInfo())
            delay(intervalMs)
        }
    }

    /**
     * Gets the current memory usage information.
     */
    private suspend fun getMemoryInfo(): MemoryMetrics = withContext(Dispatchers.IO) {
        val mi = ActivityManager.MemoryInfo()
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        activityManager.getMemoryInfo(mi)

        val availableMem = mi.availMem
        val totalMem = mi.totalMem
        val percentUsed = ((totalMem - availableMem) / totalMem.toFloat() * 100).roundToInt()

        // Convert to more readable units (GB)
        val availableGb = (availableMem / (1024.0 * 1024.0 * 1024.0)).toFloat().round(1)
        val totalGb = (totalMem / (1024.0 * 1024.0 * 1024.0)).toFloat().round(1)

        MemoryMetrics(
            availableMem = availableMem,
            totalMem = totalMem,
            percentUsed = percentUsed,
            availableGb = availableGb,
            totalGb = totalGb
        )
    }

    /**
     * Gets the current battery information.
     */
    private fun getBatteryInfo(): BatteryMetrics {
        val intent = context.registerReceiver(null,
            IntentFilter(Intent.ACTION_BATTERY_CHANGED))

        val level = intent?.getIntExtra(BatteryManager.EXTRA_LEVEL, 0) ?: 0
        val scale = intent?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100
        val batteryPct = level * 100 / scale

        val status = intent?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
        val isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
            status == BatteryManager.BATTERY_STATUS_FULL

        return BatteryMetrics(
            level = batteryPct,
            isCharging = isCharging
        )
    }

    /**
     * Gets the current temperature information.
     */
    private fun getTemperatureInfo(): TemperatureMetrics {
        val intent = context.registerReceiver(null,
            IntentFilter(Intent.ACTION_BATTERY_CHANGED))

        // Battery temperature is reported in tenths of a degree Celsius
        val tempTenthsC = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
        val tempC = tempTenthsC / 10.0f

        val warningLevel = when {
            tempC >= 45.0f -> TemperatureWarningLevel.HIGH
            tempC >= 40.0f -> TemperatureWarningLevel.MEDIUM
            else -> TemperatureWarningLevel.NORMAL
        }

        return TemperatureMetrics(
            temperature = tempC,
            warningLevel = warningLevel
        )
    }

    private fun Float.round(decimals: Int): Float {
        var multiplier = 1.0f
        repeat(decimals) { multiplier *= 10 }
        return (this * multiplier).roundToInt() / multiplier
    }

    companion object {
        private const val MEMORY_POLLING_INTERVAL = 5000L
        private const val BATTERY_POLLING_INTERVAL = 10000L
        private const val TEMP_POLLING_INTERVAL = 10000L
    }
}

/**
 * Data class containing memory usage metrics.
 */
data class MemoryMetrics(
    val availableMem: Long,
    val totalMem: Long,
    val percentUsed: Int,
    val availableGb: Float,
    val totalGb: Float
)

/**
 * Data class containing battery information.
 */
data class BatteryMetrics(
    val level: Int,
    val isCharging: Boolean
)

/**
 * Warning levels for temperature.
 */
enum class TemperatureWarningLevel {
    NORMAL,
    MEDIUM,
    HIGH
}

/**
 * Data class containing temperature information.
 */
data class TemperatureMetrics(
    val temperature: Float,
    val warningLevel: TemperatureWarningLevel
)
