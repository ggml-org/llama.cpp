package com.arm.aiplayground.monitoring

/**
 * Data class containing battery information.
 */
data class BatteryMetrics(
    val level: Int,
    val isCharging: Boolean
)

/**
 * Data class containing memory usage metrics.
 */
data class MemoryMetrics(
    val availableMem: Long,
    val totalMem: Long,
    val percentUsed: Int,
    val availableGB: Float,
    val totalGB: Float
)

/**
 * Data class containing temperature information.
 */
data class TemperatureMetrics(
    private val tempCelsiusValue: Float,
    val warningLevel: TemperatureWarningLevel
) {
    val celsiusDisplay: String
        get() = "${tempCelsiusValue.toInt()}°C"

    val fahrenheitDisplay: String
        get() = "${(tempCelsiusValue * 9/5 + 32).toInt()}°F"

    fun getDisplay(useFahrenheit: Boolean) =
        if (useFahrenheit) fahrenheitDisplay else celsiusDisplay
}

enum class TemperatureWarningLevel {
    NORMAL,
    MEDIUM,
    HIGH
}

/**
 * Data class containing storage usage metrics.
 */
data class StorageMetrics(
    val usedGB: Float,
    val availableGB: Float
)
