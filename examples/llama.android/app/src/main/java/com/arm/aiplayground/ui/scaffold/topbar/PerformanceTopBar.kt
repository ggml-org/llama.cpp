package com.arm.aiplayground.ui.scaffold.topbar

import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.Thermostat
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.arm.aiplayground.monitoring.MemoryMetrics
import com.arm.aiplayground.monitoring.TemperatureMetrics
import com.arm.aiplayground.monitoring.TemperatureWarningLevel
import com.arm.aiplayground.ui.scaffold.ScaffoldEvent
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PerformanceTopBar(
    title: String,
    memoryMetrics: MemoryMetrics?,
    temperatureDisplay: Pair<TemperatureMetrics, Boolean>?,
    onScaffoldEvent: (ScaffoldEvent) -> Unit,
    onNavigateBack: (() -> Unit)? = null,
    onMenuOpen: (() -> Unit)? = null,
) {
    TopAppBar(
        title = { Text(title) },
        navigationIcon = {
            when {
                onNavigateBack != null -> {
                    IconButton(onClick = onNavigateBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                }

                onMenuOpen != null -> {
                    IconButton(onClick = onMenuOpen) {
                        Icon(
                            imageVector = Icons.Default.Menu,
                            contentDescription = "Menu"
                        )
                    }
                }
            }
        },
        actions = {
            // Temperature indicator (optional)
            temperatureDisplay?.let { (temperatureMetrics, useFahrenheit) ->
                TemperatureIndicator(
                    temperatureMetrics = temperatureMetrics,
                    useFahrenheit = useFahrenheit,
                    onScaffoldEvent = onScaffoldEvent,
                )
            }

            // Memory indicator
            memoryMetrics?.let {
                MemoryIndicator(memoryUsage = it, onScaffoldEvent = onScaffoldEvent)
            }
        },
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = MaterialTheme.colorScheme.surface,
            titleContentColor = MaterialTheme.colorScheme.onSurface
        )
    )
}

@Composable
private fun MemoryIndicator(
    memoryUsage: MemoryMetrics,
    onScaffoldEvent: (ScaffoldEvent) -> Unit,
) {
    val availableGB = String.format(Locale.getDefault(), "%.1f", memoryUsage.availableGB)
    val totalGB = String.format(Locale.getDefault(), "%.1f", memoryUsage.totalGB)

    OutlinedButton(
        modifier = Modifier.padding(end = 8.dp),
        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp),
        onClick = {
            onScaffoldEvent(ScaffoldEvent.ShowSnackbar(
                message = "Free RAM available: $availableGB GB\nTotal RAM on your device: $totalGB GB",
                withDismissAction = true,
            ))
        }
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                imageVector = Icons.Default.Memory,
                contentDescription = "RAM usage",
                tint = when {
                    memoryUsage.availableGB < 1 -> MaterialTheme.colorScheme.error
                    memoryUsage.availableGB < 3 -> MaterialTheme.colorScheme.tertiary
                    else -> MaterialTheme.colorScheme.onSurface
                }
            )

            Text(
                modifier = Modifier.padding(start = 2.dp),
                text =  "$availableGB / $totalGB GB",
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

@Composable
private fun TemperatureIndicator(
    temperatureMetrics: TemperatureMetrics,
    useFahrenheit: Boolean,
    onScaffoldEvent: (ScaffoldEvent) -> Unit,
) {
    val temperatureDisplay = temperatureMetrics.getDisplay(useFahrenheit)

    val temperatureWarning = when (temperatureMetrics.warningLevel) {
        TemperatureWarningLevel.HIGH -> "Your device is HEATED UP to $temperatureDisplay, please cool it down before continue using the app."
        TemperatureWarningLevel.MEDIUM -> "Your device is warming up to $temperatureDisplay."
        else -> "Your device's temperature is $temperatureDisplay."
    }
    val warningDismissible = temperatureMetrics.warningLevel != TemperatureWarningLevel.HIGH

    OutlinedButton(
        modifier = Modifier.padding(end = 8.dp),
        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp),
        onClick = {
            onScaffoldEvent(ScaffoldEvent.ShowSnackbar(
                message = temperatureWarning,
                withDismissAction = warningDismissible,
            ))
        }
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                imageVector = when (temperatureMetrics.warningLevel) {
                    TemperatureWarningLevel.HIGH -> Icons.Default.WarningAmber
                    else -> Icons.Default.Thermostat
                },
                contentDescription = "Device temperature",
                tint = when (temperatureMetrics.warningLevel) {
                    TemperatureWarningLevel.HIGH -> MaterialTheme.colorScheme.error
                    TemperatureWarningLevel.MEDIUM -> MaterialTheme.colorScheme.tertiary
                    else -> MaterialTheme.colorScheme.onSurface
                }
            )

            Text(
                modifier = Modifier.padding(start = 4.dp),
                text = temperatureDisplay,
                style = MaterialTheme.typography.bodySmall,
                color = when (temperatureMetrics.warningLevel) {
                    TemperatureWarningLevel.HIGH -> MaterialTheme.colorScheme.error
                    TemperatureWarningLevel.MEDIUM -> MaterialTheme.colorScheme.tertiary
                    else -> MaterialTheme.colorScheme.onSurface
                }
            )
        }
    }
}
