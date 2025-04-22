package com.example.llama.revamp.ui.scaffold.topbar

import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
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
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.llama.revamp.monitoring.MemoryMetrics
import com.example.llama.revamp.monitoring.TemperatureMetrics
import com.example.llama.revamp.monitoring.TemperatureWarningLevel
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PerformanceTopBar(
    title: String,
    memoryMetrics: MemoryMetrics,
    temperatureDisplay: Pair<TemperatureMetrics, Boolean>?,
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
                    useFahrenheit = useFahrenheit
                )

                Spacer(modifier = Modifier.Companion.width(8.dp))
            }

            // Memory indicator
            MemoryIndicator(memoryUsage = memoryMetrics)
        },
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = MaterialTheme.colorScheme.surface,
            titleContentColor = MaterialTheme.colorScheme.onSurface
        )
    )
}

@Composable
private fun MemoryIndicator(memoryUsage: MemoryMetrics) {
    Row(
        modifier = Modifier.Companion.padding(end = 8.dp),
        verticalAlignment = Alignment.Companion.CenterVertically
    ) {
        Icon(
            imageVector = Icons.Default.Memory,
            contentDescription = "RAM usage",
            tint = when {
                memoryUsage.availableGb < 1 -> MaterialTheme.colorScheme.error
                memoryUsage.availableGb < 3 -> MaterialTheme.colorScheme.tertiary
                else -> MaterialTheme.colorScheme.onSurface
            }
        )

        Spacer(modifier = Modifier.width(4.dp))

        Text(
            text = String.Companion.format(
                Locale.getDefault(), "%.1f / %.1f GB", memoryUsage.availableGb, memoryUsage.totalGb
            ),
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

@Composable
private fun TemperatureIndicator(temperatureMetrics: TemperatureMetrics, useFahrenheit: Boolean) {
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

        Spacer(modifier = Modifier.width(2.dp))

        Text(
            text = temperatureMetrics.getDisplay(useFahrenheit),
            style = MaterialTheme.typography.bodySmall,
            color = when (temperatureMetrics.warningLevel) {
                TemperatureWarningLevel.HIGH -> MaterialTheme.colorScheme.error
                TemperatureWarningLevel.MEDIUM -> MaterialTheme.colorScheme.tertiary
                else -> MaterialTheme.colorScheme.onSurface
            }
        )
    }
}
