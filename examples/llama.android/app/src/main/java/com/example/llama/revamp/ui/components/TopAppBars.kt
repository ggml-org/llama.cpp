package com.example.llama.revamp.ui.components

import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.SdStorage
import androidx.compose.material.icons.filled.Storage
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

// TopAppBars.kt
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DefaultTopBar(
    title: String,
    onNavigateBack: (() -> Unit)? = null,
    onMenuOpen: (() -> Unit)? = null
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
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = MaterialTheme.colorScheme.surface,
            titleContentColor = MaterialTheme.colorScheme.onSurface
        )
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PerformanceTopBar(
    title: String,
    memoryMetrics: MemoryMetrics,
    temperatureMetrics: TemperatureMetrics,
    onNavigateBack: (() -> Unit)? = null,
    onMenuOpen: (() -> Unit)? = null,
    showTemperature: Boolean = false,
    useFahrenheit: Boolean = false,
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
            if (showTemperature) {
                TemperatureIndicator(
                    temperature = temperatureMetrics,
                    useFahrenheit = useFahrenheit
                )

                Spacer(modifier = Modifier.width(8.dp))
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

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun StorageTopBar(
    title: String,
    storageUsed: Float,
    storageTotal: Float,
    onNavigateBack: (() -> Unit)? = null,
) {
    TopAppBar(
        title = { Text(title) },
        navigationIcon = {
            if (onNavigateBack != null) {
                IconButton(onClick = onNavigateBack) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back"
                    )
                }
            }
        },
        actions = {
            StorageIndicator(usedGB = storageUsed, totalGB = storageTotal)
        },
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = MaterialTheme.colorScheme.surface,
            titleContentColor = MaterialTheme.colorScheme.onSurface
        )
    )
}

@Composable
fun MemoryIndicator(memoryUsage: MemoryMetrics) {
    Row(modifier = Modifier.padding(end = 8.dp), verticalAlignment = Alignment.CenterVertically) {
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

        val memoryText = String.format("%.1f / %.1f GB", memoryUsage.availableGb, memoryUsage.totalGb)

        Text(
            text = memoryText,
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

@Composable
fun TemperatureIndicator(temperature: TemperatureMetrics, useFahrenheit: Boolean) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Icon(
            imageVector = when (temperature.warningLevel) {
                TemperatureWarningLevel.HIGH -> Icons.Default.WarningAmber
                else -> Icons.Default.Thermostat
            },
            contentDescription = "Device temperature",
            tint = when (temperature.warningLevel) {
                TemperatureWarningLevel.HIGH -> MaterialTheme.colorScheme.error
                TemperatureWarningLevel.MEDIUM -> MaterialTheme.colorScheme.tertiary
                else -> MaterialTheme.colorScheme.onSurface
            }
        )

        Spacer(modifier = Modifier.width(2.dp))

        val tempDisplay = if (useFahrenheit) temperature.fahrenheitDisplay else temperature.celsiusDisplay

        Text(
            text = tempDisplay,
            style = MaterialTheme.typography.bodySmall,
            color = when (temperature.warningLevel) {
                TemperatureWarningLevel.HIGH -> MaterialTheme.colorScheme.error
                TemperatureWarningLevel.MEDIUM -> MaterialTheme.colorScheme.tertiary
                else -> MaterialTheme.colorScheme.onSurface
            }
        )
    }
}

@Composable
fun StorageIndicator(usedGB: Float, totalGB: Float) {
    Row(modifier = Modifier.padding(end = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Icon(
            imageVector = Icons.Default.SdStorage,
            contentDescription = "Storage",
            tint = when {
                usedGB / totalGB > 0.9f -> MaterialTheme.colorScheme.error
                usedGB / totalGB > 0.7f -> MaterialTheme.colorScheme.tertiary
                else -> MaterialTheme.colorScheme.onSurface
            }
        )

        Spacer(modifier = Modifier.width(2.dp))

        Text(
            text = String.format("%.1f / %.1f GB", usedGB, totalGB),
            style = MaterialTheme.typography.bodySmall
        )
    }
}
