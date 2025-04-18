package com.example.llama.revamp.ui.components

import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.SdStorage
import androidx.compose.material.icons.filled.Thermostat
import androidx.compose.material.icons.filled.WarningAmber
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Label
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.llama.revamp.data.repository.StorageMetrics
import com.example.llama.revamp.monitoring.MemoryMetrics
import com.example.llama.revamp.monitoring.TemperatureMetrics
import com.example.llama.revamp.monitoring.TemperatureWarningLevel
import java.util.Locale

/**
 * [TopAppBar] configurations
 */
sealed class TopBarConfig {
    abstract val title: String
    abstract val navigationIcon: NavigationIcon

    // Default/simple top bar with only a navigation icon
    data class Default(
        override val title: String,
        override val navigationIcon: NavigationIcon
    ) : TopBarConfig()

    // Performance monitoring top bar with RAM and optional temperature
    data class Performance(
        override val title: String,
        override val navigationIcon: NavigationIcon,
        val memoryMetrics: MemoryMetrics,
        val temperatureInfo: Pair<TemperatureMetrics, Boolean>?,
    ) : TopBarConfig()

    // Storage management top bar with used & total storage
    data class Storage(
        override val title: String,
        override val navigationIcon: NavigationIcon,
        val storageMetrics: StorageMetrics?
    ) : TopBarConfig()
}

/**
 * Helper class for navigation icon configuration
 */
sealed class NavigationIcon {
    data class Back(val onNavigateBack: () -> Unit) : NavigationIcon()
    data class Menu(val onMenuOpen: () -> Unit) : NavigationIcon()
    object None : NavigationIcon()
}

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
    storageMetrics: StorageMetrics?,
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
            storageMetrics?.let {
                StorageIndicator(storageMetrics = it)
            }
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

        Text(
            text = String.format(
                Locale.getDefault(), "%.1f / %.1f GB", memoryUsage.availableGb, memoryUsage.totalGb
            ),
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

@Composable
fun TemperatureIndicator(temperatureMetrics: TemperatureMetrics, useFahrenheit: Boolean) {
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

@Composable
fun StorageIndicator(storageMetrics: StorageMetrics) {
    val usedGb = storageMetrics.usedGB
    val availableGb = storageMetrics.availableGB

    Row(modifier = Modifier.padding(end = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        Icon(
            imageVector = Icons.Default.SdStorage,
            contentDescription = "Storage",
            tint = when {
                availableGb < 5.0f -> MaterialTheme.colorScheme.error
                availableGb < 10.0f -> MaterialTheme.colorScheme.tertiary
                else -> MaterialTheme.colorScheme.onSurface
            }
        )

        Spacer(modifier = Modifier.width(2.dp))

        Text(
            text = String.format(Locale.getDefault(), "%.1f / %.1f GB", usedGb, availableGb),
            style = MaterialTheme.typography.bodySmall
        )
    }
}
