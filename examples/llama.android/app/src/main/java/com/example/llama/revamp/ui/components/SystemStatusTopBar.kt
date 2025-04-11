package com.example.llama.revamp.ui.components

import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.BatteryAlert
import androidx.compose.material.icons.filled.BatteryFull
import androidx.compose.material.icons.filled.BatteryStd
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

/**
 * Top app bar that displays system status information and navigation controls.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SystemStatusTopBar(
    title: String,
    memoryUsage: String,
    batteryLevel: Int,
    temperature: Float,
    useFahrenheit: Boolean = false,
    onBackPressed: (() -> Unit)? = null,
    onMenuPressed: (() -> Unit)? = null,
    onRerunPressed: (() -> Unit)? = null,
    onSharePressed: (() -> Unit)? = null
) {
    CenterAlignedTopAppBar(
        title = { Text(title) },
        navigationIcon = {
            when {
                onBackPressed != null -> {
                    IconButton(onClick = onBackPressed) {
                        Icon(
                            imageVector = Icons.Default.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                }
                onMenuPressed != null -> {
                    IconButton(onClick = onMenuPressed) {
                        Icon(
                            imageVector = Icons.Default.Menu,
                            contentDescription = "Menu"
                        )
                    }
                }
            }
        },
        actions = {
            // Memory usage
            Text(
                text = memoryUsage,
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(end = 8.dp)
            )

            // Battery and temperature
            Row(verticalAlignment = Alignment.CenterVertically) {
                // Battery icon and percentage
                Icon(
                    imageVector = when {
                        batteryLevel > 70 -> Icons.Default.BatteryFull
                        batteryLevel > 30 -> Icons.Default.BatteryStd
                        else -> Icons.Default.BatteryAlert
                    },
                    contentDescription = "Battery level",
                    tint = when {
                        batteryLevel <= 15 -> MaterialTheme.colorScheme.error
                        else -> MaterialTheme.colorScheme.onSurface
                    }
                )

                Text(
                    text = "$batteryLevel%",
                    style = MaterialTheme.typography.bodySmall
                )

                Spacer(modifier = Modifier.width(8.dp))

                // Temperature display
                val tempDisplay = if (useFahrenheit) {
                    "${(temperature * 9/5 + 32).toInt()}°F"
                } else {
                    "${temperature.toInt()}°C"
                }

                val tempTint = when {
                    temperature >= 45 -> MaterialTheme.colorScheme.error
                    temperature >= 40 -> Color(0xFFFFA500) // Orange warning color
                    else -> MaterialTheme.colorScheme.onSurface
                }

                Text(
                    text = tempDisplay,
                    style = MaterialTheme.typography.bodySmall,
                    color = tempTint
                )
            }

            // Optional action buttons
            onRerunPressed?.let {
                IconButton(onClick = it) {
                    Icon(
                        imageVector = Icons.Default.Refresh,
                        contentDescription = "Rerun benchmark"
                    )
                }
            }

            onSharePressed?.let {
                IconButton(onClick = it) {
                    Icon(
                        imageVector = Icons.Default.Share,
                        contentDescription = "Share results"
                    )
                }
            }
        },
        colors = TopAppBarDefaults.centerAlignedTopAppBarColors(
            containerColor = MaterialTheme.colorScheme.surface,
            titleContentColor = MaterialTheme.colorScheme.onSurface
        )
    )
}
