package com.example.llama.ui.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.llama.APP_NAME
import com.example.llama.data.preferences.UserPreferences
import com.example.llama.ui.components.ArmFeaturesVisualizerClickable
import com.example.llama.viewmodel.SettingsViewModel

/**
 * Screen for general app settings
 */
@Composable
fun SettingsGeneralScreen(
    viewModel: SettingsViewModel,
) {
    // Collect state from ViewModel
    val isMonitoringEnabled by viewModel.isMonitoringEnabled.collectAsState()
    val useFahrenheit by viewModel.useFahrenheitUnit.collectAsState()
    val themeMode by viewModel.themeMode.collectAsState()
    val detectedTier = viewModel.detectedTier

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState())
    ) {
        SettingsCategory(title = "Performance Monitoring") {
            SettingsSwitch(
                title = "Enable Monitoring",
                description = "Display memory, battery and temperature info",
                checked = isMonitoringEnabled,
                onCheckedChange = { viewModel.setMonitoringEnabled(it) }
            )

            Spacer(modifier = Modifier.height(8.dp))

            HorizontalDivider()

            Spacer(modifier = Modifier.height(8.dp))

            SettingsSwitch(
                title = "Use Fahrenheit",
                description = "Display temperature in Fahrenheit instead of Celsius",
                checked = useFahrenheit,
                onCheckedChange = { viewModel.setUseFahrenheitUnit(it) }
            )
        }

        SettingsCategory(title = "Theme") {
            Text(
                text = "Theme Mode",
                style = MaterialTheme.typography.titleMedium
            )

            Text(
                text = "Follow system setting or override with your choice",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(modifier = Modifier.height(16.dp))

            SingleChoiceSegmentedButtonRow(
                modifier = Modifier.fillMaxWidth()
            ) {
                SegmentedButton(
                    selected = themeMode == UserPreferences.THEME_MODE_AUTO,
                    onClick = { viewModel.setThemeMode(UserPreferences.THEME_MODE_AUTO) },
                    shape = SegmentedButtonDefaults.itemShape(index = 0, count = 3)
                ) {
                    Text("Auto")
                }

                SegmentedButton(
                    selected = themeMode == UserPreferences.THEME_MODE_LIGHT,
                    onClick = { viewModel.setThemeMode(UserPreferences.THEME_MODE_LIGHT) },
                    shape = SegmentedButtonDefaults.itemShape(index = 1, count = 3)
                ) {
                    Text("Light")
                }

                SegmentedButton(
                    selected = themeMode == UserPreferences.THEME_MODE_DARK,
                    onClick = { viewModel.setThemeMode(UserPreferences.THEME_MODE_DARK) },
                    shape = SegmentedButtonDefaults.itemShape(index = 2, count = 3)
                ) {
                    Text("Dark")
                }
            }
        }

        // ARM Features Visualizer with Tier Information description
        SettingsCategory(title = "About your device") {
            Text(
                text = "ARM Capabilities",
                style = MaterialTheme.typography.titleMedium
            )

            Text(
                text = "Hardware-accelerated AI features",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            detectedTier?.let { tier ->
                Spacer(modifier = Modifier.height(8.dp))

                ArmFeaturesVisualizerClickable(detectedTier = detectedTier)

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = "Optimization Tier: ${tier.name}",
                    style = MaterialTheme.typography.titleMedium
                )

                Text(
                    text = tier.description,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
        }

        SettingsCategory(title = "About this app") {
            Text(
                text = APP_NAME,
                style = MaterialTheme.typography.titleMedium
            )

            Text(
                text = "Version 1.0.0",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Local inference for LLM models on your device powered by Arm® technologies.",
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}

@Composable
fun SettingsCategory(
    title: String,
    content: @Composable () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)
    ) {
        Text(
            text = title,
            style = MaterialTheme.typography.labelLarge,
            modifier = Modifier.padding(bottom = 8.dp)
        )

        Card(modifier = Modifier.fillMaxWidth()) {
            Column( modifier = Modifier.fillMaxWidth().padding(16.dp)) {
                content()
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
    }
}

@Composable
fun SettingsSwitch(
    title: String,
    description: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(
            modifier = Modifier.weight(1f)
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium
            )

            Text(
                text = description,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange
        )
    }
}
