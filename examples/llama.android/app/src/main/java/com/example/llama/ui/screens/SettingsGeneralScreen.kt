package com.example.llama.ui.screens

import android.content.Intent
import android.llama.cpp.ArmFeaturesMapper
import android.llama.cpp.ArmFeaturesMapper.DisplayItem
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
import androidx.compose.material3.ExperimentalMaterial3Api
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
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.net.toUri
import com.example.llama.APP_NAME
import com.example.llama.BuildConfig
import com.example.llama.data.source.prefs.ColorThemeMode
import com.example.llama.data.source.prefs.DarkThemeMode
import com.example.llama.ui.components.ArmFeaturesVisualizer
import com.example.llama.viewmodel.SettingsViewModel
import kotlin.math.sqrt

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
    val colorThemeMode by viewModel.colorThemeMode.collectAsState()
    val darkThemeMode by viewModel.darkThemeMode.collectAsState()
    val detectedTier = viewModel.detectedTier

    val supportedFeatures = remember(detectedTier) {
        ArmFeaturesMapper.getFeatureDisplayData(detectedTier)
    }

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

            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

            SettingsSwitch(
                title = "Use Fahrenheit",
                description = "Display temperature in Fahrenheit instead of Celsius",
                checked = useFahrenheit,
                onCheckedChange = { viewModel.setUseFahrenheitUnit(it) }
            )
        }

        SettingsCategory(title = "Styling") {
            // Color theme mode
            Text(
                text = "Color Theme",
                style = MaterialTheme.typography.titleMedium
            )

            Text(
                text = "Arm® or follow your system dynamic colors",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(modifier = Modifier.height(16.dp))

            SingleChoiceSegmentedButtonRow(
                modifier = Modifier.fillMaxWidth()
            ) {
                ColorThemeMode.entries.forEachIndexed { index, mode ->
                    val weight = sqrt(sqrt(mode.label.length.toFloat()))

                    SegmentedButton(
                        modifier = Modifier.weight(weight),
                        selected = colorThemeMode == mode,
                        onClick = { viewModel.setColorThemeMode(mode) },
                        shape = SegmentedButtonDefaults.itemShape(
                            index = index,
                            count = ColorThemeMode.entries.size
                        )
                    ) {
                        Text(mode.label)
                    }
                }
            }

            HorizontalDivider(modifier = Modifier.padding(vertical = 12.dp))

            // Dark theme mode
            Text(
                text = "Dark Mode",
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
                DarkThemeMode.entries.forEachIndexed { index, mode ->
                    SegmentedButton(
                        selected = darkThemeMode == mode,
                        onClick = { viewModel.setDarkThemeMode(mode) },
                        shape = SegmentedButtonDefaults.itemShape(index = index, count = DarkThemeMode.entries.size)
                    ) {
                        Text(mode.label)
                    }
                }
            }
        }

        // ARM Features Visualizer with Tier Information description
        detectedTier?.let { tier ->
            SettingsCategory(title = "About your device") {
                Text(
                    text = "AI Accelerated by Arm®",
                    style = MaterialTheme.typography.titleMedium
                )

                Text(
                    text = "Available hardware capabilities on your device are highlighted below:",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(vertical = 8.dp)
                )

                supportedFeatures?.let {
                    ArmFeaturesVisualizerClickable(supportedFeatures = it)
                }

                Text(
                    text = "Tap a feature above to learn more about how it accelerates Generative AI!",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(vertical = 8.dp)
                )
            }
        }

        SettingsCategory(title = "About this app") {
            Text(
                text = APP_NAME,
                style = MaterialTheme.typography.titleMedium
            )

            Text(
                text = "Version ${BuildConfig.VERSION_NAME}",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(vertical = 4.dp)
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Run Large Language Models locally at your fingertips, harness the power of mobile AI with Arm®.",
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

/**
 * Alternative version with clickable features that open ARM documentation.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ArmFeaturesVisualizerClickable(
    supportedFeatures: List<DisplayItem>,
) {
    val context = LocalContext.current

    ArmFeaturesVisualizer(
        supportedFeatures = supportedFeatures,
        onFeatureClick = { feature ->
            // Open ARM documentation in browser
            val intent = Intent(Intent.ACTION_VIEW, feature.armDocUrl.toUri())
            context.startActivity(intent)
        }
    )
}
