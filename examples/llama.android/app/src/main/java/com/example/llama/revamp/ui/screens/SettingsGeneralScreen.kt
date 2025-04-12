package com.example.llama.revamp.ui.screens

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
import androidx.compose.material3.DrawerState
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
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
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.llama.revamp.data.preferences.UserPreferences
import com.example.llama.revamp.monitoring.PerformanceMonitor
import com.example.llama.revamp.navigation.NavigationActions
import com.example.llama.revamp.ui.components.AppScaffold
import com.example.llama.revamp.util.ViewModelFactoryProvider
import com.example.llama.revamp.viewmodel.PerformanceViewModel

/**
 * Screen for general app settings
 */
@Composable
fun SettingsGeneralScreen(
    onBackPressed: () -> Unit,
    drawerState: DrawerState,
    navigationActions: NavigationActions,
    onMenuClicked: () -> Unit
) {
    // Create dependencies for PerformanceViewModel
    val context = LocalContext.current
    val performanceMonitor = remember { PerformanceMonitor(context) }
    val userPreferences = remember { UserPreferences(context) }

    // Create factory for PerformanceViewModel
    val factory = remember { ViewModelFactoryProvider.getPerformanceViewModelFactory(performanceMonitor, userPreferences) }

    // Get ViewModel instance with factory
    val performanceViewModel: PerformanceViewModel = viewModel(factory = factory)

    // Collect state from ViewModel
    val isMonitoringEnabled by performanceViewModel.isMonitoringEnabled.collectAsState()
    val useFahrenheit by performanceViewModel.useFahrenheitUnit.collectAsState()

    AppScaffold(
        title = "Settings",
        navigationActions = navigationActions,
        onBackPressed = onBackPressed,
        onMenuPressed = onMenuClicked
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp)
                .verticalScroll(rememberScrollState())
        ) {
            SettingsCategory(title = "Performance Monitoring") {
                SettingsSwitch(
                    title = "Enable Monitoring",
                    description = "Display memory, battery and temperature information",
                    checked = isMonitoringEnabled,
                    onCheckedChange = { performanceViewModel.setMonitoringEnabled(it) }
                )

                SettingsSwitch(
                    title = "Use Fahrenheit",
                    description = "Display temperature in Fahrenheit instead of Celsius",
                    checked = useFahrenheit,
                    onCheckedChange = { performanceViewModel.setUseFahrenheitUnit(it) }
                )
            }

            SettingsCategory(title = "Theme") {
                SettingsSwitch(
                    title = "Dark Theme",
                    description = "Use dark theme throughout the app",
                    checked = true, // This would be connected to theme state in a real app
                    onCheckedChange = { /* TODO: Implement theme switching */ }
                )
            }

            SettingsCategory(title = "About") {
                Card(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Text(
                            text = "Local LLM",
                            style = MaterialTheme.typography.titleLarge
                        )

                        Text(
                            text = "Version 1.0.0",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )

                        Spacer(modifier = Modifier.height(8.dp))

                        Text(
                            text = "Local inference for LLM models on your device.",
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun SettingsCategory(
    title: String,
    content: @Composable () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp)
    ) {
        Text(
            text = title,
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(bottom = 8.dp)
        )

        Card(
            modifier = Modifier.fillMaxWidth()
        ) {
            Column {
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
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp)
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

    HorizontalDivider()
}
