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
import androidx.compose.material3.Divider
import androidx.compose.material3.DrawerState
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
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
 * Tabs for the settings screen.
 */
enum class SettingsTab {
    GENERAL,
    MODEL_MANAGEMENT,
    ADVANCED
}

@Composable
fun SettingsScreen(
    selectedTab: SettingsTab = SettingsTab.GENERAL,
    onBackPressed: () -> Unit,
    drawerState: DrawerState,
    navigationActions: NavigationActions
) {
    // State for tab selection
    var currentTab by remember { mutableStateOf(selectedTab) }

    // Create dependencies for PerformanceViewModel
    val context = LocalContext.current
    val performanceMonitor = remember { PerformanceMonitor(context) }
    val userPreferences = remember { UserPreferences(context) }

    // Create factory for PerformanceViewModel
    val factory = remember { ViewModelFactoryProvider.getPerformanceViewModelFactory(performanceMonitor, userPreferences) }

    // Get ViewModel instance with factory
    val performanceViewModel: PerformanceViewModel = viewModel(factory = factory)

    AppScaffold(
        title = "Settings",
        drawerState = drawerState,
        navigationActions = navigationActions,
        onBackPressed = onBackPressed
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
        ) {
            // Tabs for different settings categories
            TabRow(selectedTabIndex = currentTab.ordinal) {
                Tab(
                    selected = currentTab == SettingsTab.GENERAL,
                    onClick = { currentTab = SettingsTab.GENERAL },
                    text = { Text("General") }
                )

                Tab(
                    selected = currentTab == SettingsTab.MODEL_MANAGEMENT,
                    onClick = { currentTab = SettingsTab.MODEL_MANAGEMENT },
                    text = { Text("Models") }
                )

                Tab(
                    selected = currentTab == SettingsTab.ADVANCED,
                    onClick = { currentTab = SettingsTab.ADVANCED },
                    text = { Text("Advanced") }
                )
            }

            // Content for the selected tab
            when (currentTab) {
                SettingsTab.GENERAL -> GeneralSettings(performanceViewModel)
                SettingsTab.MODEL_MANAGEMENT -> ModelManagementSettings()
                SettingsTab.ADVANCED -> AdvancedSettings()
            }
        }
    }
}

@Composable
fun GeneralSettings(performanceViewModel: PerformanceViewModel) {
    val isMonitoringEnabled by performanceViewModel.isMonitoringEnabled.collectAsState()
    val useFahrenheit by performanceViewModel.useFahrenheitUnit.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxWidth()
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
    }
}

@Composable
fun ModelManagementSettings() {
    // This would be populated with actual functionality in a real implementation
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp)
            .verticalScroll(rememberScrollState())
    ) {
        Text(
            text = "Model Management",
            style = MaterialTheme.typography.titleLarge,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        Text(
            text = "This section will allow you to download, delete, and manage LLM models.",
            style = MaterialTheme.typography.bodyMedium
        )
    }
}

@Composable
fun AdvancedSettings() {
    // This would be populated with actual functionality in a real implementation
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp)
            .verticalScroll(rememberScrollState())
    ) {
        Text(
            text = "Advanced Settings",
            style = MaterialTheme.typography.titleLarge,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        Text(
            text = "This section will contain advanced settings such as memory management, cache configuration, and debugging options.",
            style = MaterialTheme.typography.bodyMedium
        )
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

    Divider()
}
