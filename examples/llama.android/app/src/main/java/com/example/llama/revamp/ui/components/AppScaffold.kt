package com.example.llama.revamp.ui.components

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.DrawerState
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.llama.revamp.data.preferences.UserPreferences
import com.example.llama.revamp.monitoring.PerformanceMonitor
import com.example.llama.revamp.navigation.NavigationActions
import com.example.llama.revamp.util.ViewModelFactoryProvider
import com.example.llama.revamp.viewmodel.PerformanceViewModel

/**
 * Main scaffold for the app that provides the top bar with system status
 * and wraps content in a consistent layout.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppScaffold(
    title: String,
    navigationActions: NavigationActions,
    drawerState: DrawerState = rememberDrawerState(initialValue = DrawerValue.Closed),
    snackbarHostState: SnackbarHostState = remember { SnackbarHostState() },
    onBackPressed: (() -> Unit)? = null,
    onMenuPressed: (() -> Unit)? = null,
    onRerunPressed: (() -> Unit)? = null,
    onSharePressed: (() -> Unit)? = null,
    content: @Composable (PaddingValues) -> Unit
) {
    // Create dependencies for PerformanceViewModel
    val context = LocalContext.current
    val performanceMonitor = remember { PerformanceMonitor(context) }
    val userPreferences = remember { UserPreferences(context) }

    // Create factory for PerformanceViewModel
    val factory = remember { ViewModelFactoryProvider.getPerformanceViewModelFactory(performanceMonitor, userPreferences) }

    // Get ViewModel instance with factory
    val performanceViewModel: PerformanceViewModel = viewModel(factory = factory)

    // Collect performance metrics
    val memoryUsage by performanceViewModel.memoryUsage.collectAsState()
    val batteryInfo by performanceViewModel.batteryInfo.collectAsState()
    val temperatureInfo by performanceViewModel.temperatureInfo.collectAsState()
    val useFahrenheit by performanceViewModel.useFahrenheitUnit.collectAsState()

    // Formatted memory usage
    val memoryText = "${memoryUsage.availableGb}GB available"

    AppNavigationDrawer(
        drawerState = drawerState,
        navigationActions = navigationActions
    ) {
        Scaffold(
            topBar = {
                SystemStatusTopBar(
                    title = title,
                    memoryUsage = memoryText,
                    batteryLevel = batteryInfo.level,
                    temperature = temperatureInfo.temperature,
                    useFahrenheit = useFahrenheit,
                    onBackPressed = onBackPressed,
                    onMenuPressed = onMenuPressed,
                    onRerunPressed = onRerunPressed,
                    onSharePressed = onSharePressed
                )
            },
            snackbarHost = {
                SnackbarHost(hostState = snackbarHostState)
            },
            content = content
        )
    }
}
