package com.example.llama.revamp.ui.components

import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.llama.revamp.data.preferences.UserPreferences
import com.example.llama.revamp.monitoring.PerformanceMonitor
import com.example.llama.revamp.util.ViewModelFactoryProvider
import com.example.llama.revamp.viewmodel.PerformanceViewModel

// DefaultAppScaffold.kt
@Composable
fun DefaultAppScaffold(
    title: String,
    onNavigateBack: (() -> Unit)? = null,
    onMenuOpen: (() -> Unit)? = null,
    snackbarHostState: SnackbarHostState = remember { SnackbarHostState() },
    content: @Composable (PaddingValues) -> Unit
) {
    Scaffold(
        topBar = {
            DefaultTopBar(
                title = title,
                onNavigateBack = onNavigateBack,
                onMenuOpen = onMenuOpen
            )
        },
        snackbarHost = {
            SnackbarHost(hostState = snackbarHostState)
        },
        content = content
    )
}

// PerformanceAppScaffold.kt
@Composable
fun PerformanceAppScaffold(
    title: String,
    onNavigateBack: (() -> Unit)? = null,
    onMenuOpen: (() -> Unit)? = null,
    showTemperature: Boolean = false,
    snackbarHostState: SnackbarHostState = remember { SnackbarHostState() },
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
    val temperatureInfo by performanceViewModel.temperatureMetrics.collectAsState()
    val useFahrenheit by performanceViewModel.useFahrenheitUnit.collectAsState()

    Scaffold(
        topBar = {
            PerformanceTopBar(
                title = title,
                memoryMetrics = memoryUsage,
                temperatureDisplay = if (showTemperature) Pair(temperatureInfo, useFahrenheit) else null,
                onNavigateBack = onNavigateBack,
                onMenuOpen = onMenuOpen,
            )
        },
        snackbarHost = {
            SnackbarHost(hostState = snackbarHostState)
        },
        content = content
    )
}

// StorageAppScaffold.kt
@Composable
fun StorageAppScaffold(
    title: String,
    storageUsed: Float,
    storageTotal: Float,
    onNavigateBack: (() -> Unit)? = null,
    snackbarHostState: SnackbarHostState = remember { SnackbarHostState() },
    content: @Composable (PaddingValues) -> Unit
) {
    Scaffold(
        topBar = {
            StorageTopBar(
                title = title,
                storageUsed = storageUsed,
                storageTotal = storageTotal,
                onNavigateBack = onNavigateBack,
            )
        },
        snackbarHost = {
            SnackbarHost(hostState = snackbarHostState)
        },
        content = content
    )
}
