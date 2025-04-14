package com.example.llama.revamp.ui.components

import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.hilt.navigation.compose.hiltViewModel
import com.example.llama.revamp.viewmodel.PerformanceViewModel

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

@Composable
fun PerformanceAppScaffold(
    performanceViewModel: PerformanceViewModel = hiltViewModel(),
    title: String,
    onNavigateBack: (() -> Unit)? = null,
    onMenuOpen: (() -> Unit)? = null,
    showTemperature: Boolean = false,
    snackbarHostState: SnackbarHostState = remember { SnackbarHostState() },
    content: @Composable (PaddingValues) -> Unit
) {
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

@Composable
fun StorageAppScaffold(
    title: String,
    storageUsed: Float,
    storageTotal: Float,
    onNavigateBack: (() -> Unit)? = null,
    snackbarHostState: SnackbarHostState = remember { SnackbarHostState() },
    bottomBar: @Composable () -> Unit = {},
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
        bottomBar = bottomBar,
        content = content
    )
}
