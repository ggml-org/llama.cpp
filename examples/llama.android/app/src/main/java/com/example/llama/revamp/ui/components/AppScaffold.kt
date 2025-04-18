package com.example.llama.revamp.ui.components

import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember


/**
 * Events called back from child screens
 */
sealed class ScaffoldEvent {
    data class ShowSnackbar(
        val message: String,
        val duration: SnackbarDuration = SnackbarDuration.Short,
        val withDismissAction: Boolean = false,
        val actionLabel: String? = null,
        val onAction: (() -> Unit)? = null
    ) : ScaffoldEvent()

     data class ChangeTitle(val newTitle: String) : ScaffoldEvent()
}

@Composable
fun AppScaffold(
    topBarconfig: TopBarConfig,
    bottomBarConfig: BottomBarConfig = BottomBarConfig.None,
    snackbarHostState: SnackbarHostState = remember { SnackbarHostState() },
    content: @Composable (PaddingValues) -> Unit
) {
    val topBar: @Composable () -> Unit = {
        when (topBarconfig) {
            is TopBarConfig.Performance -> {
                PerformanceTopBar(
                    title = topBarconfig.title,
                    memoryMetrics = topBarconfig.memoryMetrics,
                    temperatureDisplay = topBarconfig.temperatureInfo,
                    onNavigateBack = (topBarconfig.navigationIcon as? TopBarConfig.NavigationIcon.Back)?.onNavigateBack,
                    onMenuOpen = (topBarconfig.navigationIcon as? TopBarConfig.NavigationIcon.Menu)?.onMenuOpen
                )
            }

            is TopBarConfig.Storage -> {
                StorageTopBar(
                    title = topBarconfig.title,
                    storageMetrics = topBarconfig.storageMetrics,
                    onNavigateBack = (topBarconfig.navigationIcon as? TopBarConfig.NavigationIcon.Back)?.onNavigateBack
                )
            }

            is TopBarConfig.Default -> {
                DefaultTopBar(
                    title = topBarconfig.title,
                    onNavigateBack = (topBarconfig.navigationIcon as? TopBarConfig.NavigationIcon.Back)?.onNavigateBack,
                    onMenuOpen = (topBarconfig.navigationIcon as? TopBarConfig.NavigationIcon.Menu)?.onMenuOpen
                )
            }
        }
    }

    val bottomBar: @Composable () -> Unit = {
        when (val config = bottomBarConfig) {
            is BottomBarConfig.None -> {
                /* No bottom bar */
            }
            is BottomBarConfig.ModelsManagement -> {
                ModelsManagementBottomBar(
                    sorting = config.sorting,
                    filtering = config.filtering,
                    selection = config.selection,
                    importing = config.importing,
                )
            }
        }
    }

    Scaffold(
        topBar = topBar,
        bottomBar = bottomBar,
        snackbarHost = { SnackbarHost(hostState = snackbarHostState) },
        content = content
    )
}
