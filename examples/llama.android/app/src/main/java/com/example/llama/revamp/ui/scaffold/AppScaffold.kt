package com.example.llama.revamp.ui.scaffold

import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember

/**
 * Configuration of both [TopBarConfig] and [BottomBarConfig]
 */
data class ScaffoldConfig(
    val topBarConfig: TopBarConfig,
    val bottomBarConfig: BottomBarConfig = BottomBarConfig.None
)

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
        when (val topConfig = topBarconfig) {
            is TopBarConfig.None -> {}

            is TopBarConfig.Default -> DefaultTopBar(
                title = topBarconfig.title,
                onNavigateBack = topConfig.navigationIcon.backAction,
                onMenuOpen = topConfig.navigationIcon.menuAction,
            )

            is TopBarConfig.Performance -> PerformanceTopBar(
                title = topBarconfig.title,
                memoryMetrics = topBarconfig.memoryMetrics,
                temperatureDisplay = topBarconfig.temperatureInfo,
                onNavigateBack = topConfig.navigationIcon.backAction,
                onMenuOpen = topConfig.navigationIcon.menuAction,
            )

            is TopBarConfig.Storage -> StorageTopBar(
                title = topBarconfig.title,
                storageMetrics = topBarconfig.storageMetrics,
                onNavigateBack = topConfig.navigationIcon.backAction,
            )
        }
    }

    val bottomBar: @Composable () -> Unit = {
        when (val config = bottomBarConfig) {
            is BottomBarConfig.None -> { /* No bottom bar */ }

            is BottomBarConfig.ModelSelection -> {
                ModelSelectionBottomBar(
                    search = config.search,
                    sorting = config.sorting,
                    filtering = config.filtering,
                    runAction = config.runAction
                )
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

// Helper functions to obtain navigation actions if exist
private val NavigationIcon.backAction: (() -> Unit)?
    get() = (this as? NavigationIcon.Back)?.onNavigateBack

private val NavigationIcon.menuAction: (() -> Unit)?
    get() = (this as? NavigationIcon.Menu)?.onMenuOpen
