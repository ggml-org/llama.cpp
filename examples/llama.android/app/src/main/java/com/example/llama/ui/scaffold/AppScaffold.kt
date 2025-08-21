package com.example.llama.ui.scaffold

import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import com.example.llama.ui.scaffold.bottombar.BenchmarkBottomBar
import com.example.llama.ui.scaffold.bottombar.BottomBarConfig
import com.example.llama.ui.scaffold.bottombar.ConversationBottomBar
import com.example.llama.ui.scaffold.bottombar.ModelSelectionBottomBar
import com.example.llama.ui.scaffold.bottombar.ModelsManagementBottomBar
import com.example.llama.ui.scaffold.topbar.DefaultTopBar
import com.example.llama.ui.scaffold.topbar.NavigationIcon
import com.example.llama.ui.scaffold.topbar.PerformanceTopBar
import com.example.llama.ui.scaffold.topbar.StorageTopBar
import com.example.llama.ui.scaffold.topbar.TopBarConfig

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

    data class ShareText(
        val text: String,
        val title: String? = null,
        val mimeType: String = "text/plain"
    ) : ScaffoldEvent()
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
            is TopBarConfig.None -> {}

            is TopBarConfig.Default -> DefaultTopBar(
                title = topBarconfig.title,
                onNavigateBack = topBarconfig.navigationIcon.backAction,
                onMenuOpen = topBarconfig.navigationIcon.menuAction,
            )

            is TopBarConfig.Performance -> PerformanceTopBar(
                title = topBarconfig.title,
                memoryMetrics = topBarconfig.memoryMetrics,
                temperatureDisplay = topBarconfig.temperatureInfo,
                onNavigateBack = topBarconfig.navigationIcon.backAction,
                onMenuOpen = topBarconfig.navigationIcon.menuAction,
            )

            is TopBarConfig.Storage -> StorageTopBar(
                title = topBarconfig.title,
                storageMetrics = topBarconfig.storageMetrics,
                onNavigateBack = topBarconfig.navigationIcon.backAction,
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

            is BottomBarConfig.Benchmark -> {
                BenchmarkBottomBar(
                    engineIdle = config.engineIdle,
                    onShare = config.onShare,
                    onRerun = config.onRerun,
                    onClear = config.onClear,
                    showModelCard = config.showModelCard,
                    onToggleModelCard = config.onToggleModelCard,
                )
            }

            is BottomBarConfig.Conversation -> {
                ConversationBottomBar(
                    isReady = config.isEnabled,
                    textFieldState = config.textFieldState,
                    onSendClick = config.onSendClick,
                    showModelCard = config.showModelCard,
                    onToggleModelCard = config.onToggleModelCard,
                    onAttachPhotoClick = config.onAttachPhotoClick,
                    onAttachFileClick = config.onAttachFileClick,
                    onAudioInputClick = config.onAudioInputClick,
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
