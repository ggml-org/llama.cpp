package com.arm.aiplayground.ui.scaffold

import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import com.arm.aiplayground.ui.scaffold.bottombar.BenchmarkBottomBar
import com.arm.aiplayground.ui.scaffold.bottombar.BottomBarConfig
import com.arm.aiplayground.ui.scaffold.bottombar.ConversationBottomBar
import com.arm.aiplayground.ui.scaffold.bottombar.ModelsBrowsingBottomBar
import com.arm.aiplayground.ui.scaffold.bottombar.ModelsDeletingBottomBar
import com.arm.aiplayground.ui.scaffold.bottombar.ModelsManagementBottomBar
import com.arm.aiplayground.ui.scaffold.bottombar.ModelsSearchingBottomBar
import com.arm.aiplayground.ui.scaffold.topbar.DefaultTopBar
import com.arm.aiplayground.ui.scaffold.topbar.ModelsBrowsingTopBar
import com.arm.aiplayground.ui.scaffold.topbar.NavigationIcon
import com.arm.aiplayground.ui.scaffold.topbar.PerformanceTopBar
import com.arm.aiplayground.ui.scaffold.topbar.ModelsManagementTopBar
import com.arm.aiplayground.ui.scaffold.topbar.TopBarConfig

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
    onScaffoldEvent: (ScaffoldEvent) -> Unit,
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

            is TopBarConfig.ModelsBrowsing -> ModelsBrowsingTopBar(
                title = topBarconfig.title,
                showTooltip = topBarconfig.showTooltip,
                showManagingToggle = topBarconfig.showManagingToggle,
                onToggleManaging = topBarconfig.onToggleManaging,
                onNavigateBack = topBarconfig.navigationIcon.backAction,
                onMenuOpen = topBarconfig.navigationIcon.menuAction
            )

            is TopBarConfig.ModelsDeleting -> DefaultTopBar(
                title = topBarconfig.title,
                titleColor = MaterialTheme.colorScheme.error,
                navigationIconTint = MaterialTheme.colorScheme.error,
                onQuit = topBarconfig.navigationIcon.quitAction
            )

            is TopBarConfig.ModelsManagement -> ModelsManagementTopBar(
                title = topBarconfig.title,
                storageMetrics = topBarconfig.storageMetrics,
                onScaffoldEvent = onScaffoldEvent,
                onNavigateBack = topBarconfig.navigationIcon.backAction,
            )

            is TopBarConfig.Performance -> PerformanceTopBar(
                title = topBarconfig.title,
                memoryMetrics = topBarconfig.memoryMetrics,
                temperatureDisplay = topBarconfig.temperatureInfo,
                onScaffoldEvent = onScaffoldEvent,
                onNavigateBack = topBarconfig.navigationIcon.backAction,
                onMenuOpen = topBarconfig.navigationIcon.menuAction,
            )
        }
    }

    val bottomBar: @Composable () -> Unit = {
        when (val config = bottomBarConfig) {
            is BottomBarConfig.None -> { /* No bottom bar */ }

            is BottomBarConfig.Models.Browsing -> {
                ModelsBrowsingBottomBar(
                    isSearchingEnabled = config.isSearchingEnabled,
                    onToggleSearching = config.onToggleSearching,
                    sortingConfig = config.sorting,
                    filteringConfig = config.filtering,
                    runActionConfig = config.runAction
                )
            }

            is BottomBarConfig.Models.Searching -> {
                ModelsSearchingBottomBar(
                    textFieldState = config.textFieldState,
                    onQuitSearching = config.onQuitSearching,
                    onSearch = config.onSearch,
                    runActionConfig = config.runAction,
                )
            }

            is BottomBarConfig.Models.Managing -> {
                ModelsManagementBottomBar(
                    isDeletionEnabled = config.isDeletionEnabled,
                    onToggleDeleting = config.onToggleDeleting,
                    sortingConfig = config.sorting,
                    filteringConfig = config.filtering,
                    importingConfig = config.importing
                )
            }

            is BottomBarConfig.Models.Deleting -> {
                ModelsDeletingBottomBar(config)
            }

            is BottomBarConfig.Benchmark -> {
                BenchmarkBottomBar(
                    showShareFab = config.showShareFab,
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
private val NavigationIcon.menuAction: (() -> Unit)?
    get() = (this as? NavigationIcon.Menu)?.onMenuOpen

private val NavigationIcon.backAction: (() -> Unit)?
    get() = (this as? NavigationIcon.Back)?.onNavigateBack

private val NavigationIcon.quitAction: (() -> Unit)?
    get() = (this as? NavigationIcon.Quit)?.onQuit
