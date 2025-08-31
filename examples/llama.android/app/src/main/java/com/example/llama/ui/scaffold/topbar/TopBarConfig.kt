package com.example.llama.ui.scaffold.topbar

import com.example.llama.monitoring.MemoryMetrics
import com.example.llama.monitoring.StorageMetrics
import com.example.llama.monitoring.TemperatureMetrics
import com.example.llama.viewmodel.ModelScreenUiMode

/**
 * [TopAppBar] configurations
 */
sealed class TopBarConfig {
    abstract val title: String
    abstract val navigationIcon: NavigationIcon

    // No top bar at all
    data class None(
        override val title: String = "",
        override val navigationIcon: NavigationIcon = NavigationIcon.None
    ) : TopBarConfig()

    // Default/simple top bar with only a navigation icon
    data class Default(
        override val title: String,
        override val navigationIcon: NavigationIcon
    ) : TopBarConfig()

    // Model management top bar with a toggle to turn on/off manage mode
    data class ModelsBrowsing(
        override val title: String,
        override val navigationIcon: NavigationIcon,
        val onToggleManaging: (() -> Unit)?,
    ) : TopBarConfig()

    // Model batch-deletion top bar with a toggle to turn on/off manage mode
    data class ModelsDeleting(
        override val title: String,
        override val navigationIcon: NavigationIcon,
    ) : TopBarConfig()

    // Performance monitoring top bar with RAM and optional temperature
    data class Performance(
        override val title: String,
        override val navigationIcon: NavigationIcon,
        val memoryMetrics: MemoryMetrics?,
        val temperatureInfo: Pair<TemperatureMetrics, Boolean>?,
    ) : TopBarConfig()

    // Storage management top bar with used & total storage
    data class ModelsManagement(
        override val title: String,
        override val navigationIcon: NavigationIcon,
        val storageMetrics: StorageMetrics?
    ) : TopBarConfig()
}

/**
 * Helper class for navigation icon configuration
 */
sealed class NavigationIcon {
    data class Menu(val onMenuOpen: () -> Unit) : NavigationIcon()
    data class Back(val onNavigateBack: () -> Unit) : NavigationIcon()
    data class Quit(val onQuit: () -> Unit) : NavigationIcon()
    data object None : NavigationIcon()
}
