package com.example.llama.revamp.ui.scaffold.topbar

import com.example.llama.revamp.data.repository.StorageMetrics
import com.example.llama.revamp.monitoring.MemoryMetrics
import com.example.llama.revamp.monitoring.TemperatureMetrics

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

    // Performance monitoring top bar with RAM and optional temperature
    data class Performance(
        override val title: String,
        override val navigationIcon: NavigationIcon,
        val memoryMetrics: MemoryMetrics,
        val temperatureInfo: Pair<TemperatureMetrics, Boolean>?,
    ) : TopBarConfig()

    // Storage management top bar with used & total storage
    data class Storage(
        override val title: String,
        override val navigationIcon: NavigationIcon,
        val storageMetrics: StorageMetrics?
    ) : TopBarConfig()
}

/**
 * Helper class for navigation icon configuration
 */
sealed class NavigationIcon {
    data class Back(val onNavigateBack: () -> Unit) : NavigationIcon()
    data class Menu(val onMenuOpen: () -> Unit) : NavigationIcon()
    object None : NavigationIcon()
}
