package com.example.llama.ui.scaffold.bottombar

import androidx.compose.foundation.text.input.TextFieldState
import com.example.llama.data.model.ModelFilter
import com.example.llama.data.model.ModelInfo
import com.example.llama.data.model.ModelSortOrder
import com.example.llama.viewmodel.PreselectedModelToRun

/**
 * [BottomAppBar] configurations
 */
sealed class BottomBarConfig {

    object None : BottomBarConfig()

    sealed class Models : BottomBarConfig() {

        data class Browsing(
            val isSearchingEnabled: Boolean,
            val onToggleSearching: () -> Unit,
            val sorting: SortingConfig,
            val filtering: FilteringConfig,
            val runAction: RunActionConfig,
        ) : BottomBarConfig() {
            data class SortingConfig(
                val isEnabled: Boolean,
                val currentOrder: ModelSortOrder,
                val isMenuVisible: Boolean,
                val toggleMenu: (Boolean) -> Unit,
                val selectOrder: (ModelSortOrder) -> Unit
            )

            data class FilteringConfig(
                val isEnabled: Boolean,
                val filters: Map<ModelFilter, Boolean>,
                val onToggleFilter: (ModelFilter, Boolean) -> Unit,
                val onClearFilters: () -> Unit,
                val isMenuVisible: Boolean,
                val toggleMenu: (Boolean) -> Unit
            )
        }

        data class Searching(
            val textFieldState: TextFieldState,
            val onQuitSearching: () -> Unit,
            val onSearch: (String) -> Unit,
            val runAction: RunActionConfig,
        ) : BottomBarConfig()

        data class Managing(
            val isDeletionEnabled: Boolean,
            val onToggleDeleting: () -> Unit,
            val sorting: SortingConfig,
            val filtering: FilteringConfig,
            val importing: ImportConfig,
        ) : BottomBarConfig() {
            data class SortingConfig(
                val isEnabled: Boolean,
                val currentOrder: ModelSortOrder,
                val isMenuVisible: Boolean,
                val toggleMenu: (Boolean) -> Unit,
                val selectOrder: (ModelSortOrder) -> Unit
            )

            data class FilteringConfig(
                val isEnabled: Boolean,
                val filters: Map<ModelFilter, Boolean>,
                val onToggleFilter: (ModelFilter, Boolean) -> Unit,
                val onClearFilters: () -> Unit,
                val isMenuVisible: Boolean,
                val toggleMenu: (Boolean) -> Unit
            )

            data class ImportConfig(
                val showTooltip: Boolean,
                val isMenuVisible: Boolean,
                val toggleMenu: (Boolean) -> Unit,
                val importFromLocal: () -> Unit,
                val importFromHuggingFace: () -> Unit
            )
        }

        data class Deleting(
            val onQuitDeleting: () -> Unit,
            val selectedModels: Map<String, ModelInfo>,
            val selectAllFilteredModels: () -> Unit,
            val clearAllSelectedModels: () -> Unit,
            val deleteSelected: () -> Unit
        ) : BottomBarConfig()

        data class RunActionConfig(
            val preselectedModelToRun: PreselectedModelToRun?,
            val onClickRun: (PreselectedModelToRun) -> Unit,
        )
    }

    data class Benchmark(
        val showShareFab: Boolean,
        val engineIdle: Boolean,
        val onShare: () -> Unit,
        val onRerun: () -> Unit,
        val onClear: () -> Unit,
        val showModelCard: Boolean,
        val onToggleModelCard: (Boolean) -> Unit,
    ) : BottomBarConfig()

    data class Conversation(
        val isEnabled: Boolean,
        val textFieldState: TextFieldState,
        val onSendClick: () -> Unit,
        val showModelCard: Boolean,
        val onToggleModelCard: (Boolean) -> Unit,
        val onAttachPhotoClick: (() -> Unit)?,
        val onAttachFileClick: (() -> Unit)?,
        val onAudioInputClick: (() -> Unit)?,
    ) : BottomBarConfig()
}
