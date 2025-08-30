package com.example.llama.ui.scaffold.bottombar

import androidx.compose.foundation.text.input.TextFieldState
import com.example.llama.data.model.ModelFilter
import com.example.llama.data.model.ModelInfo
import com.example.llama.data.model.ModelSortOrder
import com.example.llama.viewmodel.Preselection

/**
 * [BottomAppBar] configurations
 */
sealed class BottomBarConfig {

    object None : BottomBarConfig()

    data class ModelSelection(
        val search: SearchConfig,
        val sorting: SortingConfig,
        val filtering: FilteringConfig,
        val runAction: RunActionConfig
    ) : BottomBarConfig() {
        data class SearchConfig(
            val isActive: Boolean,
            val onToggleSearch: (Boolean) -> Unit,
            val textFieldState: TextFieldState,
            val onSearch: (String) -> Unit,
        )

        data class SortingConfig(
            val currentOrder: ModelSortOrder,
            val isMenuVisible: Boolean,
            val toggleMenu: (Boolean) -> Unit,
            val selectOrder: (ModelSortOrder) -> Unit
        )

        data class FilteringConfig(
            val isActive: Boolean,
            val filters: Map<ModelFilter, Boolean>,
            val onToggleFilter: (ModelFilter, Boolean) -> Unit,
            val onClearFilters: () -> Unit,
            val isMenuVisible: Boolean,
            val toggleMenu: (Boolean) -> Unit
        )

        data class RunActionConfig(
            val preselection: Preselection?,
            val onClickRun: (Preselection) -> Unit,
        )
    }

    data class ModelsManagement(
        val sorting: SortingConfig,
        val filtering: FilteringConfig,
        val selection: SelectionConfig,
        val importing: ImportConfig
    ) : BottomBarConfig() {
        data class SortingConfig(
            val currentOrder: ModelSortOrder,
            val isMenuVisible: Boolean,
            val toggleMenu: (Boolean) -> Unit,
            val selectOrder: (ModelSortOrder) -> Unit
        )

        data class FilteringConfig(
            val isActive: Boolean,
            val filters: Map<ModelFilter, Boolean>,
            val onToggleFilter: (ModelFilter, Boolean) -> Unit,
            val onClearFilters: () -> Unit,
            val isMenuVisible: Boolean,
            val toggleMenu: (Boolean) -> Unit
        )

        data class SelectionConfig(
            val isActive: Boolean,
            val toggleMode: (Boolean) -> Unit,
            val selectedModels: Map<String, ModelInfo>,
            val toggleAllSelection: (Boolean) -> Unit,
            val deleteSelected: () -> Unit
        )

        data class ImportConfig(
            val isMenuVisible: Boolean,
            val toggleMenu: (Boolean) -> Unit,
            val importFromLocal: () -> Unit,
            val importFromHuggingFace: () -> Unit
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
