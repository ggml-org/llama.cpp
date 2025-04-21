package com.example.llama.revamp.ui.scaffold

import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.text.input.TextFieldState
import androidx.compose.foundation.text.input.clearText
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Sort
import androidx.compose.material.icons.automirrored.outlined.Backspace
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.ClearAll
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.FilterAlt
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.SearchOff
import androidx.compose.material.icons.filled.SelectAll
import androidx.compose.material.icons.outlined.FilterAlt
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.Checkbox
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import com.example.llama.R
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.viewmodel.ModelSortOrder

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
            val filters: Map<String, Boolean>, // Filter name -> enabled
            val onToggleFilter: (String, Boolean) -> Unit,
            val onClearFilters: () -> Unit,
            val isMenuVisible: Boolean,
            val toggleMenu: (Boolean) -> Unit
        )

        data class RunActionConfig(
            val selectedModel: ModelInfo?,
            val onRun: (ModelInfo) -> Unit
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
            val onClick: () -> Unit
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

    // TODO-han.yin: add bottom bar config for Conversation Screen!
}

@Composable
fun ModelSelectionBottomBar(
    search: BottomBarConfig.ModelSelection.SearchConfig,
    sorting: BottomBarConfig.ModelSelection.SortingConfig,
    filtering: BottomBarConfig.ModelSelection.FilteringConfig,
    runAction: BottomBarConfig.ModelSelection.RunActionConfig
) {
    BottomAppBar(
        actions = {
            if (search.isActive) {
                // Quit search action
                IconButton(onClick = { search.onToggleSearch(false) }) {
                    Icon(
                        imageVector = Icons.Default.SearchOff,
                        contentDescription = "Quit search mode"
                    )
                }

                // Clear query action
                IconButton(onClick = { search.textFieldState.clearText() }) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Outlined.Backspace,
                        contentDescription = "Clear query text"
                    )
                }
            } else {
                // Enter search action
                IconButton(onClick = { search.onToggleSearch(true) }) {
                    Icon(
                        imageVector = Icons.Default.Search,
                        contentDescription = "Search models"
                    )
                }

                // Sorting action
                IconButton(onClick = { sorting.toggleMenu(true) }) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.Sort,
                        contentDescription = "Sort models"
                    )
                }

                // Sorting dropdown menu
                DropdownMenu(
                    expanded = sorting.isMenuVisible,
                    onDismissRequest = { sorting.toggleMenu(false) }
                ) {
                    val sortOptions = listOf(
                        Triple(ModelSortOrder.NAME_ASC, "Name (A-Z)", "Sort by name in ascending order"),
                        Triple(ModelSortOrder.NAME_DESC, "Name (Z-A)", "Sort by name in descending order"),
                        Triple(ModelSortOrder.SIZE_ASC, "Size (Smallest first)", "Sort by size in ascending order"),
                        Triple(ModelSortOrder.SIZE_DESC, "Size (Largest first)", "Sort by size in descending order"),
                        Triple(ModelSortOrder.LAST_USED, "Last used", "Sort by last used")
                    )

                    sortOptions.forEach { (order, label, description) ->
                        DropdownMenuItem(
                            text = { Text(label) },
                            trailingIcon = {
                                if (sorting.currentOrder == order)
                                    Icon(
                                        imageVector = Icons.Default.Check,
                                        contentDescription = "$description, selected"
                                    )
                            },
                            onClick = { sorting.selectOrder(order) }
                        )
                    }
                }

                // Filter action
                IconButton(onClick = { filtering.toggleMenu(true) }) {
                    Icon(
                        imageVector =
                            if (filtering.isActive) Icons.Default.FilterAlt
                            else Icons.Outlined.FilterAlt,
                        contentDescription = "Filter models"
                    )
                }

                // Filter dropdown menu
                DropdownMenu(
                    expanded = filtering.isMenuVisible,
                    onDismissRequest = { filtering.toggleMenu(false) }
                ) {
                    Text(
                        text = "Filter by",
                        style = MaterialTheme.typography.labelMedium,
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                    )

                    filtering.filters.forEach { (filter, isEnabled) ->
                        DropdownMenuItem(
                            text = { Text(filter) },
                            leadingIcon = {
                                Checkbox(
                                    checked = isEnabled,
                                    onCheckedChange = null
                                )
                            },
                            onClick = { filtering.onToggleFilter(filter, !isEnabled) }
                        )
                    }

                    HorizontalDivider()

                    DropdownMenuItem(
                        text = { Text("Clear filters") },
                        onClick = {
                            filtering.onClearFilters()
                            filtering.toggleMenu(false)
                        }
                    )
                }
            }
        },
        floatingActionButton = {
            // Only show FAB if a model is selected
            runAction.selectedModel?.let { model ->
                FloatingActionButton(
                    onClick = { runAction.onRun(model) },
                    containerColor = MaterialTheme.colorScheme.primary
                ) {
                    Icon(
                        imageVector = Icons.Default.PlayArrow,
                        contentDescription = "Run with selected model"
                    )
                }
            }
        }
    )
}

@Composable
fun ModelsManagementBottomBar(
    sorting: BottomBarConfig.ModelsManagement.SortingConfig,
    filtering: BottomBarConfig.ModelsManagement.FilteringConfig,
    selection: BottomBarConfig.ModelsManagement.SelectionConfig,
    importing: BottomBarConfig.ModelsManagement.ImportConfig
) {
    BottomAppBar(
        actions = {
            if (selection.isActive) {
                /* Multi-selection mode actions */
                IconButton(onClick = { selection.toggleAllSelection(true) }) {
                    Icon(
                        imageVector = Icons.Default.SelectAll,
                        contentDescription = "Select all"
                    )
                }

                IconButton(onClick = { selection.toggleAllSelection(false) }) {
                    Icon(
                        imageVector = Icons.Default.ClearAll,
                        contentDescription = "Deselect all"
                    )
                }

                IconButton(
                    onClick = selection.deleteSelected,
                    enabled = selection.selectedModels.isNotEmpty()
                ) {
                    Icon(
                        imageVector = Icons.Default.Delete,
                        contentDescription = "Delete selected",
                        tint = if (selection.selectedModels.isNotEmpty())
                            MaterialTheme.colorScheme.error
                        else
                            MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.3f)
                    )
                }

            } else {
                /* Default mode actions */

                // Sorting action
                IconButton(onClick = { sorting.toggleMenu(true) }) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.Sort,
                        contentDescription = "Sort models"
                    )
                }

                // Sorting dropdown menu
                DropdownMenu(
                    expanded = sorting.isMenuVisible,
                    onDismissRequest = { sorting.toggleMenu(false) }
                ) {
                    val sortOptions = listOf(
                        Triple(ModelSortOrder.NAME_ASC, "Name (A-Z)", "Sort by name in ascending order"),
                        Triple(ModelSortOrder.NAME_DESC, "Name (Z-A)", "Sort by name in descending order"),
                        Triple(ModelSortOrder.SIZE_ASC, "Size (Smallest first)", "Sort by size in ascending order"),
                        Triple(ModelSortOrder.SIZE_DESC, "Size (Largest first)", "Sort by size in descending order"),
                        Triple(ModelSortOrder.LAST_USED, "Last used", "Sort by last used")
                    )

                    sortOptions.forEach { (order, label, description) ->
                        DropdownMenuItem(
                            text = { Text(label) },
                            trailingIcon = {
                                if (sorting.currentOrder == order)
                                    Icon(
                                        imageVector = Icons.Default.Check,
                                        contentDescription = "$description, selected"
                                    )
                            },
                            onClick = { sorting.selectOrder(order) }
                        )
                    }
                }

                // Filtering action
                IconButton(
                    onClick = filtering.onClick
                ) {
                    Icon(
                        imageVector = Icons.Default.FilterAlt,
                        contentDescription = "Filter models"
                    )
                }

                // Selection action
                IconButton(onClick = { selection.toggleMode(true) }) {
                    Icon(
                        imageVector = Icons.Default.DeleteSweep,
                        contentDescription = "Delete models"
                    )
                }
            }
        },
        floatingActionButton = {
            FloatingActionButton(
                onClick = { if (selection.isActive) selection.toggleMode(false) else importing.toggleMenu(true) },
                containerColor = MaterialTheme.colorScheme.primaryContainer
            ) {
                Icon(
                    imageVector = if (selection.isActive) Icons.Default.Close else Icons.Default.Add,
                    contentDescription = if (selection.isActive) "Exit selection mode" else "Add model"
                )
            }

            // Add model dropdown menu
            DropdownMenu(
                expanded = importing.isMenuVisible,
                onDismissRequest = { importing.toggleMenu(false) }
            ) {
                DropdownMenuItem(
                    text = { Text("Import local model") },
                    leadingIcon = {
                        Icon(
                            imageVector = Icons.Default.FolderOpen,
                            contentDescription = "Import a local model on the device"
                        )
                    },
                    onClick = importing.importFromLocal
                )
                DropdownMenuItem(
                    text = { Text("Download from HuggingFace") },
                    leadingIcon = {
                        Icon(
                            painter = painterResource(id = R.drawable.logo_huggingface),
                            contentDescription = "Browse and download a model from HuggingFace",
                            modifier = Modifier.size(24.dp),
                            tint = Color.Unspecified,
                        )
                    },
                    onClick = importing.importFromHuggingFace
                )
            }
        }
    )
}
