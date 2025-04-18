package com.example.llama.revamp.ui.components

import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Sort
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.ClearAll
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.FilterAlt
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.SelectAll
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FloatingActionButton
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
import com.example.llama.revamp.ui.components.BottomBarConfig.ModelsManagement
import com.example.llama.revamp.viewmodel.ModelSortOrder

/**
 * [BottomAppBar] configurations
 */
sealed class BottomBarConfig {

    object None : BottomBarConfig()

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

    // TODO-han.yin: add more bottom bar types here
}

@Composable
fun ModelsManagementBottomBar(
    sorting: ModelsManagement.SortingConfig,
    filtering: ModelsManagement.FilteringConfig,
    selection: ModelsManagement.SelectionConfig,
    importing: ModelsManagement.ImportConfig
) {
    BottomAppBar(
        actions = {
            if (selection.isActive) {
                // Multi-selection mode actions
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
                // Default mode actions
                IconButton(onClick = { sorting.toggleMenu(true) }) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.Sort,
                        contentDescription = "Sort models"
                    )
                }

                // Sort dropdown menu
                DropdownMenu(
                    expanded = sorting.isMenuVisible,
                    onDismissRequest = { sorting.toggleMenu(false) }
                ) {
                    DropdownMenuItem(
                        text = { Text("Name (A-Z)") },
                        trailingIcon = {
                            if (sorting.currentOrder == ModelSortOrder.NAME_ASC)
                                Icon(
                                    imageVector = Icons.Default.Check,
                                    contentDescription = "Sort by name in ascending order, selected"
                                )
                        },
                        onClick = {
                            sorting.selectOrder(ModelSortOrder.NAME_ASC)
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Name (Z-A)") },
                        trailingIcon = {
                            if (sorting.currentOrder == ModelSortOrder.NAME_DESC)
                                Icon(
                                    imageVector = Icons.Default.Check,
                                    contentDescription = "Sort by name in descending order, selected"
                                )
                        },
                        onClick = {
                            sorting.selectOrder(ModelSortOrder.NAME_DESC)
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Size (Smallest first)") },
                        trailingIcon = {
                            if (sorting.currentOrder == ModelSortOrder.SIZE_ASC)
                                Icon(
                                    imageVector = Icons.Default.Check,
                                    contentDescription = "Sort by size in ascending order, selected"
                                )
                        },
                        onClick = {
                            sorting.selectOrder(ModelSortOrder.SIZE_ASC)
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Size (Largest first)") },
                        trailingIcon = {
                            if (sorting.currentOrder == ModelSortOrder.SIZE_DESC)
                                Icon(
                                    imageVector = Icons.Default.Check,
                                    contentDescription = "Sort by size in descending order, selected"
                                )
                        },
                        onClick = {
                            sorting.selectOrder(ModelSortOrder.SIZE_DESC)
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Last used") },
                        trailingIcon = {
                            if (sorting.currentOrder == ModelSortOrder.LAST_USED)
                                Icon(
                                    imageVector = Icons.Default.Check,
                                    contentDescription = "Sort by last used, selected"
                                )
                        },
                        onClick = {
                            sorting.selectOrder(ModelSortOrder.LAST_USED)
                        }
                    )
                }

                IconButton(
                    onClick = filtering.onClick
                ) {
                    Icon(
                        imageVector = Icons.Default.FilterAlt,
                        contentDescription = "Filter models"
                    )
                }

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
