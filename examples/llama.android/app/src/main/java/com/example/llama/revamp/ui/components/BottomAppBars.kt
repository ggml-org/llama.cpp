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
import com.example.llama.revamp.viewmodel.ModelSortOrder

/**
 * [BottomAppBar] configurations
 */
sealed class BottomBarConfig {

    object None : BottomBarConfig()

    data class ModelsManagement(
        val isMultiSelectionMode: Boolean,
        val selectedModels: Map<String, ModelInfo>,
        val onSelectAll: () -> Unit,
        val onDeselectAll: () -> Unit,
        val onDeleteSelected: () -> Unit,
        val onSortClicked: () -> Unit,
        val onFilterClicked: () -> Unit,
        val onDeleteModeClicked: () -> Unit,
        val onAddModelClicked: () -> Unit,
        val onExitSelectionMode: () -> Unit,
        val showSortMenu: Boolean,
        val onSortMenuDismissed: () -> Unit,
        val currentSortOrder: ModelSortOrder,
        val onSortOptionSelected: (ModelSortOrder) -> Unit,
        val showImportModelMenu: Boolean,
        val onImportMenuDismissed: () -> Unit,
        val onImportLocalModelClicked: () -> Unit,
        val onImportHuggingFaceClicked: () -> Unit
    ) : BottomBarConfig()

    // TODO-han.yin: add more bottom bar types here
}

@Composable
fun ModelsManagementBottomBar(
    isMultiSelectionMode: Boolean,
    selectedModels: Map<String, ModelInfo>,
    onSelectAll: () -> Unit,
    onDeselectAll: () -> Unit,
    onDeleteSelected: () -> Unit,
    onSortClicked: () -> Unit,
    onFilterClicked: () -> Unit,
    onDeleteModeClicked: () -> Unit,
    onAddModelClicked: () -> Unit,
    onExitSelectionMode: () -> Unit,
    showSortMenu: Boolean,
    onSortMenuDismissed: () -> Unit,
    currentSortOrder: ModelSortOrder,
    onSortOptionSelected: (ModelSortOrder) -> Unit,
    showImportModelMenu: Boolean,
    onImportMenuDismissed: () -> Unit,
    onImportLocalModelClicked: () -> Unit,
    onImportHuggingFaceClicked: () -> Unit
) {
    BottomAppBar(
        actions = {
            if (isMultiSelectionMode) {
                // Multi-selection mode actions
                IconButton(onClick = onSelectAll) {
                    Icon(
                        imageVector = Icons.Default.SelectAll,
                        contentDescription = "Select all"
                    )
                }

                IconButton(onClick = onDeselectAll) {
                    Icon(
                        imageVector = Icons.Default.ClearAll,
                        contentDescription = "Deselect all"
                    )
                }

                IconButton(
                    onClick = onDeleteSelected,
                    enabled = selectedModels.isNotEmpty()
                ) {
                    Icon(
                        imageVector = Icons.Default.Delete,
                        contentDescription = "Delete selected",
                        tint = if (selectedModels.isNotEmpty())
                            MaterialTheme.colorScheme.error
                        else
                            MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.3f)
                    )
                }
            } else {
                // Default mode actions
                IconButton(onClick = onSortClicked) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.Sort,
                        contentDescription = "Sort models"
                    )
                }

                // Sort dropdown menu
                DropdownMenu(
                    expanded = showSortMenu,
                    onDismissRequest = onSortMenuDismissed
                ) {
                    DropdownMenuItem(
                        text = { Text("Name (A-Z)") },
                        trailingIcon = {
                            if (currentSortOrder == ModelSortOrder.NAME_ASC)
                                Icon(
                                    imageVector = Icons.Default.Check,
                                    contentDescription = "Sort by name in ascending order, selected"
                                )
                        },
                        onClick = {
                            onSortOptionSelected(ModelSortOrder.NAME_ASC)
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Name (Z-A)") },
                        trailingIcon = {
                            if (currentSortOrder == ModelSortOrder.NAME_DESC)
                                Icon(
                                    imageVector = Icons.Default.Check,
                                    contentDescription = "Sort by name in descending order, selected"
                                )
                        },
                        onClick = {
                            onSortOptionSelected(ModelSortOrder.NAME_DESC)
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Size (Smallest first)") },
                        trailingIcon = {
                            if (currentSortOrder == ModelSortOrder.SIZE_ASC)
                                Icon(
                                    imageVector = Icons.Default.Check,
                                    contentDescription = "Sort by size in ascending order, selected"
                                )
                        },
                        onClick = {
                            onSortOptionSelected(ModelSortOrder.SIZE_ASC)
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Size (Largest first)") },
                        trailingIcon = {
                            if (currentSortOrder == ModelSortOrder.SIZE_DESC)
                                Icon(
                                    imageVector = Icons.Default.Check,
                                    contentDescription = "Sort by size in descending order, selected"
                                )
                        },
                        onClick = {
                            onSortOptionSelected(ModelSortOrder.SIZE_DESC)
                        }
                    )
                    DropdownMenuItem(
                        text = { Text("Last used") },
                        trailingIcon = {
                            if (currentSortOrder == ModelSortOrder.LAST_USED)
                                Icon(
                                    imageVector = Icons.Default.Check,
                                    contentDescription = "Sort by last used, selected"
                                )
                        },
                        onClick = {
                            onSortOptionSelected(ModelSortOrder.LAST_USED)
                        }
                    )
                }

                IconButton(
                    onClick = onFilterClicked
                ) {
                    Icon(
                        imageVector = Icons.Default.FilterAlt,
                        contentDescription = "Filter models"
                    )
                }

                IconButton(onClick = onDeleteModeClicked) {
                    Icon(
                        imageVector = Icons.Default.DeleteSweep,
                        contentDescription = "Delete models"
                    )
                }
            }
        },
        floatingActionButton = {
            FloatingActionButton(
                onClick = if (isMultiSelectionMode) onExitSelectionMode else onAddModelClicked,
                containerColor = MaterialTheme.colorScheme.primaryContainer
            ) {
                Icon(
                    imageVector = if (isMultiSelectionMode) Icons.Default.Close else Icons.Default.Add,
                    contentDescription = if (isMultiSelectionMode) "Exit selection mode" else "Add model"
                )
            }

            // Add model dropdown menu
            DropdownMenu(
                expanded = showImportModelMenu,
                onDismissRequest = onImportMenuDismissed
            ) {
                DropdownMenuItem(
                    text = { Text("Import local model") },
                    leadingIcon = {
                        Icon(
                            imageVector = Icons.Default.FolderOpen,
                            contentDescription = "Import a local model on the device"
                        )
                    },
                    onClick = onImportLocalModelClicked
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
                    onClick = onImportHuggingFaceClicked
                )
            }
        }
    )
}
