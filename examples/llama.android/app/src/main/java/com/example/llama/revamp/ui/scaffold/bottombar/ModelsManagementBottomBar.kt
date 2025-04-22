package com.example.llama.revamp.ui.scaffold.bottombar

import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Sort
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.ClearAll
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.FilterAlt
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.SelectAll
import androidx.compose.material.icons.outlined.DeleteSweep
import androidx.compose.material.icons.outlined.FilterAlt
import androidx.compose.material.icons.outlined.FilterAltOff
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
import com.example.llama.revamp.data.model.ModelSortOrder

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

                IconButton(onClick = { selection.toggleAllSelection(false) }) {
                    Icon(
                        imageVector = Icons.Default.ClearAll,
                        contentDescription = "Deselect all"
                    )
                }

                IconButton(onClick = { selection.toggleAllSelection(true) }) {
                    Icon(
                        imageVector = Icons.Default.SelectAll,
                        contentDescription = "Select all"
                    )
                }
            } else {
                /* Default mode actions */

                // Multi-selection action
                IconButton(onClick = { selection.toggleMode(true) }) {
                    Icon(
                        imageVector = Icons.Outlined.DeleteSweep,
                        contentDescription = "Delete models"
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
                        Triple(
                            ModelSortOrder.NAME_ASC,
                            "Name (A-Z)",
                            "Sort by name in ascending order"
                        ),
                        Triple(
                            ModelSortOrder.NAME_DESC,
                            "Name (Z-A)",
                            "Sort by name in descending order"
                        ),
                        Triple(
                            ModelSortOrder.SIZE_ASC,
                            "Size (Smallest first)",
                            "Sort by size in ascending order"
                        ),
                        Triple(
                            ModelSortOrder.SIZE_DESC,
                            "Size (Largest first)",
                            "Sort by size in descending order"
                        ),
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
                        modifier = Modifier.Companion.padding(horizontal = 16.dp, vertical = 8.dp)
                    )

                    filtering.filters.forEach { (filter, isEnabled) ->
                        DropdownMenuItem(
                            text = { Text(filter.displayName) },
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
                        leadingIcon = {
                            Icon(
                                imageVector = Icons.Outlined.FilterAltOff,
                                contentDescription = "Clear all filters"
                            )
                        },
                        onClick = {
                            filtering.onClearFilters()
                            filtering.toggleMenu(false)
                        }
                    )
                }
            }
        },
        floatingActionButton = {
            FloatingActionButton(
                onClick = {
                    if (selection.isActive) selection.toggleMode(false) else importing.toggleMenu(
                        true
                    )
                },
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
                            modifier = Modifier.Companion.size(24.dp),
                            tint = Color.Companion.Unspecified,
                        )
                    },
                    onClick = importing.importFromHuggingFace
                )
            }
        }
    )
}
