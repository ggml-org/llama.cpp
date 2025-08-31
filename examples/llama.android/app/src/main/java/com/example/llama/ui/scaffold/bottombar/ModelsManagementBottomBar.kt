package com.example.llama.ui.scaffold.bottombar

import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Sort
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.FilterAlt
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.outlined.DeleteSweep
import androidx.compose.material.icons.outlined.FilterAlt
import androidx.compose.material.icons.outlined.FilterAltOff
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.Checkbox
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.IconButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.PlainTooltip
import androidx.compose.material3.Text
import androidx.compose.material3.TooltipAnchorPosition
import androidx.compose.material3.TooltipBox
import androidx.compose.material3.TooltipDefaults
import androidx.compose.material3.rememberTooltipState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import com.example.llama.R
import com.example.llama.data.model.ModelSortOrder

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelsManagementBottomBar(
    isDeletionEnabled: Boolean,
    onToggleDeleting: () -> Unit,
    sortingConfig: BottomBarConfig.Models.Managing.SortingConfig,
    filteringConfig: BottomBarConfig.Models.Managing.FilteringConfig,
    importingConfig: BottomBarConfig.Models.Managing.ImportConfig,
) {
    val tooltipState = rememberTooltipState(
        initialIsVisible = false,
        isPersistent = importingConfig.showTooltip)

    LaunchedEffect(importingConfig) {
        if (importingConfig.showTooltip && !importingConfig.isMenuVisible) {
            tooltipState.show()
        }
    }

    BottomAppBar(
        actions = {
            // Batch-deletion action
            IconButton(enabled = isDeletionEnabled, onClick = onToggleDeleting) {
                Icon(
                    imageVector = Icons.Outlined.DeleteSweep,
                    contentDescription = "Delete models",)
            }

            // Sorting action
            IconButton(
                enabled = sortingConfig.isEnabled,
                onClick = { sortingConfig.toggleMenu(true) }
            ) {
                Icon(
                    imageVector = Icons.AutoMirrored.Filled.Sort,
                    contentDescription = "Sort models"
                )
            }

            // Sorting dropdown menu
            DropdownMenu(
                expanded = sortingConfig.isMenuVisible,
                onDismissRequest = { sortingConfig.toggleMenu(false) }
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
                            if (sortingConfig.currentOrder == order)
                                Icon(
                                    imageVector = Icons.Default.Check,
                                    contentDescription = "$description, selected"
                                )
                        },
                        onClick = { sortingConfig.selectOrder(order) }
                    )
                }
            }

            // Filtering action
            val hasFilters = filteringConfig.filters.any { it.value }
            IconButton(
                enabled = filteringConfig.isEnabled,
                onClick = { filteringConfig.toggleMenu(true) },
                colors = IconButtonDefaults.iconButtonColors().copy(
                    contentColor = if (hasFilters) MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.onSurfaceVariant
                ),
            ) {
                Icon(
                    imageVector =
                        if (hasFilters) Icons.Default.FilterAlt
                        else Icons.Outlined.FilterAlt,
                    contentDescription = "Filter models",
                )
            }

            // Filter dropdown menu
            DropdownMenu(
                expanded = filteringConfig.isMenuVisible,
                onDismissRequest = { filteringConfig.toggleMenu(false) }
            ) {
                Text(
                    text = "Filter by",
                    style = MaterialTheme.typography.labelMedium,
                    modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
                )

                filteringConfig.filters.forEach { (filter, isEnabled) ->
                    DropdownMenuItem(
                        text = { Text(filter.displayName) },
                        leadingIcon = {
                            Checkbox(
                                checked = isEnabled,
                                onCheckedChange = null
                            )
                        },
                        onClick = { filteringConfig.onToggleFilter(filter, !isEnabled) }
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
                        filteringConfig.onClearFilters()
                        filteringConfig.toggleMenu(false)
                    }
                )
            }
        },
        floatingActionButton = {
            TooltipBox(
                positionProvider = TooltipDefaults.rememberTooltipPositionProvider(
                    TooltipAnchorPosition.Above),
                state = tooltipState,
                tooltip = {
                    PlainTooltip {
                        Text("Tap this button to install your first model!")
                    }
                },
                onDismissRequest = {}
            ) {
                FloatingActionButton(
                    onClick = { importingConfig.toggleMenu(true) },
                ) {
                    Icon(
                        imageVector = Icons.Default.Add,
                        contentDescription = "Add model"
                    )
                }
            }

            // Add model dropdown menu
            DropdownMenu(
                expanded = importingConfig.isMenuVisible,
                onDismissRequest = { importingConfig.toggleMenu(false) }
            ) {
                DropdownMenuItem(
                    text = { Text("Import a local GGUF model") },
                    leadingIcon = {
                        Icon(
                            imageVector = Icons.Default.FolderOpen,
                            contentDescription = "Import a local model on the device"
                        )
                    },
                    onClick = importingConfig.importFromLocal
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
                    onClick = importingConfig.importFromHuggingFace
                )
            }
        }
    )
}
