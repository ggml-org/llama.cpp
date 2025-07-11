package com.example.llama.ui.scaffold.bottombar

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.input.clearText
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Sort
import androidx.compose.material.icons.automirrored.outlined.Backspace
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.FilterAlt
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.SearchOff
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
import androidx.compose.ui.unit.dp
import com.example.llama.data.model.ModelSortOrder

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
            // Only show FAB if a model is selected
            AnimatedVisibility(
                visible = runAction.preselection != null,
                enter = scaleIn() + fadeIn(),
                exit = scaleOut() + fadeOut()
            ) {
                FloatingActionButton(
                    onClick = { runAction.preselection?.let { runAction.onClickRun(it) } },
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
