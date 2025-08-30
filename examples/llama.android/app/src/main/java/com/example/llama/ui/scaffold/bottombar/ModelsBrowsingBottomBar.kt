package com.example.llama.ui.scaffold.bottombar

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Sort
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.FilterAlt
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Search
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
fun ModelsBrowsingBottomBar(
    onToggleSearching: () -> Unit,
    sortingConfig: BottomBarConfig.Models.Browsing.SortingConfig,
    filteringConfig: BottomBarConfig.Models.Browsing.FilteringConfig,
    runActionConfig: BottomBarConfig.Models.RunActionConfig,
) {
    BottomAppBar(
        actions = {
            // Enter search action
            IconButton(onClick = onToggleSearching) {
                Icon(
                    imageVector = Icons.Default.Search,
                    contentDescription = "Search models"
                )
            }

            // Sorting action
            IconButton(onClick = { sortingConfig.toggleMenu(true) }) {
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

            // Filter action
            IconButton(onClick = { filteringConfig.toggleMenu(true) }) {
                Icon(
                    imageVector =
                        if (filteringConfig.isActive) Icons.Default.FilterAlt
                        else Icons.Outlined.FilterAlt,
                    contentDescription = "Filter models"
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
            // Only show FAB if a model is selected
            AnimatedVisibility(
                visible = runActionConfig.preselectedModelToRun != null,
                enter = scaleIn() + fadeIn(),
                exit = scaleOut() + fadeOut()
            ) {
                FloatingActionButton(
                    onClick = {
                        runActionConfig.preselectedModelToRun?.let {
                            runActionConfig.onClickRun(it)
                        }
                    },
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
