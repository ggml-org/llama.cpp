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
import androidx.compose.ui.unit.dp
import com.example.llama.data.model.ModelSortOrder

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelsBrowsingBottomBar(
    isSearchingEnabled: Boolean,
    onToggleSearching: () -> Unit,
    sortingConfig: BottomBarConfig.Models.Browsing.SortingConfig,
    filteringConfig: BottomBarConfig.Models.Browsing.FilteringConfig,
    runActionConfig: BottomBarConfig.Models.RunActionConfig,
) {
    val tooltipState = rememberTooltipState(
        initialIsVisible = runActionConfig.showTooltip,
        isPersistent = runActionConfig.showTooltip
    )

    LaunchedEffect(runActionConfig.preselectedModelToRun) {
        if (runActionConfig.showTooltip && runActionConfig.preselectedModelToRun != null) {
            tooltipState.show()
        }
    }

    BottomAppBar(
        actions = {
            // Enter search action
            IconButton(
                enabled = isSearchingEnabled,
                onClick = onToggleSearching
            ) {
                Icon(
                    imageVector = Icons.Default.Search,
                    contentDescription = "Search models"
                )
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

            // Filter action
            val hasFilters = filteringConfig.filters.any { it.value }
            IconButton(
                enabled = filteringConfig.isEnabled,
                colors = IconButtonDefaults.iconButtonColors().copy(
                    contentColor = if (hasFilters) MaterialTheme.colorScheme.primary
                    else MaterialTheme.colorScheme.onSurfaceVariant
                ),
                onClick = { filteringConfig.toggleMenu(true) }
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
                        Text("Tap this button to run your first model!")
                    }
                },
                onDismissRequest = {}
            ) {
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
                        containerColor = MaterialTheme.colorScheme.primary,
                    ) {
                        Icon(
                            imageVector = Icons.Default.PlayArrow,
                            contentDescription = "Run with selected model",
                        )
                    }
                }
            }
        }
    )
}
