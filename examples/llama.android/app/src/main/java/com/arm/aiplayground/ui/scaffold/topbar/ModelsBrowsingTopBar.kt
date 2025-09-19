package com.arm.aiplayground.ui.scaffold.topbar

import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.PlainTooltip
import androidx.compose.material3.Text
import androidx.compose.material3.TooltipAnchorPosition
import androidx.compose.material3.TooltipBox
import androidx.compose.material3.TooltipDefaults
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.material3.rememberTooltipState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelsBrowsingTopBar(
    title: String,
    showTooltip: Boolean,
    showManagingToggle: Boolean,
    onToggleManaging: () -> Unit,
    onNavigateBack: (() -> Unit)? = null,
    onMenuOpen: (() -> Unit)? = null,
) {
    val tooltipState = rememberTooltipState(
        initialIsVisible = showTooltip,
        isPersistent = showTooltip
    )

    LaunchedEffect(showTooltip, showManagingToggle) {
        if (showTooltip && showManagingToggle) {
            tooltipState.show()
        }
    }

    TopAppBar(
        title = { Text(title) },
        navigationIcon = {
            when {
                onNavigateBack != null -> {
                    IconButton(onClick = onNavigateBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back"
                        )
                    }
                }

                onMenuOpen != null -> {
                    IconButton(onClick = onMenuOpen) {
                        Icon(
                            imageVector = Icons.Default.Menu,
                            contentDescription = "Menu"
                        )
                    }
                }
            }
        },
        actions = {
            if (showManagingToggle) {
                TooltipBox(
                    positionProvider = TooltipDefaults.rememberTooltipPositionProvider(
                        TooltipAnchorPosition.Below),
                    state = tooltipState,
                    tooltip = {
                        PlainTooltip {
                            Text("Tap this button to install another model or manage your models!")
                        }
                    },
                    onDismissRequest = {}
                ) {
                    ModelManageActionToggle(onToggleManaging)
                }
            }
        },
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = MaterialTheme.colorScheme.surface,
            titleContentColor = MaterialTheme.colorScheme.onSurface
        )
    )
}

@Composable
private fun ModelManageActionToggle(
    onToggleManaging: () -> Unit,
) {
    FilledTonalButton(
        modifier = Modifier.padding(end = 8.dp),
        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp),
        onClick = onToggleManaging
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                imageVector = Icons.Default.Build,
                contentDescription = "Manage models",
                tint = MaterialTheme.colorScheme.onSurface,
            )

            Text(
                modifier = Modifier.padding(start = 4.dp),
                text = "Manage",
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}
