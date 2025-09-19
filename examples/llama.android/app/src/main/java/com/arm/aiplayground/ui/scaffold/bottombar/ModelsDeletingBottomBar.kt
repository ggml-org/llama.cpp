package com.arm.aiplayground.ui.scaffold.bottombar

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ClearAll
import androidx.compose.material.icons.filled.DeleteForever
import androidx.compose.material.icons.filled.SelectAll
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable


@Composable
fun ModelsDeletingBottomBar(
    deleting: BottomBarConfig.Models.Deleting,
) {
    BottomAppBar(
        actions = {
            IconButton(onClick = { deleting.clearAllSelectedModels() }) {
                Icon(
                    imageVector = Icons.Default.ClearAll,
                    contentDescription = "Deselect all"
                )
            }

            IconButton(onClick = { deleting.selectAllFilteredModels() }) {
                Icon(
                    imageVector = Icons.Default.SelectAll,
                    contentDescription = "Select all"
                )
            }
        },
        floatingActionButton = {
            AnimatedVisibility(
                visible = deleting.selectedModels.isNotEmpty(),
                enter = scaleIn() + fadeIn(),
                exit = scaleOut() + fadeOut()
            ) {
                FloatingActionButton(
                    onClick = {
                        deleting.deleteSelected()
                    },
                    containerColor = MaterialTheme.colorScheme.error
                ) {
                    Icon(
                        imageVector = Icons.Default.DeleteForever,
                        contentDescription = "Delete selected models",
                        tint = MaterialTheme.colorScheme.onError,
                    )
                }
            }
        }
    )
}
