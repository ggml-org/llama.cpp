package com.arm.aiplayground.ui.scaffold.bottombar

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.foundation.text.input.TextFieldState
import androidx.compose.foundation.text.input.clearText
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.Backspace
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.SearchOff
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable

@Composable
fun ModelsSearchingBottomBar(
    textFieldState: TextFieldState,
    onQuitSearching: () -> Unit,
    onSearch: (String) -> Unit, // TODO-han.yin: somehow this is unused?
    runActionConfig: BottomBarConfig.Models.RunActionConfig,
) {
    BottomAppBar(
        actions = {
            // Quit search action
            IconButton(onClick = onQuitSearching) {
                Icon(
                    imageVector = Icons.Default.SearchOff,
                    contentDescription = "Quit search mode"
                )
            }

            // Clear query action
            IconButton(onClick = { textFieldState.clearText() }) {
                Icon(
                    imageVector = Icons.AutoMirrored.Outlined.Backspace,
                    contentDescription = "Clear query text"
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
                    containerColor = MaterialTheme.colorScheme.primary,
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
