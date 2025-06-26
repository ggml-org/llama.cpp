package com.example.llama.ui.scaffold.bottombar

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Badge
import androidx.compose.material.icons.filled.ClearAll
import androidx.compose.material.icons.filled.Replay
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.outlined.Badge
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable

@Composable
fun BenchmarkBottomBar(
    engineIdle: Boolean,
    onShare: () -> Unit,
    onRerun: () -> Unit,
    onClear: () -> Unit,
    showModelCard: Boolean,
    onToggleModelCard: (Boolean) -> Unit,
) {
    BottomAppBar(
        actions = {
            IconButton(onClick = onRerun) {
                Icon(
                    imageVector = Icons.Default.Replay,
                    contentDescription = "Run the benchmark again",
                    tint =
                        if (engineIdle) MaterialTheme.colorScheme.onSurface
                        else MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.3f)
                )
            }

            IconButton(onClick = onClear) {
                Icon(
                    imageVector = Icons.Default.ClearAll,
                    contentDescription = "Clear benchmark results"
                )
            }

            IconButton(onClick = { onToggleModelCard(!showModelCard) } ) {
                Icon(
                    imageVector = if (showModelCard) Icons.Default.Badge else Icons.Outlined.Badge,
                    contentDescription = "${if (showModelCard) "Hide" else "Show"} model card"
                )
            }
        },
        floatingActionButton = {
            // Only show FAB if the benchmark result is ready
            AnimatedVisibility(
                visible = engineIdle,
                enter = scaleIn() + fadeIn(),
                exit = scaleOut() + fadeOut()
            ) {
                FloatingActionButton(
                    onClick = onShare,
                    containerColor = MaterialTheme.colorScheme.primary
                ) {
                    Icon(
                        imageVector = Icons.Default.Share,
                        contentDescription = "Share the benchmark results"
                    )
                }
            }
        }
    )
}
