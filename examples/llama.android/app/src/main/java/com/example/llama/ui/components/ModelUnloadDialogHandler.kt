package com.example.llama.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.llama.viewmodel.UnloadModelState

/**
 * Reusable component for handling model unloading dialogs
 *
 * @param [UnloadModelState]:
 *  - Hidden: default state without showing any UI
 *  - Confirming: show dismissible [UnloadModelDialog] and asks for user confirmation to unload current model
 *  - Unloading: show non-dismissible [UnloadModelDialog] while unloading model
 *  - Error: show [UnloadModelErrorDialog] to prompt error message to user
 */
@Composable
fun ModelUnloadDialogHandler(
    message: String,
    unloadModelState: UnloadModelState,
    onUnloadConfirmed: (onNavigateBack: () -> Unit) -> Unit,
    onUnloadDismissed: () -> Unit,
    onNavigateBack: () -> Unit
) {
    when (unloadModelState) {
        is UnloadModelState.Confirming -> {
            UnloadModelDialog(
                message = message,
                onConfirm = {
                    onUnloadConfirmed(onNavigateBack)
                },
                onDismiss = onUnloadDismissed,
                isUnloading = false,
            )
        }
        is UnloadModelState.Unloading -> {
            UnloadModelDialog(
                message = message,
                onConfirm = {
                    onUnloadConfirmed(onNavigateBack)
                },
                onDismiss = onUnloadDismissed,
                isUnloading = true
            )
        }
        is UnloadModelState.Error -> {
            UnloadModelErrorDialog(
                errorMessage = unloadModelState.message,
                onConfirm = {
                    onUnloadDismissed()
                    onNavigateBack()
                },
                onDismiss = onUnloadDismissed
            )
        }
        is UnloadModelState.Hidden -> {
            // Dialog not shown
        }
    }
}

@Composable
private fun UnloadModelDialog(
    message: String,
    onConfirm: () -> Unit,
    onDismiss: () -> Unit,
    isUnloading: Boolean = false
) {
    AlertDialog(
        onDismissRequest = {
            // Ignore dismiss requests while unloading the model
            if (!isUnloading) onDismiss()
        },
        title = {
            Text("Confirm Exit")
        },
        text = {
            Column {
                Text(message)

                if (isUnloading) {
                    Spacer(modifier = Modifier.height(16.dp))
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.Center,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(24.dp),
                            strokeWidth = 2.dp
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Unloading model...")
                    }
                }
            }
        },
        confirmButton = {
            TextButton(
                onClick = onConfirm,
                enabled = !isUnloading
            ) {
                Text("Yes, Exit")
            }
        },
        dismissButton = {
            TextButton(
                onClick = onDismiss,
                enabled = !isUnloading
            ) {
                Text("Cancel")
            }
        }
    )
}

@Composable
private fun UnloadModelErrorDialog(
    errorMessage: String,
    onConfirm: () -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(
                text = "Error Unloading Model",
                color = MaterialTheme.colorScheme.error
            )
        },
        text = {
            Column {
                Text(
                    text = errorMessage,
                    style = MaterialTheme.typography.bodyMedium
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = "You may need to restart the app if this problem persists.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        },
        confirmButton = {
            TextButton(onClick = onConfirm) { Text("Continue") }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Stay on Screen") }
        }
    )
}
