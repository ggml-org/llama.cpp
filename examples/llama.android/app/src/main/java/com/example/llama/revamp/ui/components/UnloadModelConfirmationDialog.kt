package com.example.llama.revamp.ui.components

import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable

/**
 * Confirmation dialog shown when the user attempts to navigate away from
 * a screen that would require unloading the current model.
 */
@Composable
fun UnloadModelConfirmationDialog(
    onConfirm: () -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text("Confirm Exit")
        },
        text = {
            Text(
                "Going back will unload the current model. " +
                    "This operation cannot be undone. " +
                    "Any unsaved conversation will be lost."
            )
        },
        confirmButton = {
            TextButton(onClick = onConfirm) {
                Text("Yes, Exit")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}
