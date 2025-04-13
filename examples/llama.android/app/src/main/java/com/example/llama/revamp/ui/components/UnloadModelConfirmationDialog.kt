package com.example.llama.revamp.ui.components

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
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * Confirmation dialog shown when the user attempts to navigate away from
 * a screen that would require unloading the current model.
 */
@Composable
fun UnloadModelConfirmationDialog(
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
                Text(
                    "Going back will unload the current model. " +
                        "This operation cannot be undone. " +
                        "Any unsaved conversation will be lost."
                )

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
