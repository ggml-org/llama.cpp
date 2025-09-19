package com.arm.aiplayground.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AlertDialogDefaults
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.DialogProperties

data class InfoAction(
    val icon: ImageVector,
    val label: String,
    val onAction: () -> Unit
)

@Composable
fun InfoAlertDialog(
    modifier: Modifier = Modifier,
    isCritical: Boolean = false,
    allowDismiss: Boolean = false,
    onDismiss: () -> Unit = {},
    icon: ImageVector,
    title: String,
    message: String? = null,
    action: InfoAction? = null,
    confirmButton: @Composable () -> Unit = {},
) {
    AlertDialog(
        modifier = modifier,
        onDismissRequest = onDismiss,
        properties = DialogProperties(
            dismissOnBackPress = allowDismiss,
            dismissOnClickOutside = allowDismiss,
        ),
        icon = {
            Icon(
                imageVector = icon,
                contentDescription = null,
                modifier = Modifier.size(64.dp),
            )
        },
        title = {
            Text(
                modifier = Modifier.fillMaxWidth(),
                text = title,
                style = MaterialTheme.typography.headlineSmall,
                textAlign = TextAlign.Center,
                fontWeight = FontWeight.SemiBold
            )
        },
        text = {
            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                message?.let {
                    Text(
                        modifier = Modifier.padding(top = 8.dp),
                        text = it,
                        style = MaterialTheme.typography.bodyLarge,
                        textAlign = TextAlign.Center,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                action?.let {
                    Button(
                        modifier = Modifier.padding(top = 24.dp),
                        onClick = it.onAction,
                    ) {
                        Icon(
                            imageVector = it.icon,
                            contentDescription = null,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(it.label)
                    }
                }
            }

        },
        containerColor = if (isCritical) MaterialTheme.colorScheme.errorContainer else AlertDialogDefaults.containerColor,
        iconContentColor =  if (isCritical) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary,
        titleContentColor = if (isCritical) MaterialTheme.colorScheme.onErrorContainer else AlertDialogDefaults.titleContentColor,
        textContentColor = if (isCritical) MaterialTheme.colorScheme.onErrorContainer else AlertDialogDefaults.textContentColor,
        confirmButton = confirmButton,
    )
}

@Composable
fun InfoView(
    modifier: Modifier = Modifier,
    icon: ImageVector,
    title: String,
    message: String? = null,
    action: InfoAction? = null
) {
    InfoView(
        modifier = modifier,
        title = title,
        icon = {
            Icon(
                imageVector = icon,
                contentDescription = null,
                modifier = Modifier.size(64.dp),
                tint = MaterialTheme.colorScheme.primary.copy(alpha = 0.6f)
            )
        },
        message = message,
        action = action
    )
}

@Composable
fun InfoView(
    modifier: Modifier = Modifier,
    icon: @Composable () -> Unit,
    title: String,
    message: String? = null,
    action: InfoAction? = null
) {
    Column(
        modifier = modifier.padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        icon()

        Text(
            modifier = Modifier.padding(top = 16.dp),
            text = title,
            style = MaterialTheme.typography.headlineSmall,
            textAlign = TextAlign.Center,
            fontWeight = FontWeight.SemiBold
        )

        message?.let {
            Text(
                modifier = Modifier.padding(top = 8.dp),
                text = it,
                style = MaterialTheme.typography.bodyLarge,
                textAlign = TextAlign.Center,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        action?.let {
            Button(
                modifier = Modifier.padding(top = 24.dp),
                onClick = it.onAction,
            ) {
                Icon(
                    imageVector = it.icon,
                    contentDescription = null,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(it.label)
            }
        }
    }
}
