package com.example.llama.revamp.ui.components

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.llama.revamp.data.model.ModelInfo
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Reusable card component for displaying model information.
 * Can be configured for selection mode or normal display mode.
 */
@Composable
fun ModelCard(
    model: ModelInfo,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    isSelected: Boolean? = null,  // `null`: not in selection mode, otherwise true/false
    actionButton: @Composable (() -> Unit)? = null
) {
    Card(
        modifier = modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        colors = when {
            isSelected == true -> CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
            isSelected == false -> CardDefaults.cardColors()
            else -> CardDefaults.cardColors()  // Not in selection mode
        },
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Show checkbox if in selection mode
            if (isSelected != null) {
                Checkbox(
                    checked = isSelected,
                    onCheckedChange = { onClick() },
                    modifier = Modifier.padding(end = 8.dp)
                )
            }

            // Model info
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = model.name,
                    style = MaterialTheme.typography.titleMedium
                )

                Spacer(modifier = Modifier.height(4.dp))

                // Model details row (parameters, quantization, size)
                Row {
                    if (model.parameters != null) {
                        Text(
                            text = model.parameters,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    if (model.quantization != null) {
                        Text(
                            text = " • ${model.quantization}",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    Text(
                        text = " • ${model.formattedSize}",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                Spacer(modifier = Modifier.height(4.dp))

                // Context length
                if (model.contextLength != null) {
                    Text(
                        text = "Context Length: ${model.contextLength}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                // Last used date
                model.lastUsed?.let { lastUsed ->
                    val dateFormat = SimpleDateFormat("MMM d, yyyy", Locale.getDefault())
                    Text(
                        text = "Last used: ${dateFormat.format(Date(lastUsed))}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            // Custom action button or built-in ones
            actionButton?.invoke() ?: Spacer(modifier = Modifier.width(8.dp))
        }
    }
}

/**
 * Predefined action buttons for ModelCard
 */
object ModelCardActions {
    @Composable
    fun PlayButton(onClick: () -> Unit) {
        IconButton(onClick = onClick) {
            Icon(
                imageVector = Icons.Default.PlayArrow,
                contentDescription = "Select model",
                tint = MaterialTheme.colorScheme.primary
            )
        }
    }

    @Composable
    fun InfoButton(onClick: () -> Unit) {
        IconButton(onClick = onClick) {
            Icon(
                imageVector = Icons.Default.Info,
                contentDescription = "Model details"
            )
        }
    }
}
