package com.example.llama.revamp.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
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
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.llama.revamp.data.model.ModelInfo
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Standard model card for selection lists
 */
@Composable
fun ModelCard(
    model: ModelInfo,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    isSelected: Boolean? = null,
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
            ModelInfoContent(
                model = model,
                modifier = Modifier.weight(1f),
                contentPadding = PaddingValues(0.dp)
            )

            // Custom action button or built-in ones
            actionButton?.invoke() ?: Spacer(modifier = Modifier.width(8.dp))
        }
    }
}

/**
 * Expandable card that shows model info and system prompt
 */
@Composable
fun ModelCardWithSystemPrompt(
    model: ModelInfo,
    systemPrompt: String?,
    modifier: Modifier = Modifier,
    initiallyExpanded: Boolean = false
) {
    var expanded by remember { mutableStateOf(initiallyExpanded) }

    Card(
        modifier = modifier
            .fillMaxWidth(),
        onClick = {
            if (systemPrompt != null) {
                expanded = !expanded
            }
        }
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            // Model info section
            ModelInfoContent(
                model = model,
                contentPadding = PaddingValues(0.dp)
            )

            // Add divider between model info and system prompt
            if (!systemPrompt.isNullOrBlank()) {
                HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "System Prompt",
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.weight(1f)
                    )

                    Text(
                        text = if (expanded) "Hide" else "Show",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.primary
                    )
                }

                AnimatedVisibility(
                    visible = expanded,
                    enter = fadeIn() + expandVertically(),
                    exit = fadeOut() + shrinkVertically()
                ) {
                    Text(
                        text = systemPrompt,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 8.dp)
                    )
                }
            }
        }
    }
}


/**
 * Core model info display component that can be used by other card variants
 */
@Composable
private fun ModelInfoContent(
    model: ModelInfo,
    modifier: Modifier = Modifier,
    contentPadding: PaddingValues = PaddingValues(16.dp)
) {
    Column(modifier = modifier.padding(contentPadding)) {
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
