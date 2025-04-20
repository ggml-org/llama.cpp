package com.example.llama.revamp.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LocalMinimumInteractiveComponentSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.example.llama.revamp.data.model.ModelInfo
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Displays model information in a card format with core details.
 *
 * This component shows essential model information like name, context length,
 * architecture, and quantization in a compact card format. It can be used
 * in lists where only basic information is needed.
 *
 * @param model The model information to display
 * @param onClick Action to perform when the card is clicked
 * @param isSelected Optional selection state (shows checkbox when not null)
 */
@Composable
fun ModelCardCore(
    model: ModelInfo,
    onClick: () -> Unit,
    isSelected: Boolean? = null,
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        colors = when (isSelected) {
            true -> CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
            false -> CardDefaults.cardColors()
            else -> CardDefaults.cardColors()
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

            // Core model info
            ModelCardContentCore(
                model = model,
                modifier = Modifier.weight(1f)
            )
        }
    }
}

@Composable
fun ModelCardContentCore(
    model: ModelInfo,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        // Row 1: Model full name
        Text(
            text = model.formattedFullName,
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold
        )

        Spacer(modifier = Modifier.height(12.dp))

        // Row 2: Context length, size label
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            ModelCardContentField("Context", model.formattedContextLength)

            ModelCardContentField("Params", model.formattedParamSize)
        }

        Spacer(modifier = Modifier.height(8.dp))

        // Row 3: Architecture, quantization, formatted size
        Row(
            modifier = Modifier.fillMaxWidth(),
        ) {
            ModelCardContentField("Architecture", model.formattedArchitecture)

            Spacer(modifier = Modifier.weight(1f))

            ModelCardContentField(model.formattedQuantization, model.formattedFileSize)
        }
    }
}

/**
 * Displays model information in a card format with expandable details.
 *
 * This component shows essential model information and can be expanded to show
 * additional details such as dates, tags, and languages. The expanded state is
 * toggled by clicking on the content area of the card.
 *
 * @param model The model information to display
 * @param onClick Action to perform when the card is clicked (for selection)
 * @param expanded Whether additional details are currently shown
 * @param isSelected Optional selection state (shows checkbox when not null)
 */
@Composable
fun ModelCardExpandable(
    model: ModelInfo,
    onClick: () -> Unit,
    expanded: Boolean,
    isSelected: Boolean? = null,
) {
    var isExpanded by remember { mutableStateOf(expanded) }

    CompositionLocalProvider(LocalMinimumInteractiveComponentSize provides Dp.Unspecified) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .clickable { isExpanded = !isExpanded }
            ,
            colors = when (isSelected) {
                true -> CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
                false -> CardDefaults.cardColors()
                else -> CardDefaults.cardColors()
            },
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
        ) {
            Column(
                modifier = Modifier.padding(bottom = 16.dp)
            ) {
                Row(
                    verticalAlignment = Alignment.Top
                ) {
                    // Show checkbox if in selection mode
                    if (isSelected != null) {
                        Checkbox(
                            checked = isSelected,
                            onCheckedChange = { onClick() },
                            modifier = Modifier.padding(top = 16.dp, start = 16.dp)
                        )
                    }

                    Box(
                        modifier = Modifier.weight(1f)
                            .padding(start = 16.dp, top = 16.dp, end = 16.dp)
                    ) {
                        // Core content always visible
                        ModelCardContentCore(model = model)
                    }
                }

                // Expandable content
                AnimatedVisibility(
                    visible = isExpanded,
                    enter = expandVertically() + fadeIn(),
                    exit = shrinkVertically() + fadeOut()
                ) {
                    Box(
                        modifier = Modifier.weight(1f).padding(horizontal = 16.dp)
                    ) {
                        ExpandableModelDetails(model = model)
                    }
                }
            }
        }
    }
}

// Displays expandable details for a model, to be used only inside ModelCardExpandable.
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ExpandableModelDetails(model: ModelInfo) {
    val dateFormatter = remember { SimpleDateFormat("MMM d, yyyy", Locale.getDefault()) }

    Column(modifier = Modifier.padding(top = 16.dp)) {
        // Divider between core and expanded content
        HorizontalDivider(modifier = Modifier.padding(bottom = 16.dp))

        // Dates
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            // Added date
            ModelCardContentField("Added", dateFormatter.format(Date(model.dateAdded)))

            // Last used date (if available)
            model.dateLastUsed?.let { lastUsed ->
                ModelCardContentField("Last used", dateFormatter.format(Date(lastUsed)))
            }
        }

        // Tags (if available)
        model.tags?.let { tags ->
            Spacer(modifier = Modifier.height(8.dp))

            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                tags.forEach { tag ->
                    AssistChip(
                        onClick = { /* No action */ },
                        label = {
                            Text(
                                text = tag,
                                style = MaterialTheme.typography.bodySmall,
                                fontStyle = FontStyle.Italic,
                                fontWeight = FontWeight.Light,
                                color = MaterialTheme.colorScheme.onSurface
                            )
                        }
                    )
                }
            }
        }

        // Languages (if available)
        model.languages?.let { languages ->
            Spacer(modifier = Modifier.height(8.dp))

            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                languages.forEach { language ->
                    AssistChip(
                        onClick = { /* No action */ },
                        label = {
                            Text(
                                text = language,
                                style = MaterialTheme.typography.bodySmall,
                                fontStyle = FontStyle.Italic,
                                fontWeight = FontWeight.Light,
                                color = MaterialTheme.colorScheme.onSurface
                            )
                        }
                    )
                }
            }
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
            ModelCardContentCore(
                model = model
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

@Composable
private fun ModelCardContentField(name: String, value: String) =
    Row {
        Text(
            text = name,
            style = MaterialTheme.typography.bodyLarge,
            fontWeight = FontWeight.Normal,
            color = MaterialTheme.colorScheme.onSurface
        )

        Spacer(modifier = Modifier.width(8.dp))

        Text(
            text = value,
            style = MaterialTheme.typography.bodyLarge,
            fontWeight = FontWeight.Light,
            fontStyle = FontStyle.Italic,
            color = MaterialTheme.colorScheme.onSurface
        )
    }
