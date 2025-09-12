package com.example.llama.ui.components

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
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalMinimumInteractiveComponentSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.example.llama.data.model.ModelInfo
import com.example.llama.util.languageCodeToFlagEmoji
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Displays model information in a card format expandable with core details
 * such as context length, architecture, quantization and file size
 *
 * @param model The model information to display
 * @param isExpanded Whether additional details is expanded or shrunk
 * @param onExpanded Action to perform when the card is expanded or shrunk
 */
@Composable
fun ModelCardCoreExpandable(
    model: ModelInfo,
    containerColor: Color = MaterialTheme.colorScheme.primaryContainer,
    isExpanded: Boolean = false,
    onExpanded: ((Boolean) -> Unit)? = null
) = ModelCardCoreExpandable(
    model = model,
    containerColor = containerColor,
    isExpanded = isExpanded,
    onExpanded = onExpanded
) {
    Spacer(modifier = Modifier.height(8.dp))

    // Row 2: Context length, size label
    ModelCardContentContextRow(model)

    Spacer(modifier = Modifier.height(8.dp))

    // Row 3: Architecture, quantization, formatted size
    ModelCardContentArchitectureRow(model)
}

/**
 * Displays model information in a card format expandable with customizable extra content.
 *
 * @param model The model information to display
 * @param isExpanded Whether additional details is expanded or shrunk
 * @param onExpanded Action to perform when the card is expanded or shrunk
 */
@Composable
fun ModelCardCoreExpandable(
    model: ModelInfo,
    containerColor: Color = MaterialTheme.colorScheme.primaryContainer,
    isExpanded: Boolean = false,
    onExpanded: ((Boolean) -> Unit)? = null,
    expandableSection: @Composable () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onExpanded?.invoke(!isExpanded) },
        colors = when (isExpanded) {
            true -> CardDefaults.cardColors(containerColor = containerColor)
            false -> CardDefaults.cardColors()
        },
        elevation = CardDefaults.cardElevation(defaultElevation = 12.dp)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
        ) {
            // Row 1: Model full name + chevron
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                ModelCardContentTitleRow(model)

                CompositionLocalProvider(
                    LocalMinimumInteractiveComponentSize provides Dp.Unspecified
                ) {
                    IconButton(onClick = { onExpanded?.invoke(!isExpanded) }) {
                        Icon(
                            imageVector = if (isExpanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                            contentDescription = "Tap to ${if (isExpanded) "shrink" else "expand"} model card"
                        )
                    }
                }
            }

            // Expandable content
            AnimatedVisibility(
                visible = isExpanded,
                enter = expandVertically() + fadeIn(),
                exit = shrinkVertically() + fadeOut()
            ) {
                Column(
                    modifier = Modifier.weight(1f)
                ) {
                    expandableSection()
                }
            }
        }
    }
}

/**
 * Displays model information in a card format with expandable details.
 *
 * This component shows essential model information and can be expanded to show
 * additional details such as dates, tags, and languages.
 * The expanded state is toggled by clicking on the content area of the card.
 *
 * @param model The model information to display
 * @param isSelected Optional selection state (shows checkbox when not null)
 * @param onSelected Action to perform when the card is selected (in multi-selection mode)
 * @param isExpanded Whether additional details is expanded or shrunk
 * @param onExpanded Action to perform when the card is expanded or shrunk
 */
@Composable
fun ModelCardFullExpandable(
    model: ModelInfo,
    containerColor: Color = MaterialTheme.colorScheme.primaryContainer,
    isSelected: Boolean? = null,
    onSelected: ((Boolean) -> Unit)? = null,
    isExpanded: Boolean = false,
    onExpanded: ((Boolean) -> Unit)? = null,
) {
    CompositionLocalProvider(LocalMinimumInteractiveComponentSize provides Dp.Unspecified) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .clickable { onExpanded?.invoke(!isExpanded) },
            colors = when (isSelected) {
                true -> CardDefaults.cardColors(containerColor = containerColor)
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
                    isSelected?.let { selected ->
                        Checkbox(
                            checked = selected,
                            onCheckedChange = { onSelected?.invoke(it) },
                            modifier = Modifier.padding(top = 16.dp, start = 16.dp)
                        )
                    }

                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .padding(start = 16.dp, top = 16.dp, end = 16.dp)
                    ) {
                        // Row 1: Model full name
                        ModelCardContentTitleRow(model)
                    }
                }

                Spacer(modifier = Modifier.height(12.dp))

                Column(modifier = Modifier.padding(horizontal = 16.dp)) {
                    // Row 2: Context length, size label
                    ModelCardContentContextRow(model)

                    Spacer(modifier = Modifier.height(8.dp))

                    // Row 3: Architecture, quantization, formatted size
                    ModelCardContentArchitectureRow(model)
                }

                // Expandable content
                AnimatedVisibility(
                    visible = isExpanded,
                    enter = expandVertically() + fadeIn(),
                    exit = shrinkVertically() + fadeOut()
                ) {
                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .padding(horizontal = 16.dp)
                    ) {
                        Column(modifier = Modifier.padding(top = 12.dp)) {
                            // Divider between core and expanded content
                            HorizontalDivider(modifier = Modifier.padding(bottom = 12.dp))

                            // Row 4: Dates
                            ModelCardContentDatesRow(model)

                            // Row 5: Tags
                            model.tags?.let { tags ->
                                Spacer(modifier = Modifier.height(12.dp))

                                Text(
                                    text = "Tags",
                                    style = MaterialTheme.typography.bodyLarge,
                                    fontWeight = FontWeight.Normal,
                                    color = MaterialTheme.colorScheme.onSurface
                                )

                                Spacer(modifier = Modifier.height(6.dp))

                                ModelCardContentTagsSection(tags)
                            }

                            // Row 6: Languages
                            model.languages?.let { languages ->
                                Spacer(modifier = Modifier.height(12.dp))

                                Text(
                                    text = "Supported languages",
                                    style = MaterialTheme.typography.bodyLarge,
                                    fontWeight = FontWeight.Normal,
                                    color = MaterialTheme.colorScheme.onSurface
                                )

                                Spacer(modifier = Modifier.height(6.dp))

                                ModelCardContentLanguagesSections(languages)
                            }
                        }
                    }
                }
            }
        }
    }
}

/*
 *
 * Individual components to be orchestrated into a model card's content
 *
 */

@Composable
fun ModelCardContentTitleRow(model: ModelInfo) =
    Text(
        text = model.formattedFullName,
        style = MaterialTheme.typography.titleLarge,
        fontWeight = FontWeight.Medium
    )

@Composable
fun ModelCardContentContextRow(model: ModelInfo) =
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        ModelCardContentField("Context", model.formattedContextLength)

        ModelCardContentField("Params", model.formattedParamSize)
    }

@Composable
fun ModelCardContentArchitectureRow(model: ModelInfo) =
    Row(
        modifier = Modifier.fillMaxWidth(),
    ) {
        ModelCardContentField("Architecture", model.formattedArchitecture)

        Spacer(modifier = Modifier.weight(1f))

        ModelCardContentField(model.formattedQuantization, model.formattedFileSize)
    }

@Composable
fun ModelCardContentDatesRow(model: ModelInfo) {
    val dateFormatter = remember { SimpleDateFormat("MMM d, yyyy", Locale.getDefault()) }

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
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ModelCardContentTagsSection(tags: List<String>) =
    FlowRow(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        tags.forEach { tag ->
            AssistChip(
                enabled = false,
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

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ModelCardContentLanguagesSections(languages: List<String>) =
    FlowRow(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        languages.mapNotNull { languageCodeToFlagEmoji(it) }.forEach { emoji ->
            AssistChip(
                enabled = false,
                onClick = { /* No action */ },
                label = {
                    Text(
                        text = emoji,
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.Light,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
            )
        }
    }

@Composable
fun ModelCardContentField(name: String, value: String) =
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
