package com.example.llama.revamp.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Card
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.ui.components.ModelCardContentCore
import com.example.llama.revamp.util.FileType
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale


@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ModelDetailsScreen(
    model: ModelInfo,
) {
    Column(
        modifier = Modifier
            .padding(horizontal = 16.dp, vertical = 8.dp)
            .verticalScroll(rememberScrollState())
    ) {
        // Always show the core and expanded content
        ModelCardContentCore(model = model)

        Spacer(modifier = Modifier.height(16.dp))

        // Dates section
        Card(
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "Dates",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )

                Spacer(modifier = Modifier.height(8.dp))

                val dateFormatter = remember { SimpleDateFormat("MMM d, yyyy", Locale.getDefault()) }

                Column {
                    ListItem(
                        headlineContent = { Text("Added") },
                        supportingContent = {
                            Text(dateFormatter.format(Date(model.dateAdded)))
                        }
                    )

                    model.dateLastUsed?.let { lastUsed ->
                        ListItem(
                            headlineContent = { Text("Last used") },
                            supportingContent = {
                                Text(dateFormatter.format(Date(lastUsed)))
                            }
                        )
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Metadata sections - only show if data exists
        model.metadata.additional?.let { additional ->
            if (additional.tags?.isNotEmpty() == true || additional.languages?.isNotEmpty() == true) {
                Card(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "Additional Information",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )

                        Spacer(modifier = Modifier.height(8.dp))

                        additional.tags?.takeIf { it.isNotEmpty() }?.let { tags ->
                            Text(
                                text = "Tags",
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.Medium
                            )

                            Spacer(modifier = Modifier.height(4.dp))

                            FlowRow(
                                horizontalArrangement = Arrangement.spacedBy(4.dp),
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                tags.forEach { tag ->
                                    AssistChip(
                                        onClick = { /* No action */ },
                                        label = { Text(tag) }
                                    )
                                }
                            }

                            Spacer(modifier = Modifier.height(12.dp))
                        }

                        additional.languages?.takeIf { it.isNotEmpty() }?.let { languages ->
                            Text(
                                text = "Languages",
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.Medium
                            )

                            Spacer(modifier = Modifier.height(4.dp))

                            FlowRow(
                                horizontalArrangement = Arrangement.spacedBy(4.dp),
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                languages.forEach { language ->
                                    AssistChip(
                                        onClick = { /* No action */ },
                                        label = { Text(language) }
                                    )
                                }
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))
            }
        }

        // Technical details section
        Card(
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text = "Technical Details",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )

                Spacer(modifier = Modifier.height(8.dp))

                // Architecture details
                model.metadata.architecture?.let { architecture ->
                    ListItem(
                        headlineContent = { Text("Architecture") },
                        supportingContent = { Text(architecture.architecture ?: "Unknown") }
                    )

                    architecture.fileType?.let {
                        ListItem(
                            headlineContent = { Text("Quantization") },
                            supportingContent = { Text(FileType.fromCode(it).label) }
                        )
                    }

                    architecture.vocabSize?.let {
                        ListItem(
                            headlineContent = { Text("Vocabulary Size") },
                            supportingContent = { Text(it.toString()) }
                        )
                    }
                }

                // Context length
                model.metadata.dimensions?.contextLength?.let {
                    ListItem(
                        headlineContent = { Text("Context Length") },
                        supportingContent = { Text("$it tokens") }
                    )
                }

                // ROPE params if available
                model.metadata.rope?.let { rope ->
                    ListItem(
                        headlineContent = { Text("RoPE Base") },
                        supportingContent = {
                            rope.frequencyBase?.let { Text(it.toString()) }
                        }
                    )

                    ListItem(
                        headlineContent = { Text("RoPE Scaling") },
                        supportingContent = {
                            if (rope.scalingType != null && rope.scalingFactor != null) {
                                Text("${rope.scalingType}: ${rope.scalingFactor}")
                            } else {
                                Text("None")
                            }
                        }
                    )
                }

                // File size
                ListItem(
                    headlineContent = { Text("File Size") },
                    supportingContent = { Text(model.formattedFileSize) }
                )

                // File path
                ListItem(
                    headlineContent = { Text("File Path") },
                    supportingContent = {
                        Text(
                            text = model.path,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                )
            }
        }

        // Add author and attribution section if available
        model.metadata.author?.let { author ->
            if (author.author != null || author.organization != null || author.license != null) {
                Spacer(modifier = Modifier.height(16.dp))

                Card(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = "Attribution",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )

                        Spacer(modifier = Modifier.height(8.dp))

                        author.author?.let {
                            ListItem(
                                headlineContent = { Text("Author") },
                                supportingContent = { Text(it) }
                            )
                        }

                        author.organization?.let {
                            ListItem(
                                headlineContent = { Text("Organization") },
                                supportingContent = { Text(it) }
                            )
                        }

                        author.license?.let {
                            ListItem(
                                headlineContent = { Text("License") },
                                supportingContent = { Text(it) }
                            )
                        }

                        author.url?.let {
                            ListItem(
                                headlineContent = { Text("URL") },
                                supportingContent = {
                                    Text(
                                        text = it,
                                        maxLines = 2,
                                        overflow = TextOverflow.Ellipsis,
                                        color = MaterialTheme.colorScheme.primary
                                    )
                                }
                            )
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(24.dp))
    }
}
