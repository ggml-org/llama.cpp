package com.example.llama.ui.screens

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.DialogProperties
import com.example.llama.data.model.ModelInfo
import com.example.llama.ui.components.InfoView
import com.example.llama.ui.components.ModelCardFullExpandable
import com.example.llama.ui.scaffold.ScaffoldEvent
import com.example.llama.util.formatFileByteSize
import com.example.llama.viewmodel.ModelManagementState
import com.example.llama.viewmodel.ModelManagementState.Deletion
import com.example.llama.viewmodel.ModelManagementState.Importation
import com.example.llama.viewmodel.ModelsManagementViewModel

/**
 * Screen for managing LLM models (view, download, delete)
 */
@Composable
fun ModelsManagementScreen(
    onScaffoldEvent: (ScaffoldEvent) -> Unit,
    viewModel: ModelsManagementViewModel,
) {
    // Data: models
    val filteredModels by viewModel.filteredModels.collectAsState()

    // Selection state
    val isMultiSelectionMode by viewModel.isMultiSelectionMode.collectAsState()
    val selectedModels by viewModel.selectedModels.collectAsState()

    // Filter state
    val activeFilters by viewModel.activeFilters.collectAsState()
    val activeFiltersCount by remember(activeFilters) {
        derivedStateOf { activeFilters.count { it.value }  }
    }

    // Model management state
    val managementState by viewModel.managementState.collectAsState()

    // UI states
    var expandedModels = remember { mutableStateMapOf<String, ModelInfo>() }

    BackHandler(
        enabled = isMultiSelectionMode
            || managementState is Importation.Importing
            || managementState is Deletion.Deleting
    ) {
        if (isMultiSelectionMode) {
            // Exit selection mode if in selection mode
            viewModel.toggleSelectionMode(false)
        } else {
            /* Ignore back press while processing model management requests */
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        if (filteredModels.isEmpty()) {
            val message = when (activeFiltersCount) {
                0 -> "Tap the \"+\" button to import a model!"
                1 -> "No models match the selected filter"
                else -> "No models match the selected filters"
            }
            InfoView(
                title = "No Models Available",
                icon = Icons.Default.FolderOpen,
                message = message,
            )
        } else {
            // Model cards
            LazyColumn(
                modifier = Modifier.fillMaxSize().padding(horizontal = 16.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                items(items = filteredModels, key = { it.id }) { model ->
                    val isSelected =
                        if (isMultiSelectionMode) selectedModels.contains(model.id) else null

                    ModelCardFullExpandable(
                        model = model,
                        isSelected = isSelected,
                        onSelected = {
                            if (isMultiSelectionMode) {
                                viewModel.toggleModelSelectionById(model.id)
                            }
                        },
                        isExpanded = expandedModels.contains(model.id),
                        onExpanded = { expanded ->
                            if (expanded) {
                                expandedModels.put(model.id, model)
                            } else {
                                expandedModels.remove(model.id)
                            }
                        }
                    )
                }
            }
        }

        // Model import progress overlay
        when (val state = managementState) {
            is Importation.Confirming -> {
                ImportProgressDialog(
                    fileName = state.fileName,
                    fileSize = state.fileSize,
                    isImporting = false,
                    progress = 0.0f,
                    onConfirm = {
                        viewModel.importLocalModelFile(state.uri, state.fileName, state.fileSize)
                    },
                    onCancel = { viewModel.resetManagementState() }
                )
            }

            is Importation.Importing -> {
                ImportProgressDialog(
                    fileName = state.fileName,
                    fileSize = state.fileSize,
                    isImporting = true,
                    isCancelling = state.isCancelling,
                    progress = state.progress,
                    onConfirm = {},
                    onCancel = { viewModel.cancelOngoingLocalModelImport() },
                )
            }

            is Importation.Error -> {
                ErrorDialog(
                    title = "Import Failed",
                    message = state.message,
                    onDismiss = { viewModel.resetManagementState() }
                )
            }

            is Importation.Success -> {
                LaunchedEffect(state) {
                    onScaffoldEvent(
                        ScaffoldEvent.ShowSnackbar(
                            message = "Imported model: ${state.model.name}"
                        )
                    )

                    viewModel.resetManagementState()
                }
            }

            is Deletion.Confirming -> {
                BatchDeleteConfirmationDialog(
                    count = state.models.size,
                    onConfirm = { viewModel.deleteModels(state.models) },
                    onDismiss = { viewModel.resetManagementState() },
                    isDeleting = false
                )
            }

            is Deletion.Deleting -> {
                BatchDeleteConfirmationDialog(
                    count = state.models.size,
                    onConfirm = { /* No-op during processing */ },
                    onDismiss = { /* No-op during processing */ },
                    isDeleting = true
                )
            }

            is Deletion.Error -> {
                ErrorDialog(
                    title = "Deletion Failed",
                    message = state.message,
                    onDismiss = { viewModel.resetManagementState() }
                )
            }

            is Deletion.Success -> {
                LaunchedEffect(state) {
                    viewModel.toggleSelectionMode(false)

                    val count = state.models.size
                    onScaffoldEvent(
                        ScaffoldEvent.ShowSnackbar(
                            message = "Deleted $count ${if (count > 1) "models" else "model"}.",
                            duration = SnackbarDuration.Long
                        )
                    )
                }
            }

            is ModelManagementState.Idle -> { /* Idle state, nothing to show */ }

            else -> TODO()
        }
    }

    // TODO-han.yin: UI TO BE IMPLEMENTED
    val huggingFaceModelsFlow = viewModel.huggingFaceModels
    LaunchedEffect(Unit) {
        huggingFaceModelsFlow.collect { models ->
            val message = models.fold(
                StringBuilder("Fetched ${models.size} models from HuggingFace")
            ) { builder, model ->
                builder.append(model.id).append("\n")
            }.toString()

            onScaffoldEvent(
                ScaffoldEvent.ShowSnackbar(
                    message = message,
                    duration = SnackbarDuration.Short
                )
            )
        }
    }
}

@Composable
private fun ImportProgressDialog(
    fileName: String,
    fileSize: Long,
    isImporting: Boolean,
    isCancelling: Boolean = false,
    progress: Float,
    onConfirm: () -> Unit,
    onCancel: () -> Unit
) {
    AlertDialog(
        onDismissRequest = {
            if (!isImporting) onCancel()
        },
        properties = DialogProperties(
            dismissOnBackPress = !isImporting,
            dismissOnClickOutside = !isImporting
        ),
        title = {
            Text(if (isImporting) "Importing Model" else "Confirm Import")
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(8.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Filename
                Text(
                    text = fileName,
                    style = MaterialTheme.typography.bodyMedium,
                    fontStyle = FontStyle.Italic,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    modifier = Modifier.fillMaxWidth()
                )

                if (isImporting) {
                    Spacer(modifier = Modifier.height(24.dp))

                    // Progress bar
                    LinearProgressIndicator(
                        progress = { progress },
                        modifier = Modifier.fillMaxWidth()
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    // Percentage text
                    Text(
                        text = "${(progress * 100).toInt()}%",
                        style = MaterialTheme.typography.bodyLarge
                    )
                } else {
                    // Show confirmation text
                    Spacer(modifier = Modifier.height(16.dp))

                    Text(
                        text = "Are you sure you want to import this model (${formatFileByteSize(fileSize)})? " +
                            "This may take up to several minutes.",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))

                // Additional information
                if (isImporting) {
                    Text(
                        text = "This may take several minutes for large models",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center
                    )
                } else if (isCancelling) {
                    Text(
                        text = "Cancelling ongoing import...",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.error
                    )
                }
            }
        },
        confirmButton = {
            // Only show confirm button in confirmation state
            if (!isImporting) {
                TextButton(onClick = onConfirm) { Text("Import") }
            }
        },
        dismissButton = {
            if (!isImporting || (progress < 0.7f && !isCancelling)) {
                TextButton(onClick = onCancel, enabled = !isCancelling) {
                    Text("Cancel")
                }
            }
        }
    )
}

@Composable
private fun BatchDeleteConfirmationDialog(
    count: Int,
    onConfirm: () -> Unit,
    onDismiss: () -> Unit,
    isDeleting: Boolean = false
) {
    AlertDialog(
        // Prevent dismissal when deletion is in progress
        onDismissRequest = {
            if (!isDeleting) onDismiss()
        },
        // Prevent dismissal via back button during deletion
        properties = DialogProperties(
            dismissOnBackPress = !isDeleting,
            dismissOnClickOutside = !isDeleting
        ),
        title = {
            Text("Confirm Deletion")
        },
        text = {
            Column {
                Text(
                    "Are you sure you want to delete "
                        + "$count selected ${if (count == 1) "model" else "models"}? "
                        + "This operation cannot be undone."
                )

                if (isDeleting) {
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
                        Text("Deleting models...")
                    }
                }
            }
        },
        confirmButton = {
            TextButton(
                onClick = onConfirm,
                enabled = !isDeleting
            ) {
                Text(
                    text = "Delete",
                    color = if (!isDeleting) MaterialTheme.colorScheme.error
                    else MaterialTheme.colorScheme.onSurface.copy(alpha = 0.3f)
                )
            }
        },
        dismissButton = {
            TextButton(
                onClick = onDismiss,
                enabled = !isDeleting
            ) {
                Text("Cancel")
            }
        }
    )
}

@Composable
private fun ErrorDialog(
    title: String,
    message: String,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = { Text(message) },
        confirmButton = {
            Button(onClick = onDismiss) {
                Text("OK")
            }
        }
    )
}
