package com.example.llama.ui.screens

import android.content.Context
import android.content.Intent
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.basicMarquee
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
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
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.automirrored.filled.Help
import androidx.compose.material.icons.automirrored.outlined.ContactSupport
import androidx.compose.material.icons.filled.ArrowForward
import androidx.compose.material.icons.filled.Attribution
import androidx.compose.material.icons.filled.Celebration
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Today
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.DialogProperties
import androidx.core.net.toUri
import com.example.llama.data.model.ModelInfo
import com.example.llama.data.source.remote.HuggingFaceModel
import com.example.llama.ui.components.InfoAction
import com.example.llama.ui.components.InfoView
import com.example.llama.ui.components.ModelCardFullExpandable
import com.example.llama.ui.scaffold.ScaffoldEvent
import com.example.llama.util.formatContextLength
import com.example.llama.util.formatFileByteSize
import com.example.llama.viewmodel.ModelManagementState
import com.example.llama.viewmodel.ModelManagementState.Deletion
import com.example.llama.viewmodel.ModelManagementState.Download
import com.example.llama.viewmodel.ModelManagementState.Importation
import com.example.llama.viewmodel.ModelScreenUiMode
import com.example.llama.viewmodel.ModelsManagementViewModel
import com.example.llama.viewmodel.ModelsViewModel
import java.text.SimpleDateFormat
import java.util.Locale

/**
 * Screen for managing LLM models (view, download, delete)
 */
@Composable
fun ModelsManagementAndDeletingScreen(
    showModelImportTooltip: Boolean,
    onFirstModelImportSuccess: (ModelInfo) -> Unit,
    filteredModels: List<ModelInfo>?,
    activeFiltersCount: Int,
    isDeleting: Boolean,
    onScaffoldEvent: (ScaffoldEvent) -> Unit,
    modelsViewModel: ModelsViewModel,
    managementViewModel: ModelsManagementViewModel,
) {
    val context = LocalContext.current

    // Selection state
    val selectedModels by managementViewModel.selectedModelsToDelete.collectAsState()

    // Model management state
    val managementState by managementViewModel.managementState.collectAsState()

    // UI states
    val expandedModels = remember { mutableStateMapOf<String, ModelInfo>() }

    BackHandler(
        enabled = managementState is Importation.Importing
            || managementState is Deletion.Deleting
    ) {
        /* Ignore back press while processing model management requests */
    }

    Box(modifier = Modifier.fillMaxSize()) {
        if (filteredModels == null) {
            ModelsLoadingInProgressView()
        } else if (filteredModels.isEmpty()) {
            // Prompt the user to import a model
            val title = when (activeFiltersCount) {
                0 -> "Install your first model"
                1 -> "No models match\n the selected filter"
                else -> "No models match\n the selected filters"
            }

            val message = "If you have already obtained GGUF models on your computer, " +
                    "please transfer it onto your device, and then select \"Import a local GGUF model\".\n\n" +
                    "Otherwise, select \"Download from Hugging Face\" and pick one of the pre-selected models."

            InfoView(
                modifier = Modifier.fillMaxSize(0.9f).align(Alignment.Center),
                title = title,
                icon = Icons.Default.Info,
                message = message,
                action = InfoAction(
                    label = "Learn more",
                    icon = Icons.AutoMirrored.Default.Help,
                    onAction = {
                        val url = "https://huggingface.co/docs/hub/en/gguf"
                        val intent = Intent(Intent.ACTION_VIEW, url.toUri())
                        context.startActivity(intent)
                    }
                )
            )
        } else {
            // Model cards
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(12.dp),
                contentPadding = PaddingValues(vertical = 12.dp, horizontal = 16.dp),
            ) {
                items(items = filteredModels, key = { it.id }) { model ->
                    val isSelected =
                        if (isDeleting) selectedModels.contains(model.id) else null

                    ModelCardFullExpandable(
                        model = model,
                        isSelected = isSelected,
                        onSelected = {
                            if (isDeleting) {
                                managementViewModel.toggleModelSelectionById(filteredModels, model.id)
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
                ImportFromLocalFileDialog(
                    fileName = state.fileName,
                    fileSize = state.fileSize,
                    isImporting = false,
                    progress = 0.0f,
                    onConfirm = {
                        managementViewModel.importLocalModelFileConfirmed(
                            state.uri, state.fileName, state.fileSize
                        )
                    },
                    onCancel = { managementViewModel.resetManagementState() }
                )
            }

            is Importation.Importing -> {
                ImportFromLocalFileDialog(
                    fileName = state.fileName,
                    fileSize = state.fileSize,
                    isImporting = true,
                    isCancelling = state.isCancelling,
                    progress = state.progress,
                    onConfirm = {},
                    onCancel = { managementViewModel.cancelOngoingLocalModelImport() },
                )
            }

            is Importation.Error -> {
                ErrorDialog(
                    context = context,
                    title = "Import Failed",
                    message = state.message,
                    learnMoreUrl = state.learnMoreUrl,
                    onDismiss = { managementViewModel.resetManagementState() }
                )
            }

            is Importation.Success -> {
                if (showModelImportTooltip) {
                    FirstModelImportSuccessDialog {
                        onFirstModelImportSuccess(state.model)
                        managementViewModel.resetManagementState()
                    }
                } else {
                    LaunchedEffect(state) {
                        onScaffoldEvent(
                            ScaffoldEvent.ShowSnackbar(
                                message = "Imported model: ${state.model.name}"
                            )
                        )
                        managementViewModel.resetManagementState()
                    }
                }
            }

            is Download.Querying -> {
                ImportFromHuggingFaceDialog(
                    onCancel = { managementViewModel.resetManagementState() }
                )
            }

            is Download.Ready -> {
                ImportFromHuggingFaceDialog(
                    models = state.models,
                    onConfirm = { managementViewModel.downloadHuggingFaceModelConfirmed(it) },
                    onCancel = { managementViewModel.resetManagementState() }
                )
            }

            is Download.Dispatched -> {
                DownloadHuggingFaceDispatchedDialog(
                    state.downloadInfo.modelId,
                    onConfirm = { managementViewModel.resetManagementState() }
                )
            }

            is Download.Completed -> {
                ImportFromLocalFileDialog(
                    fileName = state.fileName,
                    fileSize = state.fileSize,
                    isImporting = false,
                    progress = 0.0f,
                    onConfirm = {
                        managementViewModel.importLocalModelFileConfirmed(
                            state.uri, state.fileName, state.fileSize
                        )
                    },
                    onCancel = { managementViewModel.resetManagementState() }
                )
            }

            is Download.Error -> {
                ErrorDialog(
                    context = context,
                    title = "Download Failed",
                    message = state.message,
                    onDismiss = { managementViewModel.resetManagementState() }
                )
            }

            is Deletion.Confirming -> {
                BatchDeleteConfirmationDialog(
                    count = state.models.size,
                    onConfirm = { managementViewModel.deleteModels(state.models) },
                    onDismiss = { managementViewModel.resetManagementState() },
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
                    context = context,
                    title = "Deletion Failed",
                    message = state.message,
                    onDismiss = { managementViewModel.resetManagementState() }
                )
            }

            is Deletion.Success -> {
                LaunchedEffect(state) {
                    modelsViewModel.toggleMode(ModelScreenUiMode.MANAGING)

                    val count = state.models.size
                    onScaffoldEvent(
                        ScaffoldEvent.ShowSnackbar(
                            message = "Deleted $count ${if (count > 1) "models" else "model"}.",
                            withDismissAction = true,
                            duration = SnackbarDuration.Long,
                        )
                    )
                }
            }

            is ModelManagementState.Idle -> { /* Idle state, nothing to show */ }
        }
    }
}

@Composable
private fun ImportFromLocalFileDialog(
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
private fun ImportFromHuggingFaceDialog(
    models: List<HuggingFaceModel>? = null,
    onConfirm: ((HuggingFaceModel) -> Unit)? = null,
    onCancel: () -> Unit,
) {
    val dateFormatter = remember { SimpleDateFormat("MMM, yyyy", Locale.getDefault()) }

    var selectedModel by remember { mutableStateOf<HuggingFaceModel?>(null) }

    AlertDialog(
        onDismissRequest = {},
        properties = DialogProperties(
            dismissOnBackPress = true,
            dismissOnClickOutside = true
        ),
        title = {
            Text(models?.let { "Fetched ${it.size} models" } ?: "Fetching models")
        },
        text = {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Text(
                    modifier = Modifier.fillMaxWidth(),
                    text = models?.let {
                        "The Hugging Face models shown here have been pre-filtered to be moderately sized and correctly quantized.\n\n" +
                            "Please use responsibly. We do not endorse or take responsibility for misuse or harmful use of these models.\n\n" +
                            "Select a model to download:"
                    } ?: "Searching on HuggingFace for open-source models free for downloading...",
                    style = MaterialTheme.typography.bodyLarge,
                    textAlign = TextAlign.Start,
                )

                if (models == null) {
                    Spacer(modifier = Modifier.height(24.dp))

                    CircularProgressIndicator(
                        modifier = Modifier.size(64.dp),
                        strokeWidth = 6.dp
                    )
                } else {
                    Spacer(modifier = Modifier.height(16.dp))

                    LazyColumn(
                        modifier = Modifier.fillMaxWidth(),
                        contentPadding = PaddingValues(vertical = 8.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        items(models) { model ->
                            HuggingFaceModelListItem(
                                model = model,
                                isSelected = model._id == selectedModel?._id,
                                dateFormatter = dateFormatter,
                                onToggleSelect = { selected ->
                                    selectedModel = if (selected) model else null
                                }
                            )
                        }
                    }
                }
            }
        },
        confirmButton = {
            onConfirm?.let { onSelect ->
                TextButton(
                    onClick = { selectedModel?.let { onSelect.invoke(it) } },
                    enabled = selectedModel != null
                ) {
                    Text("Download")
                }
            }
        },
        dismissButton = {
            TextButton(
                onClick = onCancel
            ) {
                Text("Cancel")
            }
        }
    )
}

@Composable
fun HuggingFaceModelListItem(
    model: HuggingFaceModel,
    isSelected: Boolean,
    dateFormatter: SimpleDateFormat,
    onToggleSelect: (Boolean) -> Unit,
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = when (isSelected) {
            true -> CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer
            )
            false -> CardDefaults.cardColors()
        },
        onClick = { onToggleSelect(!isSelected) }
    ) {
        Column(modifier = Modifier.fillMaxWidth().padding(8.dp)) {
            Text(
                modifier = Modifier.fillMaxWidth().basicMarquee(),
                text = model.modelId.substringAfterLast("/"),
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
            )

            Spacer(modifier = Modifier.size(8.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(0.9f)) {
                    Row(
                        modifier = Modifier.padding(bottom = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(
                            imageVector = Icons.Default.Attribution,
                            contentDescription = "Author",
                            modifier = Modifier.size(16.dp),
                            tint = MaterialTheme.colorScheme.onSurfaceVariant
                        )

                        Text(
                            modifier = Modifier.padding(start = 4.dp),
                            text = model.author,
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(modifier = Modifier.weight(5f).padding(end = 8.dp)) {
                            Icon(
                                imageVector = Icons.Default.Today,
                                contentDescription = "Author",
                                modifier = Modifier.size(16.dp),
                                tint = MaterialTheme.colorScheme.onSurfaceVariant
                            )

                            Text(
                                modifier = Modifier.padding(start = 4.dp),
                                text = dateFormatter.format(model.lastModified),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        Row(modifier = Modifier.weight(3f).padding(end = 8.dp)) {
                            Icon(
                                imageVector = Icons.Default.Favorite,
                                contentDescription = "Favorite count",
                                modifier = Modifier.size(16.dp),
                                tint = MaterialTheme.colorScheme.onSurfaceVariant
                            )

                            Text(
                                modifier = Modifier.padding(start = 4.dp),
                                text = formatContextLength(model.likes),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }

                        Row(modifier = Modifier.weight(4f).padding(end = 4.dp)) {
                            Icon(
                                imageVector = Icons.Default.Download,
                                contentDescription = "Download count",
                                modifier = Modifier.size(16.dp),
                                tint = MaterialTheme.colorScheme.onSurfaceVariant
                            )

                            Text(
                                modifier = Modifier.padding(start = 4.dp),
                                text = formatContextLength(model.downloads),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }

                Checkbox(
                    modifier = Modifier.size(32.dp).alpha(if (isSelected) 1f else 0f),
                    checked = isSelected,
                    onCheckedChange = null, // handled by parent selectable
                )
            }
        }
    }
}

@Composable
private fun DownloadHuggingFaceDispatchedDialog(
    modelId: String,
    onConfirm: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = {},
        properties = DialogProperties(
            dismissOnBackPress = false,
            dismissOnClickOutside = false
        ),
        title = {},
        text = {
            InfoView(
                title = "Download has started",
                icon = Icons.Default.Download,
                message = "Your Android system download manager has started downloading the model: $modelId.\n\n"
                    + "You can track its progress in your notification drawer.\n"
                    + "Feel free to stay on this screen, or come back to import it after complete.",
            )
        },
        confirmButton = {
            Button(onClick = onConfirm) { Text("Okay") }
        },
    )
}

@Composable
private fun FirstModelImportSuccessDialog(
    onConfirm: () -> Unit,
) {
    AlertDialog(
        // Prevent dismissal via back button during deletion
        properties = DialogProperties(
            dismissOnBackPress = false,
            dismissOnClickOutside = false
        ),
        onDismissRequest = {},
        text = {
            InfoView(
                title = "Congratulations",
                icon = Icons.Default.Celebration,
                message = "You have just installed your first Large Language Model!",
                action = InfoAction(
                    label = "Check it out",
                    icon = Icons.AutoMirrored.Default.ArrowForward,
                    onAction = onConfirm
                )
            )
        },
        confirmButton = {}
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
        containerColor = MaterialTheme.colorScheme.errorContainer,
        titleContentColor = MaterialTheme.colorScheme.onErrorContainer,
        textContentColor = MaterialTheme.colorScheme.onErrorContainer,
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
    context: Context,
    title: String,
    message: String,
    learnMoreUrl: String? = null,
    onDismiss: () -> Unit
) {
    val action = learnMoreUrl?.let { url ->
         InfoAction(
            label = "Learn more",
            icon = Icons.AutoMirrored.Outlined.ContactSupport,
            onAction = {
                val intent = Intent(Intent.ACTION_VIEW, url.toUri())
                context.startActivity(intent)
            }
        )
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        text = {
            InfoView(
                modifier = Modifier.fillMaxWidth(),
                title = title,
                icon = Icons.Default.Error,
                message = message,
                action = action
            )
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("OK")
            }
        }
    )
}
