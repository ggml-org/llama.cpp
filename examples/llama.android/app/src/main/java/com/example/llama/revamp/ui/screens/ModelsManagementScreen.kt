package com.example.llama.revamp.ui.screens

import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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
import androidx.compose.material.icons.automirrored.filled.Sort
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.ClearAll
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.FilterAlt
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.SelectAll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.DialogProperties
import androidx.hilt.navigation.compose.hiltViewModel
import com.example.llama.R
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.ui.components.StorageAppScaffold
import com.example.llama.revamp.viewmodel.ModelManagementState.Deletion
import com.example.llama.revamp.viewmodel.ModelManagementState.Importation
import com.example.llama.revamp.viewmodel.ModelSortOrder
import com.example.llama.revamp.viewmodel.ModelsManagementViewModel
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Screen for managing LLM models (view, download, delete)
 */
@Composable
fun ModelsManagementScreen(
    onBackPressed: () -> Unit,
    viewModel: ModelsManagementViewModel = hiltViewModel()
) {
    val coroutineScope = rememberCoroutineScope()
    val snackbarHostState = remember { SnackbarHostState() }

    // ViewModel states
    val storageMetrics by viewModel.storageMetrics.collectAsState()
    val sortOrder by viewModel.sortOrder.collectAsState()
    val sortedModels by viewModel.sortedModels.collectAsState()
    val managementState by viewModel.managementState.collectAsState()

    // UI state: sorting
    var showSortMenu by remember { mutableStateOf(false) }

    // UI state: importing
    var showImportModelMenu by remember { mutableStateOf(false) }
    val fileLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri -> uri?.let { viewModel.importLocalModel(it) } }

    // UI state: multi-selecting
    var isMultiSelectionMode by remember { mutableStateOf(false) }
    val selectedModels = remember { mutableStateMapOf<String, ModelInfo>() }
    val exitSelectionMode = {
        isMultiSelectionMode = false
        selectedModels.clear()
    }

    BackHandler(
        enabled = managementState is Importation.Importing
            || managementState is Deletion.Deleting
    ) {
        /* Ignore back press while processing model management requests */
    }

    StorageAppScaffold(
        title = "Models Management",
        storageUsed = storageMetrics?.usedGB ?: 0f,
        storageTotal = storageMetrics?.totalGB ?: 0f,
        onNavigateBack = onBackPressed,
        snackbarHostState = snackbarHostState,
        bottomBar = {
            BottomAppBar(
                actions = {
                    if (isMultiSelectionMode) {
                        // Multi-selection mode actions
                        IconButton(onClick = {
                            // Select all
                            selectedModels.putAll(sortedModels.map { it.id to it })
                        }) {
                            Icon(
                                imageVector = Icons.Default.SelectAll,
                                contentDescription = "Select all"
                            )
                        }

                        IconButton(onClick = {
                            // Deselect all
                            selectedModels.clear()
                        }) {
                            Icon(
                                imageVector = Icons.Default.ClearAll,
                                contentDescription = "Deselect all"
                            )
                        }

                        IconButton(
                            onClick = {
                                if (selectedModels.isNotEmpty()) {
                                    viewModel.batchDeletionClicked(selectedModels.toMap())
                                }
                            },
                            enabled = selectedModels.isNotEmpty()
                        ) {
                            Icon(
                                imageVector = Icons.Default.Delete,
                                contentDescription = "Delete selected",
                                tint = if (selectedModels.isNotEmpty())
                                    MaterialTheme.colorScheme.error
                                else
                                    MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.3f)
                            )
                        }
                    } else {
                        // Default mode actions
                        IconButton(onClick = { showSortMenu = true }) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.Sort,
                                contentDescription = "Sort models"
                            )
                        }

                        // Sort dropdown menu
                        DropdownMenu(
                            expanded = showSortMenu,
                            onDismissRequest = { showSortMenu = false }
                        ) {
                            DropdownMenuItem(
                                text = { Text("Name (A-Z)") },
                                trailingIcon = {
                                    if (sortOrder == ModelSortOrder.NAME_ASC)
                                        Icon(
                                            imageVector = Icons.Default.Check,
                                            contentDescription = "Sort by name in ascending order, selected"
                                        )
                                },
                                onClick = {
                                    viewModel.setSortOrder(ModelSortOrder.NAME_ASC)
                                    showSortMenu = false
                                }
                            )
                            DropdownMenuItem(
                                text = { Text("Name (Z-A)") },
                                trailingIcon = {
                                    if (sortOrder == ModelSortOrder.NAME_DESC)
                                        Icon(
                                            imageVector = Icons.Default.Check,
                                            contentDescription = "Sort by name in descending order, selected"
                                        )
                                },
                                onClick = {
                                    viewModel.setSortOrder(ModelSortOrder.NAME_DESC)
                                    showSortMenu = false
                                }
                            )
                            DropdownMenuItem(
                                text = { Text("Size (Smallest first)") },
                                trailingIcon = {
                                    if (sortOrder == ModelSortOrder.SIZE_ASC)
                                        Icon(
                                            imageVector = Icons.Default.Check,
                                            contentDescription = "Sort by size in ascending order, selected"
                                        )
                                },
                                onClick = {
                                    viewModel.setSortOrder(ModelSortOrder.SIZE_ASC)
                                    showSortMenu = false
                                }
                            )
                            DropdownMenuItem(
                                text = { Text("Size (Largest first)") },
                                trailingIcon = {
                                    if (sortOrder == ModelSortOrder.SIZE_DESC)
                                        Icon(
                                            imageVector = Icons.Default.Check,
                                            contentDescription = "Sort by size in descending order, selected"
                                        )
                                },
                                onClick = {
                                    viewModel.setSortOrder(ModelSortOrder.SIZE_DESC)
                                    showSortMenu = false
                                }
                            )
                            DropdownMenuItem(
                                text = { Text("Last used") },
                                trailingIcon = {
                                    if (sortOrder == ModelSortOrder.LAST_USED)
                                        Icon(
                                            imageVector = Icons.Default.Check,
                                            contentDescription = "Sort by last used, selected"
                                        )
                                },
                                onClick = {
                                    viewModel.setSortOrder(ModelSortOrder.LAST_USED)
                                    showSortMenu = false
                                }
                            )
                        }

                        IconButton(
                            onClick = {/* TODO-han.yin: implement filtering */ }
                        ) {
                            Icon(
                                imageVector = Icons.Default.FilterAlt,
                                contentDescription = "Filter models"
                            )
                        }

                        IconButton(onClick = {
                            isMultiSelectionMode = true
                        }) {
                            Icon(
                                imageVector = Icons.Default.DeleteSweep,
                                contentDescription = "Delete models"
                            )
                        }
                    }
                },
                floatingActionButton = {
                    FloatingActionButton(
                        onClick = {
                            if (isMultiSelectionMode) {
                                exitSelectionMode()
                            } else {
                                showImportModelMenu = true
                            }
                        },
                        containerColor = MaterialTheme.colorScheme.primaryContainer
                    ) {
                        Icon(
                            imageVector = if (isMultiSelectionMode) Icons.Default.Close else Icons.Default.Add,
                            contentDescription = if (isMultiSelectionMode) "Exit selection mode" else "Add model"
                        )
                    }

                    // Add model dropdown menu
                    DropdownMenu(
                        expanded = showImportModelMenu,
                        onDismissRequest = { showImportModelMenu = false }
                    ) {
                        DropdownMenuItem(
                            text = { Text("Import local model") },
                            leadingIcon = {
                                Icon(
                                    imageVector = Icons.Default.FolderOpen,
                                    contentDescription = "Import a local model on the device"
                                )
                            },
                            onClick = {
                                fileLauncher.launch(arrayOf("application/octet-stream", "*/*"))
                                showImportModelMenu = false
                            }
                        )
                        DropdownMenuItem(
                            text = { Text("Download from HuggingFace") },
                            leadingIcon = {
                                Icon(
                                    painter = painterResource(id = R.drawable.logo_huggingface),
                                    contentDescription = "Browse and download a model from HuggingFace",
                                    modifier = Modifier.size(24.dp),
                                    tint = Color.Unspecified,
                                )
                            },
                            onClick = {
                                viewModel.importFromHuggingFace()
                                showImportModelMenu = false
                            }
                        )
                    }
                }
            )
        },
    ) { paddingValues ->
        Box(modifier = Modifier.fillMaxSize()) {
            // Model cards
            ModelCardList(
                models = sortedModels,
                isMultiSelectionMode = isMultiSelectionMode,
                selectedModels = selectedModels,
                onModelClick = { modelId ->
                    if (isMultiSelectionMode) {
                        // Toggle selection
                        if (selectedModels.contains(modelId)) {
                            selectedModels.remove(modelId)
                        } else {
                            selectedModels.put(modelId, sortedModels.first { it.id == modelId } )
                        }
                    } else {
                        // View model details
                        viewModel.viewModelDetails(modelId)
                    }
                },
                onModelInfoClick = { modelId ->
                    viewModel.viewModelDetails(modelId)
                },
                modifier = Modifier.padding(paddingValues)
            )

            // Model import progress overlay
            when (val state = managementState) {
                is Importation.Importing -> {
                    ImportProgressOverlay(
                        progress = state.progress,
                        filename = state.filename,
                        onCancel = { /* Implement cancellation if needed */ }
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
                        coroutineScope.launch {
                            snackbarHostState.showSnackbar(
                                message = "Imported model: ${state.model.name}",
                                duration = SnackbarDuration.Short
                            )
                        }
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
                        exitSelectionMode()
                        coroutineScope.launch {
                            val count = state.models.size
                            snackbarHostState.showSnackbar(
                                message = "Deleted $count ${if (count > 1) "models" else "model"}.",
                                duration = SnackbarDuration.Long
                            )
                        }

                    }
                }
                else -> { /* Idle state, nothing to show */ }
            }
        }
    }
}


@Composable
private fun ModelCardList(
    models: List<ModelInfo>,
    isMultiSelectionMode: Boolean,
    selectedModels: Map<String, ModelInfo>,
    onModelClick: (String) -> Unit,
    onModelInfoClick: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    LazyColumn(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        items(
            items = models,
            key = { it.id }
        ) { model ->
            ModelCard(
                model = model,
                isMultiSelectionMode = isMultiSelectionMode,
                isSelected = selectedModels.contains(model.id),
                onClick = { onModelClick(model.id) },
                onInfoClick = { onModelInfoClick(model.id) },
            )
            Spacer(modifier = Modifier.height(8.dp))
        }
    }
}

@Composable
private fun ModelCard(
    model: ModelInfo,
    isMultiSelectionMode: Boolean,
    isSelected: Boolean,
    onClick: () -> Unit,
    onInfoClick: () -> Unit,
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        colors = if (isSelected && isMultiSelectionMode)
            CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
        else
            CardDefaults.cardColors()
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Show checkbox in selection mode
            if (isMultiSelectionMode) {
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

                Text(
                    text = "${model.parameters} • ${model.quantization} • ${model.formattedSize}",
                    style = MaterialTheme.typography.bodySmall
                )

                model.lastUsed?.let { lastUsed ->
                    val dateFormat = SimpleDateFormat("MMM d, yyyy", Locale.getDefault())
                    Text(
                        text = "Last used: ${dateFormat.format(Date(lastUsed))}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            // Only show action buttons in non-selection mode
            if (!isMultiSelectionMode) {
                IconButton(onClick = onInfoClick) {
                    Icon(
                        imageVector = Icons.Default.Info,
                        contentDescription = "Model details"
                    )
                }
            }
        }
    }
}

// TODO-han.yin: Rewrite into
@Composable
fun ImportProgressOverlay(
    progress: Float,
    filename: String,
    onCancel: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black.copy(alpha = 0.7f))
            .padding(32.dp),
        contentAlignment = Alignment.Center
    ) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
        ) {
            Column(
                modifier = Modifier.padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "Importing Model",
                    style = MaterialTheme.typography.headlineSmall
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = filename,
                    style = MaterialTheme.typography.bodyMedium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )

                Spacer(modifier = Modifier.height(24.dp))

                LinearProgressIndicator(
                    progress = { progress },
                    modifier = Modifier.fillMaxWidth()
                )

                Spacer(modifier = Modifier.height(8.dp))

                Text(
                    text = "${(progress * 100).toInt()}%",
                    style = MaterialTheme.typography.bodyLarge
                )

                Spacer(modifier = Modifier.height(16.dp))

                Text(
                    text = "This may take several minutes for large models",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )

                Spacer(modifier = Modifier.height(24.dp))

                Button(
                    onClick = onCancel,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer,
                        contentColor = MaterialTheme.colorScheme.onErrorContainer
                    )
                ) {
                    Text("Cancel")
                }
            }
        }
    }
}

@Composable
fun BatchDeleteConfirmationDialog(
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
fun ErrorDialog(
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
