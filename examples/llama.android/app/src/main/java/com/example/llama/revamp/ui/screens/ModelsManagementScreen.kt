package com.example.llama.revamp.ui.screens

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
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
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.ui.components.StorageAppScaffold
import com.example.llama.revamp.viewmodel.ModelSortOrder
import com.example.llama.revamp.viewmodel.ModelsManagementViewModel
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import com.example.llama.R

/**
 * Screen for managing LLM models (view, download, delete)
 */
@Composable
fun ModelsManagementScreen(
    onBackPressed: () -> Unit,
    viewModel: ModelsManagementViewModel = hiltViewModel()
) {
    val storageMetrics by viewModel.storageMetrics.collectAsState()
    val sortedModels by viewModel.sortedModels.collectAsState()
    val sortOrder by viewModel.sortOrder.collectAsState()

    // UI: menu states
    var showSortMenu by remember { mutableStateOf(false) }
    var showAddModelMenu by remember { mutableStateOf(false) }

    // UI: multi-selection states
    var isMultiSelectionMode by remember { mutableStateOf(false) }
    val selectedModels = remember { mutableStateMapOf<String, ModelInfo>() }
    val exitSelectionMode = {
        isMultiSelectionMode = false
        selectedModels.clear()
    }

    StorageAppScaffold(
        title = "Models Management",
        storageUsed = storageMetrics?.usedGB ?: 0f,
        storageTotal = storageMetrics?.totalGB ?: 0f,
        onNavigateBack = onBackPressed,
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
                                // Delete selected
                                if (selectedModels.isNotEmpty()) {
                                    viewModel.deleteModels(selectedModels)
                                    exitSelectionMode()
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
                                    MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.38f)
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

                        IconButton(onClick = { /* Filter action - stub for now */ }) {
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
                                showAddModelMenu = true
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
                        expanded = showAddModelMenu,
                        onDismissRequest = { showAddModelMenu = false }
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
                                // TODO-han.yin: uncomment once file picker done
                                // viewModel.importLocalModel()
                                showAddModelMenu = false
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
                                showAddModelMenu = false
                            }
                        )
                    }
                }
            )
        },
    ) { paddingValues ->
        // Main content
        ModelList(
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
            onModelDeleteClick = { modelId ->
                viewModel.deleteModel(modelId)
            },
            modifier = Modifier.padding(paddingValues)
        )
    }
}


@Composable
private fun ModelList(
    models: List<ModelInfo>,
    isMultiSelectionMode: Boolean,
    selectedModels: Map<String, ModelInfo>,
    onModelClick: (String) -> Unit,
    onModelInfoClick: (String) -> Unit,
    onModelDeleteClick: (String) -> Unit,
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
            ModelItem(
                model = model,
                isMultiSelectionMode = isMultiSelectionMode,
                isSelected = selectedModels.contains(model.id),
                onClick = { onModelClick(model.id) },
                onInfoClick = { onModelInfoClick(model.id) },
                onDeleteClick = { onModelDeleteClick(model.id) }
            )
            Spacer(modifier = Modifier.height(8.dp))
        }
    }
}

@Composable
private fun ModelItem(
    model: ModelInfo,
    isMultiSelectionMode: Boolean,
    isSelected: Boolean,
    onClick: () -> Unit,
    onInfoClick: () -> Unit,
    onDeleteClick: () -> Unit
) {
    // Model item implementation with selection support
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

                IconButton(onClick = onDeleteClick) {
                    Icon(
                        imageVector = Icons.Default.Delete,
                        contentDescription = "Delete model",
                        tint = MaterialTheme.colorScheme.error
                    )
                }
            }
        }
    }
}
