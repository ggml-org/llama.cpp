package com.example.llama.ui.screens

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ProgressIndicatorDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.llama.data.model.ModelInfo
import com.example.llama.ui.components.InfoView
import com.example.llama.ui.scaffold.ScaffoldEvent
import com.example.llama.util.formatFileByteSize
import com.example.llama.viewmodel.ModelScreenUiMode
import com.example.llama.viewmodel.ModelsManagementViewModel
import com.example.llama.viewmodel.ModelsViewModel
import com.example.llama.viewmodel.PreselectedModelToRun.RamWarning

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelsScreen(
    onConfirmSelection: (ModelInfo, RamWarning) -> Unit,
    onScaffoldEvent: (ScaffoldEvent) -> Unit,
    modelsViewModel: ModelsViewModel,
    managementViewModel: ModelsManagementViewModel,
) {
    // Data
    val filteredModels by modelsViewModel.filteredModels.collectAsState()
    val preselection by modelsViewModel.preselectedModelToRun.collectAsState()

    // UI states: Filter
    val activeFilters by modelsViewModel.activeFilters.collectAsState()
    val activeFiltersCount by remember(activeFilters) {
        derivedStateOf { activeFilters.count { it.value }  }
    }

    // UI states
    val currentMode by modelsViewModel.modelScreenUiMode.collectAsState()

    // Handle back button press
    BackHandler {
        when (currentMode) {
            ModelScreenUiMode.BROWSING -> {
                if (preselection != null) {
                    modelsViewModel.resetPreselection()
                }
            }
            ModelScreenUiMode.SEARCHING -> {
                modelsViewModel.toggleMode(ModelScreenUiMode.BROWSING)
            }
            ModelScreenUiMode.MANAGING -> {
                modelsViewModel.toggleMode(ModelScreenUiMode.BROWSING)
            }
            ModelScreenUiMode.DELETING -> {
                managementViewModel.clearSelectedModelsToDelete()
                modelsViewModel.toggleMode(ModelScreenUiMode.MANAGING)
            }
        }
    }

    Box(
        modifier = Modifier.fillMaxSize()
    ) {
        when (currentMode) {
            ModelScreenUiMode.BROWSING ->
                ModelsBrowsingScreen(
                    filteredModels = filteredModels,
                    preselection = preselection,
                    onManageModelsClicked = {
                        managementViewModel.toggleImportMenu(true)
                        modelsViewModel.toggleMode(ModelScreenUiMode.MANAGING)
                    },
                    activeFiltersCount = activeFiltersCount,
                    viewModel = modelsViewModel,
                )
            ModelScreenUiMode.SEARCHING ->
                ModelsSearchingScreen(
                    preselection = preselection,
                    viewModel = modelsViewModel
                )
            ModelScreenUiMode.MANAGING, ModelScreenUiMode.DELETING ->
                ModelsManagementAndDeletingScreen(
                    filteredModels = filteredModels,
                    isDeleting = currentMode == ModelScreenUiMode.DELETING,
                    onScaffoldEvent = onScaffoldEvent,
                    activeFiltersCount = activeFiltersCount,
                    modelsViewModel = modelsViewModel,
                    managementViewModel = managementViewModel,
                )
        }

        // Show insufficient RAM warning
        preselection?.let {
            it.ramWarning?.let { warning ->
                if (warning.showing) {
                    RamErrorDialog(
                        warning,
                        onDismiss = { modelsViewModel.dismissRamWarning() },
                        onConfirm = { onConfirmSelection(it.modelInfo, warning) }
                    )
                }
            }
        }
    }
}

@Composable
fun ModelsLoadingInProgressView() {
    InfoView(
        modifier = Modifier.fillMaxSize(),
        title = "Loading...",
        icon = {
            CircularProgressIndicator(
                modifier = Modifier.size(64.dp),
                strokeWidth = ProgressIndicatorDefaults.CircularStrokeWidth * 1.5f
            )
        },
        message = "Searching for installed models on your device...",
    )
}

@Composable
private fun RamErrorDialog(
    ramError: RamWarning,
    onDismiss: () -> Unit,
    onConfirm: () -> Unit,
) {
    val requiredRam = formatFileByteSize(ramError.requiredRam)
    val availableRam = formatFileByteSize(ramError.availableRam)

    AlertDialog(
        text = {
            InfoView(
                modifier = Modifier.fillMaxWidth(),
                title = "Insufficient RAM",
                icon = Icons.Default.Warning,
                message = "You are trying to run a $requiredRam size model, " +
                    "but currently there's only $availableRam memory available!",
            )
       },
        containerColor = MaterialTheme.colorScheme.errorContainer,
        titleContentColor = MaterialTheme.colorScheme.onErrorContainer,
        textContentColor = MaterialTheme.colorScheme.onErrorContainer,
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        },
        onDismissRequest = onDismiss,
        confirmButton = {
            TextButton(onClick = onConfirm) {
                Text(
                    text = "Proceed",
                    color = MaterialTheme.colorScheme.error
                )
            }
        }
    )
}
