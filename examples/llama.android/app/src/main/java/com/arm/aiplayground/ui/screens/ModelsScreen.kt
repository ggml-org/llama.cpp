package com.arm.aiplayground.ui.screens

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.arm.aiplayground.data.model.ModelInfo
import com.arm.aiplayground.ui.components.InfoView
import com.arm.aiplayground.ui.scaffold.ScaffoldEvent
import com.arm.aiplayground.util.formatFileByteSize
import com.arm.aiplayground.viewmodel.ModelScreenUiMode
import com.arm.aiplayground.viewmodel.ModelsManagementViewModel
import com.arm.aiplayground.viewmodel.ModelsViewModel
import com.arm.aiplayground.viewmodel.PreselectedModelToRun.RamWarning

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelsScreen(
    showModelImportTooltip: Boolean,
    onFirstModelImportSuccess: (ModelInfo) -> Unit,
    showChatTooltip: Boolean,
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
                    showChatTooltip = showChatTooltip,
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
                    showModelImportTooltip = showModelImportTooltip,
                    onFirstModelImportSuccess = onFirstModelImportSuccess,
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
        icon = {
            Icon(
                imageVector = Icons.Default.Warning,
                contentDescription = "Warning icon",
                modifier = Modifier.size(64.dp),
                tint = MaterialTheme.colorScheme.error
            )
        },
        title = {
            Text(
                modifier = Modifier.padding(top = 16.dp),
                text = "Insufficient RAM",
                style = MaterialTheme.typography.headlineSmall,
                textAlign = TextAlign.Center,
                fontWeight = FontWeight.SemiBold
            )
        },
        text = {
            Text(
               "You are trying to run a $requiredRam size model, " +
                    "but currently there's only $availableRam memory available!",
            )
       },
        containerColor = MaterialTheme.colorScheme.errorContainer,
        titleContentColor = MaterialTheme.colorScheme.onErrorContainer,
        textContentColor = MaterialTheme.colorScheme.onErrorContainer,
        dismissButton = {
            TextButton(
                onClick = onConfirm,
                colors = ButtonDefaults.textButtonColors(
                    contentColor = MaterialTheme.colorScheme.error
                )
            ) {
                Text("Proceed")
            }
        },
        onDismissRequest = onDismiss,
        confirmButton = {
            FilledTonalButton(onClick = onDismiss) {
                Text("Cancel")
            }
        }
    )
}
