package com.example.llama.ui.screens

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import com.example.llama.data.model.ModelInfo
import com.example.llama.ui.components.InfoView
import com.example.llama.ui.scaffold.ScaffoldEvent
import com.example.llama.util.formatFileByteSize
import com.example.llama.viewmodel.ModelScreenUiMode
import com.example.llama.viewmodel.ModelsViewModel
import com.example.llama.viewmodel.PreselectedModelToRun.RamWarning

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelsScreen(
    onManageModelsClicked: () -> Unit,
    onConfirmSelection: (ModelInfo, RamWarning) -> Unit,
    onScaffoldEvent: (ScaffoldEvent) -> Unit,
    viewModel: ModelsViewModel,
) {
    // Data
    val preselection by viewModel.preselectedModelToRun.collectAsState()

    // UI states
    val currentMode by viewModel.modelScreenUiMode.collectAsState()

    // Handle back button press
    BackHandler {
        when (currentMode) {
            ModelScreenUiMode.BROWSING -> {
                if (preselection != null) {
                    viewModel.resetPreselection()
                }
            }
            ModelScreenUiMode.SEARCHING -> {
                viewModel.toggleMode(ModelScreenUiMode.BROWSING)
            }
            ModelScreenUiMode.MANAGING -> {
                viewModel.toggleMode(ModelScreenUiMode.BROWSING)
            }
            ModelScreenUiMode.DELETING -> {
                viewModel.toggleAllSelectedModelsToDelete(false)
                viewModel.toggleMode(ModelScreenUiMode.MANAGING)
            }
        }
    }

    Box(
        modifier = Modifier.fillMaxSize()
    ) {
        when (currentMode) {
            ModelScreenUiMode.BROWSING ->
                ModelsBrowsingScreen(
                    onManageModelsClicked = { /* TODO-han.yin */ },
                    viewModel = viewModel
                )
            ModelScreenUiMode.SEARCHING ->
                ModelsSearchingScreen(viewModel = viewModel)
            ModelScreenUiMode.MANAGING, ModelScreenUiMode.DELETING ->
                ModelsManagementAndDeletingScreen(
                    isDeleting = currentMode == ModelScreenUiMode.DELETING,
                    onScaffoldEvent = onScaffoldEvent,
                    viewModel = viewModel
                )
        }

        // Show insufficient RAM warning
        preselection?.let {
            it.ramWarning?.let { warning ->
                if (warning.showing) {
                    RamErrorDialog(
                        warning,
                        onDismiss = { viewModel.dismissRamWarning() },
                        onConfirm = { onConfirmSelection(it.modelInfo, warning) }
                    )
                }
            }
        }
    }
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
