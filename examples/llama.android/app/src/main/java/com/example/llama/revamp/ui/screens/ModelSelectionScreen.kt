package com.example.llama.revamp.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.ui.components.ModelCardExpandable
import com.example.llama.revamp.viewmodel.ModelSelectionViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelSelectionScreen(
    onModelConfirmed: (ModelInfo) -> Unit,
    onManageModelsClicked: () -> Unit,
    viewModel: ModelSelectionViewModel,
) {
    val models by viewModel.availableModels.collectAsState()
    val preselectedModel by viewModel.preselectedModel.collectAsState()

    val handleModelSelection = { model: ModelInfo ->
        viewModel.confirmSelectedModel(model)
        onModelConfirmed(model)
    }

    Box(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp)
        ) {
            if (models.isEmpty()) {
                EmptyModelsView(onManageModelsClicked)
            } else {
                LazyColumn(
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    items(items = models, key = { it.id }) { model ->
                        ModelCardExpandable(
                            model = model,
                            isSelected = if (model == preselectedModel) true else null,
                            onSelected = { selected ->
                                if (!selected) viewModel.resetSelection()
                            },
                            isExpanded = model == preselectedModel,
                            onExpanded = { expanded ->
                                viewModel.preselectModel(model, expanded)
                            }
                        )
                    }
                }
            }
        }

        // Show FAB if a model is selected
        AnimatedVisibility(
            modifier = Modifier.padding(16.dp).align(Alignment.BottomEnd),
            visible = preselectedModel != null,
            enter = scaleIn() + fadeIn(),
            exit = scaleOut() + fadeOut()
        ) {
            FloatingActionButton(
                onClick = { preselectedModel?.let { handleModelSelection(it) } },
                containerColor = MaterialTheme.colorScheme.primary
            ) {
                Icon(
                    imageVector = Icons.Default.PlayArrow,
                    contentDescription = "Start with selected model"
                )
            }
        }
    }
}

@Composable
private fun EmptyModelsView(onManageModelsClicked: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Icon(
            imageVector = Icons.Default.FolderOpen,
            contentDescription = null,
            modifier = Modifier.size(64.dp),
            tint = MaterialTheme.colorScheme.primary.copy(alpha = 0.6f)
        )

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "No Models Available",
            style = MaterialTheme.typography.headlineSmall
        )

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            text = "Add models to get started with local LLM inference",
            style = MaterialTheme.typography.bodyMedium,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(24.dp))

        Button(onClick = onManageModelsClicked) {
            Icon(
                imageVector = Icons.Default.Add,
                contentDescription = null,
                modifier = Modifier.size(18.dp)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text("Add Models")
        }
    }
}
