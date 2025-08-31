package com.example.llama.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.llama.data.model.ModelInfo
import com.example.llama.ui.components.InfoAction
import com.example.llama.ui.components.InfoView
import com.example.llama.ui.components.ModelCardFullExpandable
import com.example.llama.viewmodel.ModelsViewModel
import com.example.llama.viewmodel.PreselectedModelToRun

@Composable
fun ModelsBrowsingScreen(
    filteredModels: List<ModelInfo>?,
    activeFiltersCount: Int,
    preselection: PreselectedModelToRun?,
    onManageModelsClicked: () -> Unit,
    viewModel: ModelsViewModel,
) {
    if (filteredModels == null) {
        ModelsLoadingInProgressView()
    } else if (filteredModels.isEmpty()) {
        // Empty model prompt
        val title = when (activeFiltersCount) {
            0 -> "No models installed yet"
            1 -> "No models match your filter"
            else -> "No models match your filters"
        }
        val message = when (activeFiltersCount) {
            0 -> "Tap the button below to install your first Large Language Model!"
            1 -> "Try removing your filter to see more results"
            else -> "Try removing some filters to see more results"
        }
        Box(modifier = Modifier.fillMaxSize()) {
            InfoView(
                modifier = Modifier.fillMaxSize(0.9f).align(Alignment.Center),
                title = title,
                icon = Icons.Default.FolderOpen,
                message = message,
                action = InfoAction(
                    label = "Get Started",
                    icon = Icons.AutoMirrored.Default.ArrowForward,
                    onAction = onManageModelsClicked
                )
            )
        }
    } else {
        // Model cards
        LazyColumn(
            Modifier.fillMaxSize(), // .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
            contentPadding = PaddingValues(vertical = 12.dp, horizontal = 16.dp),
        ) {
            items(items = filteredModels, key = { it.id }) { model ->
                ModelCardFullExpandable(
                    model = model,
                    isSelected = if (model == preselection?.modelInfo) true else null,
                    onSelected = { selected ->
                        if (!selected) viewModel.resetPreselection()
                    },
                    isExpanded = model == preselection?.modelInfo,
                    onExpanded = { expanded ->
                        viewModel.preselectModel(model, expanded)
                    }
                )
            }
        }
    }
}
