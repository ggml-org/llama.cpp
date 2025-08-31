package com.example.llama.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.runtime.Composable
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
        EmptyModelsView(activeFiltersCount, onManageModelsClicked)
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

@Composable
private fun EmptyModelsView(
    activeFiltersCount: Int,
    onManageModelsClicked: () -> Unit
) {
    val message = when (activeFiltersCount) {
        0 -> "Import some models to get started!"
        1 -> "No models match the selected filter"
        else -> "No models match the selected filters"
    }
    InfoView(
        modifier = Modifier.fillMaxSize(),
        title = "No Models Available",
        icon = Icons.Default.FolderOpen,
        message = message,
        action = InfoAction(
            label = "Add Models",
            icon = Icons.Default.Add,
            onAction = onManageModelsClicked
        )
    )
}
