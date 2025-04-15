package com.example.llama.revamp.ui.screens

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.ui.components.ModelCard
import com.example.llama.revamp.ui.components.ModelCardActions
import com.example.llama.revamp.ui.components.PerformanceAppScaffold

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelSelectionScreen(
    onModelSelected: (ModelInfo) -> Unit,
    onManageModelsClicked: () -> Unit,
    onMenuClicked: () -> Unit,
) {
    // For demo purposes, we'll use sample models
    val models = remember { ModelInfo.getSampleModels() }

    PerformanceAppScaffold(
        title = "Models",
        onMenuOpen = onMenuClicked,
        showTemperature = false
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 16.dp)
        ) {
            TextButton(
                onClick = onManageModelsClicked,
                modifier = Modifier
                    .align(Alignment.End)
                    .padding(top = 8.dp, bottom = 8.dp)
            ) {
                Text("Manage Models")
            }

            LazyColumn {
                items(models) { model ->
                    ModelCard(
                        model = model,
                        onClick = { onModelSelected(model) },
                        modifier = Modifier.padding(vertical = 4.dp),
                        isSelected = null, // Not in selection mode
                        actionButton = {
                            ModelCardActions.PlayButton(onClick = { onModelSelected(model) })
                        }
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                }
            }
        }
    }
}
