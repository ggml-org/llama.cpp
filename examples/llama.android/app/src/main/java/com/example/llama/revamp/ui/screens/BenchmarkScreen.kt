package com.example.llama.revamp.ui.screens

import android.llama.cpp.InferenceEngine.State
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.llama.revamp.data.model.ModelInfo
import com.example.llama.revamp.engine.ModelLoadingMetrics
import com.example.llama.revamp.ui.components.ModelCardContentArchitectureRow
import com.example.llama.revamp.ui.components.ModelCardContentContextRow
import com.example.llama.revamp.ui.components.ModelCardContentField
import com.example.llama.revamp.ui.components.ModelCardCoreExpandable
import com.example.llama.revamp.ui.components.ModelUnloadDialogHandler
import com.example.llama.revamp.ui.theme.MonospacedTextStyle
import com.example.llama.revamp.util.formatMilliSeconds
import com.example.llama.revamp.viewmodel.BenchmarkViewModel

@Composable
fun BenchmarkScreen(
    loadingMetrics: ModelLoadingMetrics,
    onNavigateBack: () -> Unit,
    viewModel: BenchmarkViewModel
) {
    // View model states
    val engineState by viewModel.engineState.collectAsState()
    val benchmarkResults by viewModel.benchmarkResults.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val unloadDialogState by viewModel.unloadModelState.collectAsState()

    // UI states
    var isModelCardExpanded by remember { mutableStateOf(false) }

    // Run benchmark when entering the screen
    LaunchedEffect(selectedModel) {
        viewModel.runBenchmark()
    }

    // Handle back button press
    BackHandler {
        viewModel.onBackPressed(onNavigateBack)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState())
    ) {
        // Selected model card
        selectedModel?.let { model ->
            ModelCardWithLoadingMetrics(
                model = model,
                loadingMetrics = loadingMetrics,
                isExpanded = isModelCardExpanded,
                onExpanded = { isModelCardExpanded = !isModelCardExpanded },
            )
        }

        // Benchmark results or loading indicator
        when {
            engineState is State.Benchmarking -> {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(200.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator()
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(
                            text = "Running benchmark...",
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
            }

            benchmarkResults != null -> {
                Card(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(
                                color = MaterialTheme.colorScheme.surfaceVariant,
                                shape = RoundedCornerShape(8.dp)
                            )
                            .padding(16.dp)
                    ) {
                        Text(
                            text = benchmarkResults ?: "",
                            style = MonospacedTextStyle,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }

            else -> {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(200.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "Benchmark results will appear here",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }

    // Unload confirmation dialog
    ModelUnloadDialogHandler(
        unloadModelState = unloadDialogState,
        onUnloadConfirmed = { viewModel.onUnloadConfirmed(onNavigateBack) },
        onUnloadDismissed = { viewModel.onUnloadDismissed() },
        onNavigateBack = onNavigateBack,
    )
}

@Composable
private fun ModelCardWithLoadingMetrics(
    model: ModelInfo,
    loadingMetrics: ModelLoadingMetrics,
    isExpanded: Boolean = false,
    onExpanded: ((Boolean) -> Unit)? = null,
) = ModelCardCoreExpandable(model, isExpanded, onExpanded) {
    Spacer(modifier = Modifier.height(8.dp))

    // Row 2: Context length, size label
    ModelCardContentContextRow(model)

    Spacer(modifier = Modifier.height(8.dp))

    // Row 3: Architecture, quantization, formatted size
    ModelCardContentArchitectureRow(model)

    Spacer(modifier = Modifier.height(8.dp))

    // Row 4: Model loading time
    ModelCardContentField("Loading time", formatMilliSeconds(loadingMetrics.modelLoadingTimeMs))
}
