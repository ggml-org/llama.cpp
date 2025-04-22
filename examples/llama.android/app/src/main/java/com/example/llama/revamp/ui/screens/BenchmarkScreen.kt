package com.example.llama.revamp.ui.screens

import android.llama.cpp.InferenceEngine.State
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ProgressIndicatorDefaults
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
import androidx.compose.ui.text.style.TextAlign
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
    val unloadDialogState by viewModel.unloadModelState.collectAsState()

    val showModelCard by viewModel.showModelCard.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()

    val benchmarkResults by viewModel.benchmarkResults.collectAsState()

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

    Box(
        modifier = Modifier.fillMaxSize()
    ) {
        // Benchmark results
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(8.dp),
            verticalArrangement = Arrangement.Bottom,
        ) {
            items(items = benchmarkResults) { result ->
                Card(
                    modifier = Modifier.fillMaxWidth().padding(8.dp)
                ) {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(
                                color = MaterialTheme.colorScheme.surfaceVariant,
                                shape = RoundedCornerShape(8.dp)
                            )
                            .padding(16.dp)
                    ) {
                        Text(
                            text = result.text,
                            style = MonospacedTextStyle,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )

                        Spacer(modifier = Modifier.height(4.dp))

                        ModelCardContentField("Time spent: ", formatMilliSeconds(result.duration))
                    }
                }
            }
        }

        // Loading indicator
        if (engineState is State.Benchmarking) {
            Card(
                modifier = Modifier.align(Alignment.Center),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                ),
                shape = MaterialTheme.shapes.extraLarge
            ) {
                Column(
                    modifier = Modifier.padding(horizontal = 32.dp, vertical = 48.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(64.dp),
                        strokeWidth = ProgressIndicatorDefaults.CircularStrokeWidth * 1.5f
                    )

                    Spacer(modifier = Modifier.height(16.dp))

                    Text(
                        text = "Running benchmark...",
                        style = MaterialTheme.typography.headlineSmall
                    )

                    Spacer(modifier = Modifier.height(8.dp))

                    Text(
                        text = "This usually takes a few minutes",
                        style = MaterialTheme.typography.bodyLarge,
                        textAlign = TextAlign.Center,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }

        // Selected model card and loading metrics
        if (showModelCard) {
            selectedModel?.let { model ->
                Box(
                    modifier = Modifier.padding(start = 16.dp, top = 16.dp, end = 16.dp)
                ) {
                    ModelCardWithLoadingMetrics(
                        model = model,
                        loadingMetrics = loadingMetrics,
                        isExpanded = isModelCardExpanded,
                        onExpanded = { isModelCardExpanded = !isModelCardExpanded },
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
