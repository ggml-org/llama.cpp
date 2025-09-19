package com.example.llama.ui.screens

import android.content.Intent
import android.widget.Toast
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Replay
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.ProgressIndicatorDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.net.toUri
import com.arm.aichat.InferenceEngine.State
import com.example.llama.data.model.ModelInfo
import com.example.llama.engine.ModelLoadingMetrics
import com.example.llama.ui.components.ModelCardContentArchitectureRow
import com.example.llama.ui.components.ModelCardContentContextRow
import com.example.llama.ui.components.ModelCardContentField
import com.example.llama.ui.components.ModelCardCoreExpandable
import com.example.llama.ui.components.ModelUnloadDialogHandler
import com.example.llama.ui.scaffold.ScaffoldEvent
import com.example.llama.util.TableData
import com.example.llama.util.formatMilliSeconds
import com.example.llama.util.parseMarkdownTable
import com.example.llama.viewmodel.BenchmarkResult
import com.example.llama.viewmodel.BenchmarkViewModel


@Composable
fun BenchmarkScreen(
    loadingMetrics: ModelLoadingMetrics,
    onScaffoldEvent: (ScaffoldEvent) -> Unit,
    onNavigateBack: () -> Unit,
    viewModel: BenchmarkViewModel
) {
    val context = LocalContext.current

    // View model states
    val engineState by viewModel.engineState.collectAsState()
    val unloadDialogState by viewModel.unloadModelState.collectAsState()

    val showModelCard by viewModel.showModelCard.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()

    val benchmarkResults by viewModel.benchmarkResults.collectAsState()

    // UI states
    var isModelCardExpanded by remember { mutableStateOf(true) }
    var isInitialBenchmarkRun by rememberSaveable { mutableStateOf(false) }

    // Run benchmark when entering the screen
    LaunchedEffect(selectedModel) {
        if (!isInitialBenchmarkRun) {
            isInitialBenchmarkRun = true
            viewModel.runBenchmark()
        }
    }

    // Handle back button press
    BackHandler {
        viewModel.onBackPressed(onNavigateBack)
    }

    val onInfo = {
        Toast.makeText(context, "Please refer to this post for more details on the benchmark methodology", Toast.LENGTH_SHORT).show()
        val intent = Intent(Intent.ACTION_VIEW, "https://blog.steelph0enix.dev/posts/llama-cpp-guide/#llama-bench".toUri())
        context.startActivity(intent)
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
            items(items = benchmarkResults) {
                BenchmarkResultCard(
                    result = it,
                    onRerun = { viewModel.rerunBenchmark(onScaffoldEvent) },
                    onInfo = onInfo,
                )
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
                Box(modifier = Modifier.padding(start = 16.dp, top = 16.dp, end = 16.dp)) {
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
        message = "Going back will unload the current model and clear all the benchmark results.",
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
) = ModelCardCoreExpandable(
    model = model,
    containerColor = MaterialTheme.colorScheme.tertiaryContainer,
    isExpanded = isExpanded,
    onExpanded = onExpanded
) {
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


@Composable
fun BenchmarkResultCard(
    result: BenchmarkResult,
    onRerun: () -> Unit,
    onInfo: () -> Unit,
) {
    val rawTable = parseMarkdownTable(result.text.trimIndent())
    val model = rawTable.getColumn("model").firstOrNull() ?: "Unknown"
    val parameters = rawTable.getColumn("params").firstOrNull() ?: "-"
    val size = rawTable.getColumn("size").firstOrNull() ?: "-"

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
            Row {
                Text(
                    text = "Model",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Normal,
                )

                Spacer(modifier = Modifier.width(16.dp))

                Text(
                    modifier = Modifier.weight(1f),
                    text = model,
                    textAlign = TextAlign.Start,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Light,
                    fontStyle = FontStyle.Italic,
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            Row {
                ModelCardContentField("Parameters", parameters)

                Spacer(modifier = Modifier.weight(1f))

                ModelCardContentField("Size", size)
            }

            BenchmarkResultTable(rawTable)

            ModelCardContentField("Time spent: ", formatMilliSeconds(result.duration))

            Spacer(modifier = Modifier.height(8.dp))

            Row {
                OutlinedButton(onClick = onRerun) {
                    Icon(
                        imageVector = Icons.Default.Replay,
                        contentDescription = "Run the benchmark again"
                    )
                    Text("Run again", modifier = Modifier.padding(start = 6.dp))
                }

                Spacer(modifier = Modifier.weight(1f))

                FilledTonalButton(onClick = onInfo) {
                    Icon(
                        imageVector = Icons.Default.Info,
                        contentDescription = "Information about what the result means"
                    )
                    Text("How to interpret", modifier = Modifier.padding(start = 6.dp))
                }
            }
        }
    }
}

// Needs to be aligned with `bench` implementation
private val COLUMNS_TO_KEEP = setOf("backend", "test", "t/s")
private val WEIGHTS_EACH_COLUMN = listOf(1f, 1f, 2f)

@Composable
fun BenchmarkResultTable(
    rawTable: TableData,
    columnsToKeep: Set<String> = COLUMNS_TO_KEEP,
    columnWeights: List<Float> = WEIGHTS_EACH_COLUMN
) {
    val (headers, rows) = rawTable.filterColumns(columnsToKeep)

    Column(
        modifier = Modifier
            .padding(horizontal = 12.dp, vertical = 16.dp)
            .border(1.dp, MaterialTheme.colorScheme.outline, shape = RoundedCornerShape(4.dp))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        BenchmarkResultTableRow(headers, columnWeights, isHeader = true)
        HorizontalDivider(thickness = 1.dp)
        rows.forEach { BenchmarkResultTableRow(it, columnWeights) }
    }
}

@Composable
fun BenchmarkResultTableRow(
    cells: List<String>,
    weights: List<Float>? = null,
    isHeader: Boolean = false,
) {
    val effectiveWeights = weights ?: List(cells.size) { 1f }

    Row(modifier = Modifier.fillMaxWidth()) {
        cells.forEachIndexed { index, cell ->
            Text(
                modifier = Modifier.weight(effectiveWeights.getOrElse(index) { 1f }),
                text = cell,
                textAlign = TextAlign.Center,
                style = MaterialTheme.typography.bodyLarge,
                fontWeight = if (isHeader) FontWeight.Normal else FontWeight.Light,
                fontStyle =  if (isHeader) FontStyle.Normal else FontStyle.Italic
            )
        }
    }
}
