package com.example.llama.revamp.ui.screens

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
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import com.example.llama.revamp.engine.InferenceEngine
import com.example.llama.revamp.ui.components.PerformanceAppScaffold
import com.example.llama.revamp.ui.theme.MonospacedTextStyle
import com.example.llama.revamp.viewmodel.BenchmarkViewModel
import com.example.llama.revamp.viewmodel.MainViewModel

@Composable
fun BenchmarkScreen(
    onBackPressed: () -> Unit,
    viewModel: BenchmarkViewModel = hiltViewModel()
) {
    val engineState by viewModel.engineState.collectAsState()
    val benchmarkResults by viewModel.benchmarkResults.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()

    LaunchedEffect(selectedModel) {
        viewModel.runBenchmark()
    }

    PerformanceAppScaffold(
        title = "Chat",
        onNavigateBack = onBackPressed,
        showTemperature = true
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp)
                .verticalScroll(rememberScrollState())
        ) {
            // Model info
            selectedModel?.let { model ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 16.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Text(
                            text = model.name,
                            style = MaterialTheme.typography.titleLarge
                        )

                        Text(
                            text = "${model.parameters} [${model.quantization}]",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }

            // Benchmark results or loading indicator
            when {
                engineState is InferenceEngine.State.Benchmarking -> {
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
    }
}
