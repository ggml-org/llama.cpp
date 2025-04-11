package com.example.llama.revamp.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Divider
import androidx.compose.material3.DrawerState
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.example.llama.revamp.data.model.SystemPrompt
import com.example.llama.revamp.engine.InferenceEngine
import com.example.llama.revamp.navigation.NavigationActions
import com.example.llama.revamp.ui.components.AppScaffold
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModeSelectionScreen(
    engineState: InferenceEngine.State,
    onBenchmarkSelected: () -> Unit,
    onConversationSelected: (String?) -> Unit,
    onBackPressed: () -> Unit,
    drawerState: DrawerState,
    navigationActions: NavigationActions
) {
    val staffPickedPrompts = remember { SystemPrompt.getStaffPickedPrompts() }
    val recentPrompts = remember { SystemPrompt.getRecentPrompts() }

    var selectedMode by remember { mutableStateOf<Mode?>(null) }
    var useSystemPrompt by remember { mutableStateOf(false) }
    var selectedPrompt by remember { mutableStateOf<SystemPrompt?>(null) }
    var tabIndex by remember { mutableStateOf(0) }

    // Custom prompt sheet state
    val sheetState = rememberModalBottomSheetState()
    var showCustomPromptSheet by remember { mutableStateOf(false) }
    var customPromptText by remember { mutableStateOf("") }

    val coroutineScope = rememberCoroutineScope()

    // Check if we're in a loading state
    val isLoading = engineState !is InferenceEngine.State.Uninitialized &&
        engineState !is InferenceEngine.State.LibraryLoaded &&
        engineState !is InferenceEngine.State.AwaitingUserPrompt

    AppScaffold(
        title = "Select Mode",
        drawerState = drawerState,
        navigationActions = navigationActions,
        onBackPressed = onBackPressed
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp)
        ) {
            // Mode selection cards
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 8.dp)
                    .selectable(
                        selected = selectedMode == Mode.BENCHMARK,
                        onClick = { selectedMode = Mode.BENCHMARK },
                        enabled = !isLoading,
                        role = Role.RadioButton
                    )
            ) {
                Row(
                    modifier = Modifier.padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    RadioButton(
                        selected = selectedMode == Mode.BENCHMARK,
                        onClick = null // handled by parent selectable
                    )
                    Text(
                        text = "Benchmark",
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.padding(start = 8.dp)
                    )
                }
            }

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 8.dp)
                    .selectable(
                        selected = selectedMode == Mode.CONVERSATION,
                        onClick = {
                            selectedMode = Mode.CONVERSATION
                        },
                        enabled = !isLoading,
                        role = Role.RadioButton
                    )
            ) {
                Row(
                    modifier = Modifier.padding(16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    RadioButton(
                        selected = selectedMode == Mode.CONVERSATION,
                        onClick = null // handled by parent selectable
                    )
                    Text(
                        text = "Conversation",
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.padding(start = 8.dp)
                    )
                }
            }

            // System prompt section (only visible when conversation mode is selected)
            AnimatedVisibility(visible = selectedMode == Mode.CONVERSATION) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 8.dp)
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .selectableGroup(),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "Use system prompt",
                                style = MaterialTheme.typography.titleMedium
                            )

                            Spacer(modifier = Modifier.weight(1f))

                            TextButton(
                                onClick = { useSystemPrompt = !useSystemPrompt }
                            ) {
                                Icon(
                                    imageVector = if (useSystemPrompt) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                                    contentDescription = if (useSystemPrompt) "Collapse" else "Expand"
                                )
                            }
                        }

                        AnimatedVisibility(visible = useSystemPrompt) {
                            Column {
                                TabRow(selectedTabIndex = tabIndex) {
                                    Tab(
                                        selected = tabIndex == 0,
                                        onClick = { tabIndex = 0 },
                                        text = { Text("Staff Picks") }
                                    )

                                    Tab(
                                        selected = tabIndex == 1,
                                        onClick = { tabIndex = 1 },
                                        text = { Text("Recent") }
                                    )
                                }

                                // Tab content
                                when (tabIndex) {
                                    0 -> PromptList(
                                        prompts = staffPickedPrompts,
                                        selectedPrompt = selectedPrompt,
                                        onPromptSelected = { selectedPrompt = it }
                                    )
                                    1 -> PromptList(
                                        prompts = recentPrompts,
                                        selectedPrompt = selectedPrompt,
                                        onPromptSelected = { selectedPrompt = it }
                                    )
                                }

                                // Custom prompt button
                                OutlinedButton(
                                    onClick = { showCustomPromptSheet = true },
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(vertical = 8.dp)
                                ) {
                                    Text("Custom prompt...")
                                }
                            }
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.weight(1f))

            // Start button
            Button(
                onClick = {
                    when (selectedMode) {
                        Mode.BENCHMARK -> onBenchmarkSelected()
                        Mode.CONVERSATION -> {
                            val systemPrompt = if (useSystemPrompt) {
                                selectedPrompt?.content ?: customPromptText.takeIf { it.isNotBlank() }
                            } else null
                            onConversationSelected(systemPrompt)
                        }
                        null -> { /* No mode selected */ }
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                enabled = selectedMode != null && !isLoading
            ) {
                if (isLoading) {
                    CircularProgressIndicator(
                        modifier = Modifier
                            .height(24.dp)
                            .width(24.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = when(engineState) {
                            is InferenceEngine.State.LoadingModel -> "Loading model..."
                            is InferenceEngine.State.ProcessingSystemPrompt -> "Processing system prompt..."
                            is InferenceEngine.State.ModelLoaded -> "Preparing conversation..."
                            else -> "Processing..."
                        }
                    )
                } else {
                    Text("Start")
                }
            }
        }

        // Custom prompt bottom sheet
        if (showCustomPromptSheet) {
            ModalBottomSheet(
                onDismissRequest = { showCustomPromptSheet = false },
                sheetState = sheetState
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(16.dp)
                ) {
                    Text(
                        text = "Custom System Prompt",
                        style = MaterialTheme.typography.titleLarge
                    )

                    Spacer(modifier = Modifier.height(16.dp))

                    OutlinedTextField(
                        value = customPromptText,
                        onValueChange = { customPromptText = it },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(200.dp),
                        label = { Text("Enter system prompt") },
                        placeholder = { Text("You are a helpful assistant...") }
                    )

                    Spacer(modifier = Modifier.height(16.dp))

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        TextButton(
                            onClick = {
                                coroutineScope.launch {
                                    sheetState.hide()
                                    showCustomPromptSheet = false
                                }
                            }
                        ) {
                            Text("Cancel")
                        }

                        Spacer(modifier = Modifier.weight(1f))

                        Button(
                            onClick = {
                                selectedPrompt = null
                                coroutineScope.launch {
                                    sheetState.hide()
                                    showCustomPromptSheet = false
                                }
                            },
                            enabled = customPromptText.isNotBlank()
                        ) {
                            Text("Use Custom Prompt")
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun PromptList(
    prompts: List<SystemPrompt>,
    selectedPrompt: SystemPrompt?,
    onPromptSelected: (SystemPrompt) -> Unit
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxWidth()
            .height(200.dp)
            .padding(vertical = 8.dp)
    ) {
        items(prompts) { prompt ->
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .selectable(
                        selected = selectedPrompt?.id == prompt.id,
                        onClick = { onPromptSelected(prompt) },
                        role = Role.RadioButton
                    )
                    .padding(vertical = 8.dp, horizontal = 16.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                RadioButton(
                    selected = selectedPrompt?.id == prompt.id,
                    onClick = null // handled by parent selectable
                )

                Column(
                    modifier = Modifier.padding(start = 16.dp)
                ) {
                    Text(
                        text = prompt.name,
                        style = MaterialTheme.typography.titleMedium
                    )

                    Text(
                        text = prompt.content,
                        style = MaterialTheme.typography.bodySmall,
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Divider(modifier = Modifier.padding(horizontal = 16.dp))
        }
    }
}

enum class Mode {
    BENCHMARK,
    CONVERSATION
}
