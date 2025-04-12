package com.example.llama.revamp.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DrawerState
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Switch
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.llama.revamp.data.model.SystemPrompt
import com.example.llama.revamp.data.repository.SystemPromptRepository
import com.example.llama.revamp.engine.InferenceEngine
import com.example.llama.revamp.navigation.NavigationActions
import com.example.llama.revamp.ui.components.AppScaffold
import com.example.llama.revamp.util.ViewModelFactoryProvider
import com.example.llama.revamp.viewmodel.SystemPromptViewModel

enum class SystemPromptTab {
    PRESETS, CUSTOM, RECENTS
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalFoundationApi::class)
@Composable
fun ModeSelectionScreen(
    engineState: InferenceEngine.State,
    onBenchmarkSelected: () -> Unit,
    onConversationSelected: (String?) -> Unit,
    onBackPressed: () -> Unit,
    drawerState: DrawerState,
    navigationActions: NavigationActions
) {
    // Set up SystemPromptViewModel
    val context = LocalContext.current
    val repository = remember { SystemPromptRepository(context) }
    val factory = remember { ViewModelFactoryProvider.getSystemPromptViewModelFactory(repository) }
    val viewModel: SystemPromptViewModel = viewModel(factory = factory)

    val presetPrompts by viewModel.presetPrompts.collectAsState()
    val recentPrompts by viewModel.recentPrompts.collectAsState()

    var selectedMode by remember { mutableStateOf<Mode?>(null) }
    var useSystemPrompt by remember { mutableStateOf(false) }
    var selectedPrompt by remember { mutableStateOf<SystemPrompt?>(null) }
    var selectedTab by remember { mutableStateOf(SystemPromptTab.PRESETS) }
    var customPromptText by remember { mutableStateOf("") }
    var expandedPromptId by remember { mutableStateOf<String?>(null) }

    // Automatically select first preset and expand it
    LaunchedEffect(presetPrompts) {
        if (presetPrompts.isNotEmpty() && selectedPrompt == null) {
            val firstPreset = presetPrompts.first()
            selectedPrompt = firstPreset
            expandedPromptId = firstPreset.id
        }
    }

    // Determine if a system prompt is actually selected/entered when the switch is on
    val hasActiveSystemPrompt = when {
        !useSystemPrompt -> true  // Not using system prompt, so this is fine
        selectedTab == SystemPromptTab.CUSTOM -> customPromptText.isNotBlank()
        else -> selectedPrompt != null
    }

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

            // Conversation card with integrated system prompt
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 8.dp)
                    // Only use weight if system prompt is active, otherwise wrap content
                    .then(if (useSystemPrompt) Modifier.weight(1f) else Modifier)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        // Only fill height if system prompt is active
                        .then(if (useSystemPrompt) Modifier.fillMaxSize() else Modifier)
                ) {
                    // Conversation option
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .selectable(
                                selected = selectedMode == Mode.CONVERSATION,
                                onClick = { selectedMode = Mode.CONVERSATION },
                                enabled = !isLoading,
                                role = Role.RadioButton
                            )
                            .padding(16.dp),
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

                    // System prompt row with switch
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "System prompt",
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier
                                .padding(start = 32.dp) // Align with radio text
                                .weight(1f)
                        )

                        Switch(
                            checked = useSystemPrompt,
                            onCheckedChange = {
                                useSystemPrompt = it
                                if (it && selectedMode != Mode.CONVERSATION) {
                                    selectedMode = Mode.CONVERSATION
                                }
                            },
                            enabled = !isLoading
                        )
                    }

                    // System prompt content (visible when switch is on)
                    AnimatedVisibility(
                        visible = useSystemPrompt && selectedMode == Mode.CONVERSATION,
                        enter = fadeIn() + expandVertically(),
                        exit = fadeOut() + shrinkVertically()
                    ) {
                        Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .fillMaxSize() // Fill remaining card space
                                .padding(horizontal = 16.dp, vertical = 8.dp)
                        ) {
                            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

                            // Tab selector using SegmentedButton
                            SingleChoiceSegmentedButtonRow(
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                SegmentedButton(
                                    selected = selectedTab == SystemPromptTab.PRESETS,
                                    onClick = { selectedTab = SystemPromptTab.PRESETS },
                                    shape = SegmentedButtonDefaults.itemShape(index = 0, count = 3),
                                    icon = {
                                        if (selectedTab == SystemPromptTab.PRESETS) {
                                            Icon(
                                                imageVector = Icons.Default.Check,
                                                contentDescription = null
                                            )
                                        }
                                    },
                                    label = { Text("Presets") }
                                )

                                SegmentedButton(
                                    selected = selectedTab == SystemPromptTab.CUSTOM,
                                    onClick = { selectedTab = SystemPromptTab.CUSTOM },
                                    shape = SegmentedButtonDefaults.itemShape(index = 1, count = 3),
                                    icon = {
                                        if (selectedTab == SystemPromptTab.CUSTOM) {
                                            Icon(
                                                imageVector = Icons.Default.Check,
                                                contentDescription = null
                                            )
                                        }
                                    },
                                    label = { Text("Custom") }
                                )

                                SegmentedButton(
                                    selected = selectedTab == SystemPromptTab.RECENTS,
                                    onClick = { selectedTab = SystemPromptTab.RECENTS },
                                    shape = SegmentedButtonDefaults.itemShape(index = 2, count = 3),
                                    icon = {
                                        if (selectedTab == SystemPromptTab.RECENTS) {
                                            Icon(
                                                imageVector = Icons.Default.Check,
                                                contentDescription = null
                                            )
                                        }
                                    },
                                    label = { Text("Recents") }
                                )
                            }

                            Spacer(modifier = Modifier.height(16.dp))

                            // Content based on selected tab
                            when (selectedTab) {
                                SystemPromptTab.PRESETS -> {
                                    if (presetPrompts.isEmpty()) {
                                        Text(
                                            text = "No preset prompts available.",
                                            style = MaterialTheme.typography.bodyMedium,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            modifier = Modifier.padding(16.dp)
                                        )
                                    } else {
                                        PromptList(
                                            prompts = presetPrompts,
                                            selectedPromptId = selectedPrompt?.id,
                                            expandedPromptId = expandedPromptId,
                                            onPromptSelected = {
                                                selectedPrompt = it
                                                expandedPromptId = it.id
                                            },
                                            onExpandPrompt = { expandedPromptId = it }
                                        )
                                    }
                                }

                                SystemPromptTab.CUSTOM -> {
                                    // Custom prompt editor (fill remaining space)
                                    OutlinedTextField(
                                        value = customPromptText,
                                        onValueChange = {
                                            customPromptText = it
                                            // Deselect any preset prompt if typing custom
                                            if (it.isNotBlank()) {
                                                selectedPrompt = null
                                            }
                                        },
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .fillMaxSize(),  // Fill available space
                                        label = { Text("Enter system prompt") },
                                        placeholder = { Text("You are a helpful assistant...") },
                                        minLines = 5
                                    )
                                }

                                SystemPromptTab.RECENTS -> {
                                    if (recentPrompts.isEmpty()) {
                                        Text(
                                            text = "No recent prompts found.",
                                            style = MaterialTheme.typography.bodyMedium,
                                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                                            modifier = Modifier.padding(16.dp)
                                        )
                                    } else {
                                        PromptList(
                                            prompts = recentPrompts,
                                            selectedPromptId = selectedPrompt?.id,
                                            expandedPromptId = expandedPromptId,
                                            onPromptSelected = {
                                                selectedPrompt = it
                                                expandedPromptId = it.id
                                            },
                                            onExpandPrompt = { expandedPromptId = it }
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Flexible spacer when system prompt is not active
            if (!useSystemPrompt) {
                Spacer(modifier = Modifier.weight(1f))
            }

            // Start button
            Button(
                onClick = {
                    when (selectedMode) {
                        Mode.BENCHMARK -> onBenchmarkSelected()
                        Mode.CONVERSATION -> {
                            val systemPrompt = if (useSystemPrompt) {
                                when (selectedTab) {
                                    SystemPromptTab.PRESETS, SystemPromptTab.RECENTS ->
                                        selectedPrompt?.let { prompt ->
                                            // Save the prompt to recent prompts database
                                            viewModel.savePromptToRecents(prompt)
                                            prompt.content
                                        }
                                    SystemPromptTab.CUSTOM ->
                                        customPromptText.takeIf { it.isNotBlank() }?.also { promptText ->
                                            // Save custom prompt to database
                                            viewModel.saveCustomPromptToRecents(promptText)
                                        }
                                }
                            } else null
                            onConversationSelected(systemPrompt)
                        }
                        null -> { /* No mode selected */ }
                    }
                },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp),
                enabled = selectedMode != null && !isLoading &&
                    (!useSystemPrompt || hasActiveSystemPrompt)
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
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun PromptList(
    prompts: List<SystemPrompt>,
    selectedPromptId: String?,
    expandedPromptId: String?,
    onPromptSelected: (SystemPrompt) -> Unit,
    onExpandPrompt: (String) -> Unit
) {
    LazyColumn(
        modifier = Modifier
            .fillMaxWidth()
            .fillMaxSize(), // Fill available space
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        items(
            items = prompts,
            key = { it.id }
        ) { prompt ->
            val isSelected = selectedPromptId == prompt.id
            val isExpanded = expandedPromptId == prompt.id

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .animateItemPlacement()
                    .selectable(
                        selected = isSelected,
                        onClick = {
                            onPromptSelected(prompt)
                            onExpandPrompt(prompt.id)
                        }
                    )
                    .padding(vertical = 8.dp)
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    RadioButton(
                        selected = isSelected,
                        onClick = null // Handled by selectable
                    )

                    Column(
                        modifier = Modifier
                            .weight(1f)
                            .padding(start = 8.dp)
                    ) {
                        Text(
                            text = prompt.title,
                            style = MaterialTheme.typography.titleSmall,
                            color = if (isSelected)
                                MaterialTheme.colorScheme.primary
                            else
                                MaterialTheme.colorScheme.onSurface
                        )

                        Text(
                            text = prompt.content,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = if (isExpanded) Int.MAX_VALUE else 2,
                            overflow = if (isExpanded) TextOverflow.Visible else TextOverflow.Ellipsis
                        )
                    }
                }

                if (prompt.id != prompts.last().id) {
                    HorizontalDivider(
                        modifier = Modifier.padding(top = 8.dp, start = 40.dp)
                    )
                }
            }
        }
    }
}

enum class Mode {
    BENCHMARK,
    CONVERSATION
}
