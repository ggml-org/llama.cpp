package com.example.llama.revamp.ui.screens

import android.llama.cpp.InferenceEngine.State
import androidx.activity.compose.BackHandler
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
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.example.llama.revamp.data.model.SystemPrompt
import com.example.llama.revamp.engine.ModelLoadingMetrics
import com.example.llama.revamp.ui.components.ModelCardCoreExpandable
import com.example.llama.revamp.ui.components.ModelUnloadDialogHandler
import com.example.llama.revamp.ui.components.ScaffoldEvent
import com.example.llama.revamp.viewmodel.ModelLoadingViewModel


enum class Mode {
    BENCHMARK,
    CONVERSATION
}

enum class SystemPromptTab(val label: String) {
    PRESETS("Presets"), CUSTOM("Custom"), RECENTS("Recents")
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalFoundationApi::class)
@Composable
fun ModelLoadingScreen(
    onScaffoldEvent: (ScaffoldEvent) -> Unit,
    onNavigateBack: () -> Unit,
    onNavigateToBenchmark: (ModelLoadingMetrics) -> Unit,
    onNavigateToConversation: (ModelLoadingMetrics) -> Unit,
    viewModel: ModelLoadingViewModel,
) {
    // View model states
    val engineState by viewModel.engineState.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val presetPrompts by viewModel.presetPrompts.collectAsState()
    val recentPrompts by viewModel.recentPrompts.collectAsState()
    val unloadDialogState by viewModel.unloadModelState.collectAsState()

    // UI states
    var isModelCardExpanded by remember { mutableStateOf(false) }
    var selectedMode by remember { mutableStateOf<Mode?>(null) }
    var useSystemPrompt by remember { mutableStateOf(false) }
    var showedSystemPromptWarning by remember { mutableStateOf(false) }
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
    val isLoading = engineState !is State.Initialized && engineState !is State.ModelReady

    // Handle back navigation requests
    BackHandler {
        viewModel.onBackPressed(onNavigateBack)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        // Selected model card
        selectedModel?.let { model ->
            ModelCardCoreExpandable(
                model = model,
                isExpanded = isModelCardExpanded,
                onExpanded = { isModelCardExpanded = !isModelCardExpanded },
            )

            Spacer(modifier = Modifier.height(16.dp))
        }

        // Benchmark card
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 8.dp)
                .selectable(
                    selected = selectedMode == Mode.BENCHMARK,
                    onClick = {
                        selectedMode = Mode.BENCHMARK
                        useSystemPrompt = false
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

        // Conversation card with integrated system prompt picker & editor
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(bottom = 4.dp)
                .selectable(
                    selected = selectedMode == Mode.CONVERSATION,
                    onClick = { selectedMode = Mode.CONVERSATION },
                    enabled = !isLoading,
                    role = Role.RadioButton
                )
                // Only fill height if system prompt is active
                .then(if (useSystemPrompt) Modifier.weight(1f) else Modifier)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 12.dp)
                    // Only fill height if system prompt is active
                    .then(if (useSystemPrompt) Modifier.fillMaxSize() else Modifier)
            ) {
                // Conversation option
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 16.dp, start = 16.dp, end = 16.dp),
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
                        .padding(horizontal = 16.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Use system prompt",
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier
                            .padding(start = 32.dp) // Align with radio text
                            .weight(1f)
                    )

                    Switch(
                        checked = useSystemPrompt,
                        onCheckedChange = {
                            // First show a warning message if not yet
                            if (!showedSystemPromptWarning) {
                                onScaffoldEvent(ScaffoldEvent.ShowSnackbar(
                                    message = "Model may not support system prompt!\nProceed with caution.",
                                ))
                                showedSystemPromptWarning = true
                            }

                            // Then update states
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
                            .fillMaxSize()
                            .padding(start = 48.dp, end = 16.dp)
                    ) {
                        HorizontalDivider(
                            modifier = Modifier
                                .padding(top = 4.dp, bottom = 8.dp)
                        )

                        SystemPromptTabSelector(
                            selectedTab = selectedTab,
                            onTabSelected = { selectedTab = it }
                        )

                        Spacer(modifier = Modifier.height(8.dp))

                        SystemPromptTabContent(
                            selectedTab = selectedTab,
                            presetPrompts = presetPrompts,
                            recentPrompts = recentPrompts,
                            customPromptText = customPromptText,
                            onCustomPromptChange = {
                                customPromptText = it
                                // Deselect any preset prompt if typing custom
                                if (it.isNotBlank()) {
                                    selectedPrompt = null
                                }
                            },
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

        // Flexible spacer when system prompt is not active
        if (!useSystemPrompt) {
            Spacer(modifier = Modifier.weight(1f))
        } else {
            Spacer(modifier = Modifier.height(8.dp))
        }

        // Start button
        Button(
            onClick = {
                when (selectedMode) {
                    Mode.BENCHMARK -> viewModel.onBenchmarkSelected(onNavigateToBenchmark)

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
                                    customPromptText.takeIf { it.isNotBlank() }
                                        ?.also { promptText ->
                                            // Save custom prompt to database
                                            viewModel.saveCustomPromptToRecents(
                                                promptText
                                            )
                                        }
                            }
                        } else null

                        viewModel.onConversationSelected(systemPrompt, onNavigateToConversation)
                    }

                    null -> { /* No mode selected */
                    }
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
                    text = when (engineState) {
                        is State.Initializing, State.Initialized -> "Initializing..."
                        is State.LoadingModel -> "Loading model..."
                        is State.ProcessingSystemPrompt -> "Processing system prompt..."
                        else -> "Processing..."
                    },
                    style = MaterialTheme.typography.titleMedium
                )
            } else {
                Text(text = "Start", style = MaterialTheme.typography.titleMedium)
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
private fun SystemPromptTabSelector(
    selectedTab: SystemPromptTab,
    onTabSelected: (SystemPromptTab) -> Unit
) {
    SingleChoiceSegmentedButtonRow(
        modifier = Modifier.fillMaxWidth()
    ) {
        SystemPromptTab.entries.forEachIndexed { index, tab ->
            SegmentedButton(
                selected = selectedTab == tab,
                onClick = { onTabSelected(tab) },
                shape = SegmentedButtonDefaults.itemShape(
                    index = index,
                    count = SystemPromptTab.entries.size
                ),
                icon = {
                    if (selectedTab == tab) {
                        Icon(
                            imageVector = Icons.Default.Check,
                            contentDescription = null
                        )
                    }
                },
                label = { Text(tab.label) }
            )
        }
    }
}

@Composable
private fun SystemPromptTabContent(
    selectedTab: SystemPromptTab,
    presetPrompts: List<SystemPrompt>,
    recentPrompts: List<SystemPrompt>,
    customPromptText: String,
    onCustomPromptChange: (String) -> Unit,
    selectedPromptId: String?,
    expandedPromptId: String?,
    onPromptSelected: (SystemPrompt) -> Unit,
    onExpandPrompt: (String) -> Unit
) {
    when (selectedTab) {
        SystemPromptTab.PRESETS, SystemPromptTab.RECENTS -> {
            val prompts = if (selectedTab == SystemPromptTab.PRESETS) presetPrompts else recentPrompts

            if (prompts.isEmpty()) {
                Text(
                    text =
                        if (selectedTab == SystemPromptTab.PRESETS) "No preset prompts available."
                        else "No recent prompts found.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(16.dp)
                )
            } else {
                PromptList(
                    prompts = prompts,
                    selectedPromptId = selectedPromptId,
                    expandedPromptId = expandedPromptId,
                    onPromptSelected = onPromptSelected,
                    onExpandPrompt = onExpandPrompt
                )
            }
        }

        SystemPromptTab.CUSTOM -> {
            OutlinedTextField(
                value = customPromptText,
                onValueChange = onCustomPromptChange,
                modifier = Modifier
                    .fillMaxWidth()
                    .fillMaxSize(),
                label = { Text("Enter system prompt") },
                placeholder = { Text("You are a helpful assistant...") },
                minLines = 5
            )
        }
    }
}


@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun PromptList(
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
        verticalArrangement = Arrangement.spacedBy(8.dp),
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
                        modifier = Modifier.padding(top = 8.dp, start = 32.dp)
                    )
                }
            }
        }
    }
}
