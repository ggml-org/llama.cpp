package com.example.llama.revamp.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import com.example.llama.revamp.engine.InferenceEngine
import com.example.llama.revamp.ui.components.PerformanceAppScaffold
import com.example.llama.revamp.viewmodel.MainViewModel
import com.example.llama.revamp.viewmodel.Message
import kotlinx.coroutines.launch

/**
 * Screen for LLM conversation with user.
 */
@Composable
fun ConversationScreen(
    onBackPressed: () -> Unit,
    viewModel: MainViewModel = hiltViewModel()
) {
    val engineState by viewModel.engineState.collectAsState()
    val messages by viewModel.messages.collectAsState()
    val systemPrompt by viewModel.systemPrompt.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()

    val isProcessing = engineState is InferenceEngine.State.ProcessingUserPrompt
    val isGenerating = engineState is InferenceEngine.State.Generating

    val listState = rememberLazyListState()
    var inputText by remember { mutableStateOf("") }
    val coroutineScope = rememberCoroutineScope()
    val lifecycleOwner = LocalLifecycleOwner.current

    // Auto-scroll to bottom when messages change or when typing
    val shouldScrollToBottom by remember(messages.size, isGenerating) {
        derivedStateOf { true }
    }

    LaunchedEffect(shouldScrollToBottom, messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    // Set up lifecycle-aware message monitoring
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                // Scroll to bottom when returning to the screen
                if (messages.isNotEmpty()) {
                    coroutineScope.launch {
                        listState.scrollToItem(messages.size - 1)
                    }
                }
            }
        }

        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
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
        ) {
            // System prompt display (collapsible)
            AnimatedSystemPrompt(selectedModel?.name, systemPrompt)

            // Messages list
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
            ) {
                ConversationMessageList(
                    messages = messages,
                    listState = listState,
                )
            }

            // Input area
            ConversationInputField(
                value = inputText,
                onValueChange = { inputText = it },
                onSendClick = {
                    if (inputText.isNotBlank()) {
                        viewModel.sendMessage(inputText)
                        inputText = ""
                    }
                },
                isEnabled = !isProcessing && !isGenerating
            )
        }
    }
}

@Composable
fun AnimatedSystemPrompt(modelName: String?, systemPrompt: String?) {
    var expanded by remember { mutableStateOf(false) }

    // TODO-han.yin: add model name into this card, on top of system prompt!

    if (!systemPrompt.isNullOrBlank()) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            onClick = {
                expanded = !expanded
            }
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "System Prompt",
                        style = MaterialTheme.typography.titleMedium,
                        modifier = Modifier.weight(1f)
                    )

                    Text(
                        text = if (expanded) "Hide" else "Show",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.primary
                    )
                }

                AnimatedVisibility(
                    visible = expanded,
                    enter = fadeIn() + expandVertically(),
                    exit = fadeOut() + shrinkVertically()
                ) {
                    Text(
                        text = systemPrompt,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(top = 8.dp)
                    )
                }
            }
        }
    }
}

@Composable
fun ConversationMessageList(
    messages: List<Message>,
    listState: LazyListState,
) {
    LazyColumn(
        state = listState,
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 16.dp),
        reverseLayout = false
    ) {
        items(
            items = messages,
            key = { "${it::class.simpleName}_${it.timestamp}" }
        ) { message ->
            MessageBubble(message = message)
        }

        // Add extra space at the bottom for better UX
        item { Spacer(modifier = Modifier.height(16.dp)) }
    }
}

@Composable
fun MessageBubble(message: Message) {
    when (message) {
        is Message.User -> UserMessageBubble(
            formattedTime = message.formattedTime,
            content = message.content
        )
        is Message.Assistant.Ongoing -> AssistantMessageBubble(
            formattedTime = message.formattedTime,
            content = message.content,
            isThinking = message.content.isBlank(),
            isComplete = false,
            metrics = null
        )
        is Message.Assistant.Completed -> AssistantMessageBubble(
            formattedTime = message.formattedTime,
            content = message.content,
            isThinking = false,
            isComplete = true,
            metrics = message.metrics.text
        )
    }
}

@Composable
fun UserMessageBubble(content: String, formattedTime: String) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 8.dp),
        horizontalAlignment = Alignment.End
    ) {
        // Timestamp above bubble
        Text(
            text = formattedTime,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
            modifier = Modifier.padding(bottom = 4.dp)
        )

        Row {
            Spacer(modifier = Modifier.weight(1f))

            Card(
                shape = RoundedCornerShape(16.dp, 4.dp, 16.dp, 16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                ),
                elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
            ) {
                Text(
                    text = content,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                    modifier = Modifier.padding(12.dp)
                )
            }
        }
    }
}

@Composable
fun AssistantMessageBubble(
    formattedTime: String,
    content: String,
    isThinking: Boolean,
    isComplete: Boolean,
    metrics: String? = null
) {
    Row(
        verticalAlignment = Alignment.Top,
    ) {
        // Assistant avatar
        Box(
            modifier = Modifier
                .size(36.dp)
                .clip(CircleShape)
                .background(MaterialTheme.colorScheme.primary),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = "AI",
                color = MaterialTheme.colorScheme.onPrimary,
                style = MaterialTheme.typography.labelMedium
            )
        }

        Spacer(modifier = Modifier.width(8.dp))

        Column(
            modifier = Modifier
                .fillMaxWidth()
        ) {
            // Timestamp above bubble
            if (formattedTime.isNotBlank()) {
                Text(
                    text = formattedTime,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                    modifier = Modifier.padding(bottom = 4.dp)
                )
            }

            Card(
                shape = RoundedCornerShape(4.dp, 16.dp, 16.dp, 16.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                ),
                elevation = CardDefaults.cardElevation(defaultElevation = 1.dp)
            ) {
                // Show actual content
                Text(
                    modifier = Modifier.padding(12.dp),
                    text = content,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            // Show metrics or generation status below the bubble
            Row(
                modifier = Modifier.height(20.dp).padding(top = 4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                if (!isComplete) {
                    PulsatingDots(small = true)

                    Spacer(modifier = Modifier.width(4.dp))

                    Text(
                        text = if (isThinking) "Thinking" else "",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary
                    )
                } else if (metrics != null) {
                    // Show metrics when message is complete
                    Text(
                        text = metrics,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                    )
                }
            }
        }
    }
}

@Composable
fun PulsatingDots(small: Boolean = false) {
    val transition = rememberInfiniteTransition(label = "dots")

    val animations = List(3) { index ->
        transition.animateFloat(
            initialValue = 0f,
            targetValue = 1f,
            animationSpec = infiniteRepeatable(
                animation = tween(
                    durationMillis = 1000,
                    delayMillis = index * 300,
                    easing = LinearEasing
                ),
                repeatMode = RepeatMode.Reverse
            ),
            label = "dot-$index"
        )
    }

    Row(verticalAlignment = Alignment.CenterVertically) {
        animations.forEach { animation ->
            Spacer(modifier = Modifier.width(2.dp))

            Box(
                modifier = Modifier
                    .size(if (small) 5.dp else 8.dp)
                    .clip(CircleShape)
                    .background(
                        color = MaterialTheme.colorScheme.primary.copy(
                            alpha = 0.3f + (animation.value * 0.7f)
                        )
                    )
            )

            Spacer(modifier = Modifier.width(2.dp))
        }
    }
}

@Composable
fun ConversationInputField(
    value: String,
    onValueChange: (String) -> Unit,
    onSendClick: () -> Unit,
    isEnabled: Boolean
) {
    Surface(
        modifier = Modifier
            .fillMaxWidth(),
        shadowElevation = 4.dp,
        color = MaterialTheme.colorScheme.surface
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalAlignment = Alignment.Bottom
        ) {
            OutlinedTextField(
                value = value,
                onValueChange = onValueChange,
                modifier = Modifier
                    .weight(1f)
                    .padding(end = 8.dp),
                placeholder = { Text("Message Kleidi LLaMA...") },
                maxLines = 5,
                enabled = isEnabled,
                colors = TextFieldDefaults.colors(
                    unfocusedContainerColor = Color.Transparent,
                    focusedContainerColor = Color.Transparent
                ),
                shape = RoundedCornerShape(24.dp)
            )

            IconButton(
                onClick = onSendClick,
                enabled = value.isNotBlank() && isEnabled,
                modifier = Modifier
                    .padding(bottom = 4.dp)
                    .size(48.dp)
            ) {
                if (isEnabled) {
                    Icon(
                        imageVector = Icons.Default.Send,
                        contentDescription = "Send message",
                        tint = if (value.isNotBlank())
                            MaterialTheme.colorScheme.primary
                        else
                            MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
                    )
                } else {
                    CircularProgressIndicator(
                        modifier = Modifier.size(24.dp),
                        strokeWidth = 2.dp,
                        strokeCap = StrokeCap.Round
                    )
                }
            }
        }
    }
}
