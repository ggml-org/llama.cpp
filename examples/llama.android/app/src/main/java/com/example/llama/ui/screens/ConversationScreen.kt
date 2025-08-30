package com.example.llama.ui.screens

import android.llama.cpp.InferenceEngine.State
import androidx.activity.compose.BackHandler
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
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
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.example.llama.data.model.ModelInfo
import com.example.llama.engine.ModelLoadingMetrics
import com.example.llama.ui.components.ModelCardContentArchitectureRow
import com.example.llama.ui.components.ModelCardContentContextRow
import com.example.llama.ui.components.ModelCardContentField
import com.example.llama.ui.components.ModelCardCoreExpandable
import com.example.llama.ui.components.ModelUnloadDialogHandler
import com.example.llama.util.formatMilliSeconds
import com.example.llama.viewmodel.ConversationViewModel
import com.example.llama.viewmodel.Message
import kotlinx.coroutines.FlowPreview
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.launch

private const val AUTO_REPOSITION_INTERVAL = 150L

/**
 * Screen for LLM conversation with user.
 */
@OptIn(FlowPreview::class)
@Composable
fun ConversationScreen(
    loadingMetrics: ModelLoadingMetrics,
    onNavigateBack: () -> Unit,
    viewModel: ConversationViewModel
) {
    // View model states
    val engineState by viewModel.engineState.collectAsState()
    val unloadDialogState by viewModel.unloadModelState.collectAsState()

    val showModelCard by viewModel.showModelCard.collectAsState()
    val selectedModel by viewModel.selectedModel.collectAsState()
    val systemPrompt by viewModel.systemPrompt.collectAsState()

    val messages by viewModel.messages.collectAsState()

    val isGenerating = engineState is State.Generating

    // UI states
    val lifecycleOwner = LocalLifecycleOwner.current
    val coroutineScope = rememberCoroutineScope()
    var isModelCardExpanded by remember { mutableStateOf(false) }
    val listState = rememberLazyListState()

    // Track the actual rendered size of the last message bubble
    var lastMessageBubbleHeight by remember { mutableIntStateOf(0) }

    // Track if user has manually scrolled up
    var userHasScrolledUp by remember { mutableStateOf(false) }

    // Detect if user is at the very bottom
    val isAtBottom = remember {
        derivedStateOf {
            val layoutInfo = listState.layoutInfo
            val lastVisibleItem = layoutInfo.visibleItemsInfo.lastOrNull()

            // At bottom if we can see the bottom spacer
            lastVisibleItem != null && lastVisibleItem.index == messages.size
        }
    }

    // Reset scroll flag when user returns to bottom
    LaunchedEffect(Unit) {
        snapshotFlow {
            listState.layoutInfo.visibleItemsInfo.lastOrNull()?.index == messages.size
        }
            .distinctUntilChanged()
            .filter { it }
            .collect { userHasScrolledUp = false }
    }

    // Then scroll to bottom spacer instead
    LaunchedEffect(isGenerating) {
        snapshotFlow {
            listState.layoutInfo.visibleItemsInfo.find {
                it.index == messages.size - 1
            }?.size ?: 0
        }
            .distinctUntilChanged()
            .collect { currentHeight ->
                // Only reposition if:
                if (currentHeight > lastMessageBubbleHeight     // 1. Height increased (new line)
                    && !userHasScrolledUp                       // 2. User hasn't scrolled up
                    && isAtBottom.value                         // 3. User is at bottom
                ) {
                    lastMessageBubbleHeight = currentHeight
                    listState.scrollToItem(index = messages.size, scrollOffset = 0)
                } else if (currentHeight > 0) {
                    lastMessageBubbleHeight = currentHeight
                }
            }
    }

    // Detect manual scrolling
    LaunchedEffect(listState.isScrollInProgress) {
        if (listState.isScrollInProgress && !isAtBottom.value) {
            userHasScrolledUp = true
        }
    }

    // Set up lifecycle-aware message monitoring
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            // Scroll to bottom when returning to the screen
            if (event == Lifecycle.Event.ON_RESUME) {
                if (messages.isNotEmpty()) {
                    coroutineScope.launch {
                        listState.scrollToItem(messages.size)
                    }
                }
            }
        }

        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
        }
    }

    // Handle back navigation requests
    BackHandler {
        viewModel.onBackPressed(onNavigateBack)
    }

    Box(
        modifier = Modifier.fillMaxSize()
    ) {
        ConversationMessageList(
            messages = messages,
            listState = listState,
        )

        if (showModelCard) {
            selectedModel?.let {
                Box(
                    modifier = Modifier.fillMaxWidth().padding(16.dp).align(Alignment.TopCenter)
                ) {
                    ModelCardWithSystemPrompt(
                        model = it,
                        loadingMetrics = loadingMetrics,
                        systemPrompt = systemPrompt,
                        isExpanded = isModelCardExpanded,
                        onExpanded = { isModelCardExpanded = !isModelCardExpanded }
                    )
                }
            }
        }
    }

    // Unload confirmation dialog
    ModelUnloadDialogHandler(
        message =  "Going back will unload the current model and clear the whole conversation.",
        unloadModelState = unloadDialogState,
        onUnloadConfirmed = { viewModel.onUnloadConfirmed(onNavigateBack) },
        onUnloadDismissed = { viewModel.onUnloadDismissed() },
        onNavigateBack = onNavigateBack,
    )
}


@Composable
fun ModelCardWithSystemPrompt(
    model: ModelInfo,
    loadingMetrics: ModelLoadingMetrics,
    systemPrompt: String?,
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

    if (!systemPrompt.isNullOrBlank()) {
        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

        Text(
            text = "System Prompt",
            style = MaterialTheme.typography.bodyLarge,
            fontWeight = FontWeight.Normal,
            color = MaterialTheme.colorScheme.onSurface,
        )

        Spacer(Modifier.height(6.dp))

        Text(
            text = systemPrompt,
            style = MaterialTheme.typography.bodySmall,
            fontWeight = FontWeight.ExtraLight,
            fontStyle = FontStyle.Italic,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Spacer(Modifier.height(6.dp))

        loadingMetrics.systemPromptProcessingTimeMs?.let {
            ModelCardContentField("Processing time", formatMilliSeconds(it))
        }
    }
}

@Composable
private fun ConversationMessageList(
    messages: List<Message>,
    listState: LazyListState,
) {
    LazyColumn(
        state = listState,
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(horizontal = 16.dp, vertical = 16.dp),
        verticalArrangement = Arrangement.Bottom,
    ) {
        items(
            items = messages,
            key = { "${it::class.simpleName}_${it.timestamp}" }
        ) { message ->
            MessageBubble(message = message)
        }

        // Add extra space at the bottom for better UX and a scroll target
        item(key = "bottom-spacer") {
            Spacer(modifier = Modifier.height(36.dp))
        }
    }
}

@Composable
private fun MessageBubble(message: Message) {
    when (message) {
        is Message.User -> UserMessageBubble(
            formattedTime = message.formattedTime,
            content = message.content
        )

        is Message.Assistant.Ongoing -> AssistantMessageBubble(
            formattedTime = message.formattedTime,
            content = message.content,
            isThinking = message.content.isBlank(),
            isGenerating = true,
            metrics = null
        )

        is Message.Assistant.Stopped -> AssistantMessageBubble(
            formattedTime = message.formattedTime,
            content = message.content,
            isThinking = false,
            isGenerating = false,
            metrics = message.metrics.text
        )
    }
}

@Composable
private fun UserMessageBubble(content: String, formattedTime: String) {
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
private fun AssistantMessageBubble(
    formattedTime: String,
    content: String,
    isThinking: Boolean,
    isGenerating: Boolean,
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
                modifier = Modifier
                    .height(20.dp)
                    .padding(top = 4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                if (isGenerating) {
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
private fun PulsatingDots(small: Boolean = false) {
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
