package com.example.llama.ui.scaffold.bottombar

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.input.TextFieldLineLimits
import androidx.compose.foundation.text.input.TextFieldState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Badge
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.outlined.AddPhotoAlternate
import androidx.compose.material.icons.outlined.AttachFile
import androidx.compose.material.icons.outlined.Badge
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.BottomAppBarDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import com.example.llama.APP_NAME

@Composable
fun ConversationBottomBar(
    textFieldState: TextFieldState,
    isReady: Boolean,
    onSendClick: () -> Unit,
    showModelCard: Boolean,
    onToggleModelCard: (Boolean) -> Unit,
    onAttachPhotoClick: (() -> Unit)?,
    onAttachFileClick: (() -> Unit)?,
    onAudioInputClick: (() -> Unit)?,
) {
    val placeholder = if (isReady) "Message ${APP_NAME}..." else "Please wait for ${APP_NAME} to finish"

    Column(
        modifier = Modifier.Companion.fillMaxWidth()
    ) {
        Surface(
            modifier = Modifier.Companion.fillMaxWidth(),
            color = BottomAppBarDefaults.containerColor,
            tonalElevation = BottomAppBarDefaults.ContainerElevation,
            shape = RoundedCornerShape(topStart = 32.dp, topEnd = 32.dp)
        ) {
            Box(
                modifier = Modifier.Companion.fillMaxWidth()
                    .padding(start = 16.dp, top = 16.dp, end = 16.dp),
            ) {
                OutlinedTextField(
                    state = textFieldState,
                    modifier = Modifier.Companion.fillMaxWidth(),
                    enabled = isReady,
                    placeholder = { Text(placeholder) },
                    lineLimits = TextFieldLineLimits.MultiLine(maxHeightInLines = 5),
                    colors = TextFieldDefaults.colors(
                        focusedContainerColor = MaterialTheme.colorScheme.secondaryContainer.copy(
                            alpha = 0.5f
                        ),
                        unfocusedContainerColor = MaterialTheme.colorScheme.surface,
                        disabledContainerColor = MaterialTheme.colorScheme.surfaceDim,
                    ),
                    shape = androidx.compose.foundation.shape.RoundedCornerShape(16.dp),
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Companion.Send),
                    onKeyboardAction = {
                        if (isReady) {
                            onSendClick()
                        }
                    }
                )
            }
        }

        BottomAppBar(
            actions = {
                onAttachPhotoClick?.let {
                    IconButton(onClick = it) {
                        Icon(
                            imageVector = Icons.Outlined.AddPhotoAlternate,
                            contentDescription = "Attach a photo",
                        )
                    }
                }

                onAttachFileClick?.let {
                    IconButton(onClick = it) {
                        Icon(
                            imageVector = Icons.Outlined.AttachFile,
                            contentDescription = "Attach a file",
                        )
                    }
                }

                onAudioInputClick?.let {
                    IconButton(onClick = it) {
                        Icon(
                            imageVector = Icons.Default.Mic,
                            contentDescription = "Input with voice",
                        )
                    }
                }

                IconButton(onClick = { onToggleModelCard(!showModelCard) } ) {
                    Icon(
                        imageVector = if (showModelCard) Icons.Default.Badge else Icons.Outlined.Badge,
                        contentDescription = "${if (showModelCard) "Hide" else "Show"} model card"
                    )
                }
            },
            floatingActionButton = {
                FloatingActionButton(
                    onClick = {
                        if (isReady) {
                            onSendClick()
                        }
                    },
                    containerColor = MaterialTheme.colorScheme.primary
                ) {
                    if (isReady) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.Send,
                            contentDescription = "Send message",
                        )
                    } else {
                        CircularProgressIndicator(
                            modifier = Modifier.Companion.size(24.dp),
                            strokeCap = StrokeCap.Companion.Round,
                            color = MaterialTheme.colorScheme.onPrimary,
                        )
                    }
                }
            }
        )
    }
}
