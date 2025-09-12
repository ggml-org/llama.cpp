package com.example.llama.ui.scaffold.topbar

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LocalContentColor
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DefaultTopBar(
    title: String,
    titleColor: Color = Color.Unspecified,
    navigationIconTint: Color = LocalContentColor.current,
    onNavigateBack: (() -> Unit)? = null,
    onQuit: (() -> Unit)? = null,
    onMenuOpen: (() -> Unit)? = null
) {
    TopAppBar(
        title = {
            Text(text = title, color = titleColor)
        },
        navigationIcon = {
            when {
                onQuit != null -> {
                    IconButton(onClick = onQuit) {
                        Icon(
                            imageVector = Icons.Default.Close,
                            contentDescription = "Quit",
                            tint = navigationIconTint
                        )
                    }
                }

                onNavigateBack != null -> {
                    IconButton(onClick = onNavigateBack) {
                        Icon(
                            imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "Back",
                            tint = navigationIconTint
                        )
                    }
                }

                onMenuOpen != null -> {
                    IconButton(onClick = onMenuOpen) {
                        Icon(
                            imageVector = Icons.Default.Menu,
                            contentDescription = "Menu",
                            tint = navigationIconTint
                        )
                    }
                }
            }
        },
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = MaterialTheme.colorScheme.surface,
            titleContentColor = MaterialTheme.colorScheme.onSurface
        )
    )
}
