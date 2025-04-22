package com.example.llama.revamp.ui.scaffold.topbar

import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.SdStorage
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.llama.revamp.monitoring.StorageMetrics
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun StorageTopBar(
    title: String,
    storageMetrics: StorageMetrics?,
    onNavigateBack: (() -> Unit)? = null,
) {
    TopAppBar(
        title = { Text(title) },
        navigationIcon = {
            if (onNavigateBack != null) {
                IconButton(onClick = onNavigateBack) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back"
                    )
                }
            }
        },
        actions = {
            storageMetrics?.let {
                StorageIndicator(storageMetrics = it)
            }
        },
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = MaterialTheme.colorScheme.surface,
            titleContentColor = MaterialTheme.colorScheme.onSurface
        )
    )
}

@Composable
private fun StorageIndicator(storageMetrics: StorageMetrics) {
    val usedGb = storageMetrics.usedGB
    val availableGb = storageMetrics.availableGB

    Row(
        modifier = Modifier.Companion.padding(end = 8.dp),
        verticalAlignment = Alignment.Companion.CenterVertically
    ) {
        Icon(
            imageVector = Icons.Default.SdStorage,
            contentDescription = "Storage",
            tint = when {
                availableGb < 5.0f -> MaterialTheme.colorScheme.error
                availableGb < 10.0f -> MaterialTheme.colorScheme.tertiary
                else -> MaterialTheme.colorScheme.onSurface
            }
        )

        Spacer(modifier = Modifier.Companion.width(2.dp))

        Text(
            text = String.Companion.format(
                Locale.getDefault(),
                "%.1f / %.1f GB",
                usedGb,
                availableGb
            ),
            style = MaterialTheme.typography.bodySmall
        )
    }
}
