package com.example.llama.ui.scaffold.topbar

import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.SdStorage
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.example.llama.monitoring.StorageMetrics
import com.example.llama.ui.scaffold.ScaffoldEvent
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ModelsManagementTopBar(
    title: String,
    storageMetrics: StorageMetrics?,
    onScaffoldEvent: (ScaffoldEvent) -> Unit,
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
                StorageIndicator(storageMetrics = it, onScaffoldEvent = onScaffoldEvent)
            }
        },
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = MaterialTheme.colorScheme.surface,
            titleContentColor = MaterialTheme.colorScheme.onSurface
        )
    )
}

@Composable
private fun StorageIndicator(
    storageMetrics: StorageMetrics,
    onScaffoldEvent: (ScaffoldEvent) -> Unit,
) {

    val usedGb = String.format(Locale.getDefault(), "%.1f", storageMetrics.usedGB)
    val availableGb = String.format(Locale.getDefault(), "%.1f", storageMetrics.availableGB)

    OutlinedButton(
        modifier = Modifier.padding(end = 8.dp),
        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp),
        onClick = {
            onScaffoldEvent(ScaffoldEvent.ShowSnackbar(
                message = "Your models occupy $usedGb GB storage\nRemaining free space available: $availableGb GB",
                withDismissAction = true,
            ))
        }
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                imageVector = Icons.Default.SdStorage,
                contentDescription = "Storage",
                tint = when {
                    storageMetrics.availableGB < 5.0f -> MaterialTheme.colorScheme.error
                    storageMetrics.availableGB < 10.0f -> MaterialTheme.colorScheme.tertiary
                    else -> MaterialTheme.colorScheme.onSurface
                }
            )

            Text(
                modifier = Modifier.padding(start = 4.dp),
                text = "$usedGb / $availableGb GB",
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}
