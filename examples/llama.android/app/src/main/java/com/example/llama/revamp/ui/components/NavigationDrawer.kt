package com.example.llama.revamp.ui.components

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.DrawerState
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.example.llama.revamp.navigation.NavigationActions
import kotlinx.coroutines.launch

/**
 * App navigation drawer that provides access to different sections of the app.
 */
@Composable
fun AppNavigationDrawer(
    drawerState: DrawerState,
    navigationActions: NavigationActions,
    content: @Composable () -> Unit
) {
    val coroutineScope = rememberCoroutineScope()

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ModalDrawerSheet {
                DrawerContent(
                    onHomeClicked = {
                        coroutineScope.launch {
                            drawerState.close()
                            navigationActions.navigateToModelSelection()
                        }
                    },
                    onSettingsClicked = {
                        coroutineScope.launch {
                            drawerState.close()
                            navigationActions.navigateToSettings()
                        }
                    }
                )
            }
        },
        content = content
    )
}

@Composable
private fun DrawerContent(
    onHomeClicked: () -> Unit,
    onSettingsClicked: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text(
            text = "Local LLM",
            style = MaterialTheme.typography.titleLarge,
            textAlign = TextAlign.Center,
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 16.dp)
        )

        HorizontalDivider()

        Spacer(modifier = Modifier.height(16.dp))

        // Navigation Items
        DrawerNavigationItem(
            icon = Icons.Default.Home,
            label = "Home",
            onClick = onHomeClicked
        )

        DrawerNavigationItem(
            icon = Icons.Default.Settings,
            label = "Settings",
            onClick = onSettingsClicked
        )
    }
}

@Composable
private fun DrawerNavigationItem(
    icon: ImageVector,
    label: String,
    onClick: () -> Unit
) {
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(vertical = 8.dp),
        color = MaterialTheme.colorScheme.surface
    ) {
        Column(
            modifier = Modifier.padding(8.dp)
        ) {
            Icon(
                imageVector = icon,
                contentDescription = label,
                tint = MaterialTheme.colorScheme.primary
            )

            Text(
                text = label,
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(start = 8.dp, top = 4.dp)
            )
        }
    }
}
