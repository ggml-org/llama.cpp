package com.example.llama.revamp.ui.components

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Divider
import androidx.compose.material3.DrawerState
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.example.llama.revamp.navigation.NavigationActions
import kotlinx.coroutines.launch

/**
 * App navigation drawer that provides access to different sections of the app.
 * Gesture opening is disabled when a model is loaded to prevent accidental navigation,
 * but gesture dismissal is always enabled.
 */
@Composable
fun AppNavigationDrawer(
    drawerState: DrawerState,
    navigationActions: NavigationActions,
    modelLoaded: Boolean = false,
    content: @Composable () -> Unit
) {
    val coroutineScope = rememberCoroutineScope()
    val configuration = LocalConfiguration.current

    // Calculate drawer width (60% of screen width)
    val drawerWidth = (configuration.screenWidthDp * 0.6).dp

    // Determine if gestures should be enabled
    // Always enable when drawer is open (to allow dismissal)
    // Only enable when model is not loaded (to prevent accidental opening)
    val gesturesEnabled by remember(drawerState.currentValue, modelLoaded) {
        derivedStateOf {
            drawerState.currentValue == DrawerValue.Open || !modelLoaded
        }
    }

    // Handle back button to close drawer if open
    BackHandler(enabled = drawerState.currentValue == DrawerValue.Open) {
        coroutineScope.launch {
            drawerState.close()
        }
    }

    ModalNavigationDrawer(
        drawerState = drawerState,
        gesturesEnabled = gesturesEnabled,
        drawerContent = {
            ModalDrawerSheet(
                modifier = Modifier.width(drawerWidth)
            ) {
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
