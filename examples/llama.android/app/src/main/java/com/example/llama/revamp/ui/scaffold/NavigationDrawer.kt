package com.example.llama.revamp.ui.scaffold

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Folder
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.DrawerState
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.example.llama.revamp.APP_NAME
import com.example.llama.revamp.navigation.AppDestinations
import com.example.llama.revamp.navigation.NavigationActions
import kotlinx.coroutines.launch

/**
 * App navigation drawer that provides access to different sections of the app.
 * Gesture opening can be controlled based on current screen.
 */
@Composable
fun AppNavigationDrawer(
    drawerState: DrawerState,
    navigationActions: NavigationActions,
    gesturesEnabled: Boolean,
    currentRoute: String,
    content: @Composable () -> Unit
) {
    val coroutineScope = rememberCoroutineScope()
    val configuration = LocalConfiguration.current

    // Calculate drawer width (60% of screen width)
    val drawerWidth = (configuration.screenWidthDp * 0.6).dp

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
                    currentRoute = currentRoute,
                    onNavigate = { destination ->
                        coroutineScope.launch {
                            drawerState.close()
                            destination()
                        }
                    },
                    navigationActions = navigationActions
                )
            }
        },
        content = content
    )
}

@Composable
private fun DrawerContent(
    currentRoute: String,
    onNavigate: ((Function0<Unit>)) -> Unit,
    navigationActions: NavigationActions,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        // App Header
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = APP_NAME,
                style = MaterialTheme.typography.titleLarge,
                textAlign = TextAlign.Center
            )

            Text(
                text = "v1.0.0",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 4.dp)
            )
        }

        HorizontalDivider()

        Spacer(modifier = Modifier.height(16.dp))

        // Main Navigation Items
        Text(
            text = "Navigation",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(start = 8.dp, bottom = 8.dp)
        )

        DrawerNavigationItem(
            icon = Icons.Default.Home,
            label = "Home",
            isSelected = currentRoute == AppDestinations.MODEL_SELECTION_ROUTE,
            onClick = { onNavigate { navigationActions.navigateToModelSelection() } }
        )

        Spacer(modifier = Modifier.height(24.dp))

        // Settings Group
        Text(
            text = "Settings",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(start = 8.dp, bottom = 8.dp)
        )

        DrawerNavigationItem(
            icon = Icons.Default.Settings,
            label = "General Settings",
            isSelected = currentRoute == AppDestinations.SETTINGS_GENERAL_ROUTE,
            onClick = { onNavigate { navigationActions.navigateToSettingsGeneral() } }
        )

        DrawerNavigationItem(
            icon = Icons.Default.Folder,
            label = "Models",
            isSelected = currentRoute == AppDestinations.MODELS_MANAGEMENT_ROUTE,
            onClick = { onNavigate { navigationActions.navigateToModelsManagement() } }
        )
    }
}

@Composable
private fun DrawerNavigationItem(
    icon: ImageVector,
    label: String,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    val backgroundColor = if (isSelected) {
        MaterialTheme.colorScheme.primaryContainer
    } else {
        MaterialTheme.colorScheme.surface
    }

    val contentColor = if (isSelected) {
        MaterialTheme.colorScheme.onPrimaryContainer
    } else {
        MaterialTheme.colorScheme.onSurface
    }

    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(vertical = 4.dp),
        color = backgroundColor,
        shape = MaterialTheme.shapes.small
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 12.dp, horizontal = 16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = icon,
                contentDescription = label,
                tint = contentColor,
                modifier = Modifier.size(24.dp)
            )

            Text(
                text = label,
                style = MaterialTheme.typography.bodyLarge,
                color = contentColor,
                modifier = Modifier.padding(start = 16.dp)
            )
        }
    }
}
