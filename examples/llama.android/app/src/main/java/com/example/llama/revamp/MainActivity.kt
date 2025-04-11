package com.example.llama.revamp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.example.llama.revamp.engine.InferenceEngine
import com.example.llama.revamp.navigation.AppDestinations
import com.example.llama.revamp.navigation.NavigationActions
import com.example.llama.revamp.ui.components.UnloadModelConfirmationDialog
import com.example.llama.revamp.ui.screens.BenchmarkScreen
import com.example.llama.revamp.ui.screens.ConversationScreen
import com.example.llama.revamp.ui.screens.ModelSelectionScreen
import com.example.llama.revamp.ui.screens.ModeSelectionScreen
import com.example.llama.revamp.ui.screens.SettingsScreen
import com.example.llama.revamp.ui.screens.SettingsTab
import com.example.llama.revamp.ui.theme.LlamaTheme
import com.example.llama.revamp.util.ViewModelFactoryProvider
import com.example.llama.revamp.viewmodel.MainViewModel
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            LlamaTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    AppContent()
                }
            }
        }
    }
}

@Composable
fun AppContent() {
    val navController = rememberNavController()
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val coroutineScope = rememberCoroutineScope()

    // Create inference engine
    val inferenceEngine = remember { InferenceEngine() }

    // Create factory for MainViewModel
    val factory = remember { ViewModelFactoryProvider.getMainViewModelFactory(inferenceEngine) }

    // Get ViewModel instance with factory
    val viewModel: MainViewModel = viewModel(factory = factory)

    val engineState by viewModel.engineState.collectAsState()

    val navigationActions = remember(navController) {
        NavigationActions(navController)
    }

    // Model unloading confirmation
    var showUnloadDialog by remember { mutableStateOf(false) }
    var pendingNavigation by remember { mutableStateOf<(() -> Unit)?>(null) }

    // Observe back button
    LaunchedEffect(navController) {
        navController.addOnDestinationChangedListener { _, destination, _ ->
            // Log navigation for debugging
            println("Navigation: ${destination.route}")
        }
    }

    // Handle drawer state
    val openDrawer: () -> Unit = {
        coroutineScope.launch {
            drawerState.open()
        }
    }

    // Main Content
    NavHost(
        navController = navController,
        startDestination = AppDestinations.MODEL_SELECTION_ROUTE
    ) {
        // Model Selection Screen
        composable(AppDestinations.MODEL_SELECTION_ROUTE) {
            ModelSelectionScreen(
                onModelSelected = { modelInfo ->
                    viewModel.selectModel(modelInfo)
                    navigationActions.navigateToModeSelection()
                },
                onManageModelsClicked = {
                    navigationActions.navigateToSettings(SettingsTab.MODEL_MANAGEMENT.name)
                },
                onMenuClicked = openDrawer,
                drawerState = drawerState,
                navigationActions = navigationActions
            )
        }

        // Mode Selection Screen
        composable(AppDestinations.MODE_SELECTION_ROUTE) {
            ModeSelectionScreen(
                engineState = engineState,
                onBenchmarkSelected = {
                    viewModel.prepareForBenchmark()
                    navigationActions.navigateToBenchmark()
                },
                onConversationSelected = { systemPrompt ->
                    viewModel.prepareForConversation(systemPrompt)
                    navigationActions.navigateToConversation()
                },
                onBackPressed = {
                    // Need to unload model before going back
                    if (viewModel.isModelLoaded()) {
                        showUnloadDialog = true
                        pendingNavigation = { navController.popBackStack() }
                    } else {
                        navController.popBackStack()
                    }
                },
                drawerState = drawerState,
                navigationActions = navigationActions
            )
        }

        // Conversation Screen
        composable(AppDestinations.CONVERSATION_ROUTE) {
            ConversationScreen(
                onBackPressed = {
                    // Need to unload model before going back
                    if (viewModel.isModelLoaded()) {
                        showUnloadDialog = true
                        pendingNavigation = { navController.popBackStack() }
                    } else {
                        navController.popBackStack()
                    }
                },
                drawerState = drawerState,
                navigationActions = navigationActions,
                viewModel = viewModel
            )
        }

        // Benchmark Screen
        composable(AppDestinations.BENCHMARK_ROUTE) {
            BenchmarkScreen(
                onBackPressed = {
                    // Need to unload model before going back
                    if (viewModel.isModelLoaded()) {
                        showUnloadDialog = true
                        pendingNavigation = { navController.popBackStack() }
                    } else {
                        navController.popBackStack()
                    }
                },
                onRerunPressed = {
                    viewModel.rerunBenchmark()
                },
                onSharePressed = {
                    // Stub for sharing functionality
                },
                drawerState = drawerState,
                navigationActions = navigationActions,
                viewModel = viewModel
            )
        }

        // Settings Screen
        composable(
            route = "${AppDestinations.SETTINGS_ROUTE}/{tab}",
            arguments = listOf(
                navArgument("tab") {
                    type = NavType.StringType
                    defaultValue = SettingsTab.GENERAL.name
                }
            )
        ) { backStackEntry ->
            val tabName = backStackEntry.arguments?.getString("tab") ?: SettingsTab.GENERAL.name
            val tab = try {
                SettingsTab.valueOf(tabName)
            } catch (e: IllegalArgumentException) {
                SettingsTab.GENERAL
            }

            SettingsScreen(
                selectedTab = tab,
                onBackPressed = { navController.popBackStack() },
                drawerState = drawerState,
                navigationActions = navigationActions
            )
        }
    }

    // Model unload confirmation dialog
    if (showUnloadDialog) {
        UnloadModelConfirmationDialog(
            onConfirm = {
                showUnloadDialog = false
                coroutineScope.launch {
                    viewModel.unloadModel()
                    pendingNavigation?.invoke()
                    pendingNavigation = null
                }
            },
            onDismiss = {
                showUnloadDialog = false
                pendingNavigation = null
            }
        )
    }
}
