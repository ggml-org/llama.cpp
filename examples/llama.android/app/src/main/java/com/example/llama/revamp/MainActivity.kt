package com.example.llama.revamp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.compose.BackHandler
import androidx.activity.compose.LocalOnBackPressedDispatcherOwner
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.rememberDrawerState
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
import androidx.compose.ui.Modifier
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.example.llama.revamp.navigation.AppDestinations
import com.example.llama.revamp.navigation.NavigationActions
import com.example.llama.revamp.ui.components.AppNavigationDrawer
import com.example.llama.revamp.ui.components.UnloadModelConfirmationDialog
import com.example.llama.revamp.ui.screens.BenchmarkScreen
import com.example.llama.revamp.ui.screens.ConversationScreen
import com.example.llama.revamp.ui.screens.ModelSelectionScreen
import com.example.llama.revamp.ui.screens.ModelsManagementScreen
import com.example.llama.revamp.ui.screens.ModelLoadingScreen
import com.example.llama.revamp.ui.screens.SettingsGeneralScreen
import com.example.llama.revamp.ui.theme.LlamaTheme
import com.example.llama.revamp.viewmodel.MainViewModel
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

@AndroidEntryPoint
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
fun AppContent(
    mainVewModel: MainViewModel = hiltViewModel()
) {
    val coroutineScope = rememberCoroutineScope()

    val navController = rememberNavController()
    val navigationActions = remember(navController) { NavigationActions(navController) }
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)

    val engineState by mainVewModel.engineState.collectAsState()
    // TODO-han.yin: Also use delegate for `isModelLoaded`:
    val isModelLoaded = remember(engineState) { mainVewModel.isModelLoaded() }


    // Model unloading confirmation
    var showUnloadDialog by remember { mutableStateOf(false) }
    var pendingNavigation by remember { mutableStateOf<(() -> Unit)?>(null) }

    // Get current route
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute by remember {
        derivedStateOf { navBackStackEntry?.destination?.route ?: "" }
    }

    // Determine if drawer gestures should be enabled based on route
    val drawerGesturesEnabled by remember(currentRoute, drawerState.currentValue) {
        derivedStateOf {
            // Always allow gesture dismissal when drawer is open
            if (drawerState.currentValue == DrawerValue.Open) {
                true
            } else {
                // Only enable drawer opening by gesture on these screens
                currentRoute == AppDestinations.MODEL_SELECTION_ROUTE ||
                    currentRoute == AppDestinations.SETTINGS_GENERAL_ROUTE ||
                    currentRoute == AppDestinations.MODELS_MANAGEMENT_ROUTE
            }
        }
    }

    // Determine if current route requires model unloading
    val routeNeedsModelUnloading by remember(currentRoute) {
        derivedStateOf {
            currentRoute == AppDestinations.CONVERSATION_ROUTE ||
                currentRoute == AppDestinations.BENCHMARK_ROUTE ||
                currentRoute == AppDestinations.MODEL_LOADING_ROUTE
        }
    }

    // Get local back dispatcher
    val backDispatcher = LocalOnBackPressedDispatcherOwner.current?.onBackPressedDispatcher
    val lifecycleOwner = LocalLifecycleOwner.current

    // Helper function to handle back press with model unloading check
    val handleBackWithModelCheck = {
        if (mainVewModel.isModelLoading()) {
            // If model is still loading, ignore the request
            true // Mark as handled
        } else if (mainVewModel.isModelLoaded()) {
            showUnloadDialog = true
            pendingNavigation = { navController.popBackStack() }
            true // Mark as handled
        } else {
            navController.popBackStack()
            true // Mark as handled
        }
    }

    // Register a system back handler for screens that need unload confirmation
    DisposableEffect(lifecycleOwner, backDispatcher, currentRoute, isModelLoaded) {
        val callback = object : OnBackPressedCallback(
            // Only enable for screens that need model unloading confirmation
            routeNeedsModelUnloading && isModelLoaded
        ) {
            override fun handleOnBackPressed() {
                handleBackWithModelCheck()
            }
        }

        backDispatcher?.addCallback(lifecycleOwner, callback)

        // Remove the callback when the effect leaves the composition
        onDispose {
            callback.remove()
        }
    }

    // Compose BackHandler for added protection (this handles Compose-based back navigation)
    BackHandler(
        enabled = routeNeedsModelUnloading &&
            isModelLoaded &&
            drawerState.currentValue == DrawerValue.Closed
    ) {
        handleBackWithModelCheck()
    }

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

    // Main Content with navigation drawer wrapper
    AppNavigationDrawer(
        drawerState = drawerState,
        navigationActions = navigationActions,
        gesturesEnabled = drawerGesturesEnabled,
        currentRoute = currentRoute
    ) {
        NavHost(
            navController = navController,
            startDestination = AppDestinations.MODEL_SELECTION_ROUTE
        ) {
            // Model Selection Screen
            composable(AppDestinations.MODEL_SELECTION_ROUTE) {
                ModelSelectionScreen(
                    onModelSelected = { modelInfo ->
                        mainVewModel.selectModel(modelInfo)
                        navigationActions.navigateToModelLoading()
                    },
                    onManageModelsClicked = {
                        navigationActions.navigateToModelsManagement()
                    },
                    onMenuClicked = openDrawer,
                    drawerState = drawerState,
                    navigationActions = navigationActions
                )
            }

            // Mode Selection Screen
            composable(AppDestinations.MODEL_LOADING_ROUTE) {
                ModelLoadingScreen(
                    engineState = engineState,
                    onBenchmarkSelected = {
                        mainVewModel.prepareForBenchmark()
                        navigationActions.navigateToBenchmark()
                    },
                    onConversationSelected = { systemPrompt ->
                        // Store a reference to the loading job
                        val loadingJob = coroutineScope.launch {
                            mainVewModel.prepareForConversation(systemPrompt)
                            // Check if the job wasn't cancelled before navigating
                            if (isActive) {
                                navigationActions.navigateToConversation()
                            }
                        }
                        // Update the pendingNavigation handler to cancel any ongoing loading
                        pendingNavigation = {
                            loadingJob.cancel()
                            navController.popBackStack()
                        }
                    },
                    onBackPressed = {
                        // Need to unload model before going back
                        handleBackWithModelCheck()
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
                        handleBackWithModelCheck()
                    },
                    viewModel = mainVewModel
                )
            }

            // Benchmark Screen
            composable(AppDestinations.BENCHMARK_ROUTE) {
                BenchmarkScreen(
                    onBackPressed = {
                        // Need to unload model before going back
                        handleBackWithModelCheck()
                    },
                    onRerunPressed = {
                        mainVewModel.rerunBenchmark()
                    },
                    onSharePressed = {
                        // Stub for sharing functionality
                    },
                    drawerState = drawerState,
                    navigationActions = navigationActions,
                    viewModel = mainVewModel
                )
            }

            // Settings General Screen
            composable(AppDestinations.SETTINGS_GENERAL_ROUTE) {
                SettingsGeneralScreen(
                    onBackPressed = { navController.popBackStack() },
                    drawerState = drawerState,
                    navigationActions = navigationActions,
                    onMenuClicked = openDrawer
                )
            }

            // Models Management Screen
            composable(AppDestinations.MODELS_MANAGEMENT_ROUTE) {
                ModelsManagementScreen(
                    onBackPressed = { navController.popBackStack() },
                )
            }
        }
    }

    // Model unload confirmation dialog
    var isUnloading by remember { mutableStateOf(false) }

    if (showUnloadDialog) {
        UnloadModelConfirmationDialog(
            onConfirm = {
                isUnloading = true
                coroutineScope.launch {
                    mainVewModel.unloadModel()
                    isUnloading = false
                    showUnloadDialog = false
                    pendingNavigation?.invoke()
                    pendingNavigation = null
                }
            },
            onDismiss = {
                if (!isUnloading) {
                    showUnloadDialog = false
                    pendingNavigation = null
                }
            },
            isUnloading = isUnloading
        )
    }
}
