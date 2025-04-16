package com.example.llama.revamp

import android.llama.cpp.InferenceEngine.State
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
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.example.llama.revamp.navigation.AppDestinations
import com.example.llama.revamp.navigation.NavigationActions
import com.example.llama.revamp.ui.components.AnimatedNavHost
import com.example.llama.revamp.ui.components.AppNavigationDrawer
import com.example.llama.revamp.ui.components.UnloadModelConfirmationDialog
import com.example.llama.revamp.ui.screens.BenchmarkScreen
import com.example.llama.revamp.ui.screens.ConversationScreen
import com.example.llama.revamp.ui.screens.ModelLoadingScreen
import com.example.llama.revamp.ui.screens.ModelSelectionScreen
import com.example.llama.revamp.ui.screens.ModelsManagementScreen
import com.example.llama.revamp.ui.screens.SettingsGeneralScreen
import com.example.llama.revamp.ui.theme.LlamaTheme
import com.example.llama.revamp.viewmodel.ConversationViewModel
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
    mainViewModel: MainViewModel = hiltViewModel(),
    conversationViewModel: ConversationViewModel = hiltViewModel(),
) {
    val lifecycleOwner = LocalLifecycleOwner.current
    val coroutineScope = rememberCoroutineScope()

    // Inference engine state
    val engineState by mainViewModel.engineState.collectAsState()
    val isModelUninterruptible by remember(engineState) {
        derivedStateOf {
            engineState is State.LoadingModel
                || engineState is State.Benchmarking
                || engineState is State.ProcessingUserPrompt
                || engineState is State.ProcessingSystemPrompt
        }
    }
    val isModelLoaded by remember(engineState) {
        derivedStateOf {
            engineState !is State.Uninitialized && engineState !is State.LibraryLoaded
        }
    }

    // Navigation
    val navController = rememberNavController()
    val navigationActions = remember(navController) { NavigationActions(navController) }
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute by remember {
        derivedStateOf { navBackStackEntry?.destination?.route ?: "" }
    }
    var pendingNavigation by remember { mutableStateOf<(() -> Unit)?>(null) }
    LaunchedEffect(navController) {
        navController.addOnDestinationChangedListener { _, destination, _ ->
            // Log navigation for debugging
            println("Navigation: ${destination.route}")
        }
    }

    // Determine if current route requires model unloading
    val routeNeedsModelUnloading by remember(currentRoute) {
        derivedStateOf {
            currentRoute == AppDestinations.CONVERSATION_ROUTE
                || currentRoute == AppDestinations.BENCHMARK_ROUTE
                || currentRoute == AppDestinations.MODEL_LOADING_ROUTE
        }
    }
    // Model unloading confirmation
    var showUnloadDialog by remember { mutableStateOf(false) }
    val handleBackWithModelCheck = {
        when {
            isModelUninterruptible -> {
                // If model is non-interruptible at all, ignore the request
                true // Mark as handled
            }
            isModelLoaded -> {
                showUnloadDialog = true
                pendingNavigation = { navigationActions.navigateUp() }
                true // Mark as handled
            }
            else -> {
                navigationActions.navigateUp()
                true // Mark as handled
            }
        }
    }

    // Determine if drawer gestures should be enabled based on route
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val drawerGesturesEnabled by remember(currentRoute, drawerState.currentValue) {
        derivedStateOf {
            // Always allow gesture dismissal when drawer is open
            if (drawerState.currentValue == DrawerValue.Open) true
            // Only enable drawer opening by gesture on these screens
            else currentRoute == AppDestinations.MODEL_SELECTION_ROUTE
        }
    }
    val openDrawer: () -> Unit = { coroutineScope.launch { drawerState.open() } }

    // Register a system back handler for screens that need unload confirmation
    val backDispatcher = LocalOnBackPressedDispatcherOwner.current?.onBackPressedDispatcher
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
    // Added protection to handle Compose-based back navigation
    BackHandler(
        enabled = routeNeedsModelUnloading && isModelLoaded
            && drawerState.currentValue == DrawerValue.Closed
    ) {
        handleBackWithModelCheck()
    }

    // Main Content with navigation drawer wrapper
    AppNavigationDrawer(
        drawerState = drawerState,
        navigationActions = navigationActions,
        gesturesEnabled = drawerGesturesEnabled,
        currentRoute = currentRoute
    ) {
        AnimatedNavHost(
            navController = navController,
            startDestination = AppDestinations.MODEL_SELECTION_ROUTE
        ) {
            // Model Selection Screen
            composable(AppDestinations.MODEL_SELECTION_ROUTE) {
                ModelSelectionScreen(
                    onModelSelected = { modelInfo ->
                        navigationActions.navigateToModelLoading()
                    },
                    onManageModelsClicked = {
                        navigationActions.navigateToModelsManagement()
                    },
                    onMenuClicked = openDrawer,
                )
            }

            // Mode Selection Screen
            composable(AppDestinations.MODEL_LOADING_ROUTE) {
                ModelLoadingScreen(
                    engineState = engineState,
                    onBenchmarkSelected = { prepareJob ->
                        // Wait for preparation to complete, then navigate if still active
                        val loadingJob = coroutineScope.launch {
                            prepareJob.join()
                            if (isActive) { navigationActions.navigateToBenchmark() }
                        }

                        pendingNavigation = {
                            prepareJob.cancel()
                            loadingJob.cancel()
                            navigationActions.navigateUp()
                        }
                    },
                    onConversationSelected = { systemPrompt, prepareJob ->
                        // Wait for preparation to complete, then navigate if still active
                        val loadingJob = coroutineScope.launch {
                            prepareJob.join()
                            if (isActive) { navigationActions.navigateToConversation() }
                        }

                        pendingNavigation = {
                            prepareJob.cancel()
                            loadingJob.cancel()
                            navigationActions.navigateUp()
                        }
                    },
                    onBackPressed = {
                        // Need to unload model before going back
                        handleBackWithModelCheck()
                    },
                )
            }

            // Benchmark Screen
            composable(AppDestinations.BENCHMARK_ROUTE) {
                BenchmarkScreen(
                    onBackPressed = {
                        // Need to unload model before going back
                        handleBackWithModelCheck()
                    }
                )
            }

            // Conversation Screen
            composable(AppDestinations.CONVERSATION_ROUTE) {
                ConversationScreen(
                    onBackPressed = {
                        // Need to unload model before going back
                        handleBackWithModelCheck()
                    },
                    viewModel = conversationViewModel
                )
            }

            // Settings General Screen
            composable(AppDestinations.SETTINGS_GENERAL_ROUTE) {
                SettingsGeneralScreen(
                    onBackPressed = { navigationActions.navigateUp() },
                    onMenuClicked = openDrawer
                )
            }

            // Models Management Screen
            composable(AppDestinations.MODELS_MANAGEMENT_ROUTE) {
                ModelsManagementScreen(
                    onBackPressed = { navigationActions.navigateUp() },
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
                    // Handle screen specific cleanups
                    when(engineState) {
                        is State.Benchmarking -> {}
                        is State.Generating -> conversationViewModel.clearConversation()
                        else -> {}
                    }

                    // Unload model
                    mainViewModel.unloadModel()
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
