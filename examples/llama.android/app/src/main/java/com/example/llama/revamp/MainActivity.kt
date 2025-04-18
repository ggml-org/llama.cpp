package com.example.llama.revamp

import android.llama.cpp.InferenceEngine.State
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.OnBackPressedDispatcher
import androidx.activity.compose.BackHandler
import androidx.activity.compose.LocalOnBackPressedDispatcherOwner
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.SnackbarResult
import androidx.compose.material3.Surface
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.example.llama.revamp.navigation.AppDestinations
import com.example.llama.revamp.navigation.NavigationActions
import com.example.llama.revamp.ui.components.AnimatedNavHost
import com.example.llama.revamp.ui.components.AppNavigationDrawer
import com.example.llama.revamp.ui.components.AppScaffold
import com.example.llama.revamp.ui.components.BottomBarConfig
import com.example.llama.revamp.ui.components.NavigationIcon
import com.example.llama.revamp.ui.components.ScaffoldConfig
import com.example.llama.revamp.ui.components.ScaffoldEvent
import com.example.llama.revamp.ui.components.TopBarConfig
import com.example.llama.revamp.ui.screens.BenchmarkScreen
import com.example.llama.revamp.ui.screens.ConversationScreen
import com.example.llama.revamp.ui.screens.ModelLoadingScreen
import com.example.llama.revamp.ui.screens.ModelSelectionScreen
import com.example.llama.revamp.ui.screens.ModelsManagementScreen
import com.example.llama.revamp.ui.screens.SettingsGeneralScreen
import com.example.llama.revamp.ui.theme.LlamaTheme
import com.example.llama.revamp.viewmodel.BenchmarkViewModel
import com.example.llama.revamp.viewmodel.ConversationViewModel
import com.example.llama.revamp.viewmodel.MainViewModel
import com.example.llama.revamp.viewmodel.ModelLoadingViewModel
import com.example.llama.revamp.viewmodel.ModelsManagementViewModel
import com.example.llama.revamp.viewmodel.PerformanceViewModel
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
    performanceViewModel: PerformanceViewModel = hiltViewModel(),
    modelLoadingViewModel: ModelLoadingViewModel = hiltViewModel(),
    benchmarkViewModel: BenchmarkViewModel = hiltViewModel(),
    conversationViewModel: ConversationViewModel = hiltViewModel(),
    modelsManagementViewModel: ModelsManagementViewModel = hiltViewModel(),
) {
    val coroutineScope = rememberCoroutineScope()
    val snackbarHostState = remember { SnackbarHostState() }

    // Inference engine state
    val engineState by mainViewModel.engineState.collectAsState()

    // Metric states for scaffolds
    val memoryUsage by performanceViewModel.memoryUsage.collectAsState()
    val temperatureInfo by performanceViewModel.temperatureMetrics.collectAsState()
    val useFahrenheit by performanceViewModel.useFahrenheitUnit.collectAsState()
    val storageMetrics by performanceViewModel.storageMetrics.collectAsState()

    // Navigation
    val navController = rememberNavController()
    val navigationActions = remember(navController) { NavigationActions(navController) }
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute by remember(navBackStackEntry) {
        derivedStateOf { navBackStackEntry?.destination?.route ?: "" }
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

    // Create scaffold's top & bottom bar configs based on current route
    val scaffoldConfig = when (currentRoute) {
        // Model selection screen
        AppDestinations.MODEL_SELECTION_ROUTE ->
            ScaffoldConfig(
                topBarConfig = TopBarConfig.Default(
                    title = "Models",
                    navigationIcon = NavigationIcon.Menu(openDrawer)
                )
            )

        // Model loading screen
        AppDestinations.MODEL_LOADING_ROUTE ->
            ScaffoldConfig(
                topBarConfig = TopBarConfig.Performance(
                    title = "Load Model",
                    navigationIcon = NavigationIcon.Back { navigationActions.navigateUp() },
                    memoryMetrics = memoryUsage,
                    temperatureInfo = null
                )
            )

        // Benchmark screen
        AppDestinations.BENCHMARK_ROUTE ->
            ScaffoldConfig(
                topBarConfig = TopBarConfig.Performance(
                    title = "Benchmark",
                    navigationIcon = NavigationIcon.Back {
                        benchmarkViewModel.onBackPressed { navigationActions.navigateUp() }
                    },
                    memoryMetrics = memoryUsage,
                    temperatureInfo = Pair(temperatureInfo, useFahrenheit)
                )
            )

        // Conversation screen
        AppDestinations.CONVERSATION_ROUTE ->
            ScaffoldConfig(
                topBarConfig = TopBarConfig.Performance(
                    title = "Chat",
                    navigationIcon = NavigationIcon.Back {
                        // TODO-han.yin: uncomment after [ConversationViewModel] done
                     //    conversationViewModel.onBackPressed()
                     },
                    memoryMetrics = memoryUsage,
                    temperatureInfo = Pair(temperatureInfo, useFahrenheit)
                )
            )

        // Settings screen
        AppDestinations.SETTINGS_GENERAL_ROUTE ->
            ScaffoldConfig(
                topBarConfig = TopBarConfig.Default(
                    title = "Settings",
                    navigationIcon = NavigationIcon.Back { navigationActions.navigateUp() }
                )
            )

        // Storage management screen
        AppDestinations.MODELS_MANAGEMENT_ROUTE -> {
            // Collect the needed states
            val sortOrder by modelsManagementViewModel.sortOrder.collectAsState()
            val isMultiSelectionMode by modelsManagementViewModel.isMultiSelectionMode.collectAsState()
            val selectedModels by modelsManagementViewModel.selectedModels.collectAsState()
            val showSortMenu by modelsManagementViewModel.showSortMenu.collectAsState()
            val showImportModelMenu by modelsManagementViewModel.showImportModelMenu.collectAsState()

            // Create file launcher for importing local models
            val fileLauncher = rememberLauncherForActivityResult(
                contract = ActivityResultContracts.OpenDocument()
            ) { uri -> uri?.let { modelsManagementViewModel.localModelFileSelected(it) } }

            val bottomBarConfig = BottomBarConfig.ModelsManagement(
                sorting = BottomBarConfig.ModelsManagement.SortingConfig(
                    currentOrder = sortOrder,
                    isMenuVisible = showSortMenu,
                    toggleMenu = { modelsManagementViewModel.toggleSortMenu(it) },
                    selectOrder = {
                        modelsManagementViewModel.setSortOrder(it)
                        modelsManagementViewModel.toggleSortMenu(false)
                    }
                ),
                filtering = BottomBarConfig.ModelsManagement.FilteringConfig(
                    onClick = { /* TODO: implement filtering */ },
                ),
                selection = BottomBarConfig.ModelsManagement.SelectionConfig(
                    isActive = isMultiSelectionMode,
                    toggleMode = { modelsManagementViewModel.toggleSelectionMode(it) },
                    selectedModels = selectedModels,
                    toggleAllSelection = { modelsManagementViewModel.toggleAllSelection(it) },
                    deleteSelected = {
                        if (selectedModels.isNotEmpty()) {
                            modelsManagementViewModel.batchDeletionClicked(selectedModels)
                        }
                    },
                ),
                importing = BottomBarConfig.ModelsManagement.ImportConfig(
                    isMenuVisible = showImportModelMenu,
                    toggleMenu = { show -> modelsManagementViewModel.toggleImportMenu(show) },
                    importFromLocal = {
                        fileLauncher.launch(arrayOf("application/octet-stream", "*/*"))
                        modelsManagementViewModel.toggleImportMenu(false)
                    },
                    importFromHuggingFace = {
                        modelsManagementViewModel.importFromHuggingFace()
                        modelsManagementViewModel.toggleImportMenu(false)
                    }
                )
            )

            ScaffoldConfig(
                topBarConfig = TopBarConfig.Storage(
                    title = "Models Management",
                    navigationIcon = NavigationIcon.Back { navigationActions.navigateUp() },
                    storageMetrics = storageMetrics
                ),
                bottomBarConfig = bottomBarConfig
            )
        }

        // Fallback for empty screen or unknown routes
        else -> ScaffoldConfig(
            topBarConfig = TopBarConfig.Default(title = "", navigationIcon = NavigationIcon.None)
        )
    }

    // Handle child screens' scaffold events
    val handleScaffoldEvent: (ScaffoldEvent) -> Unit = { event ->
        when (event) {
            is ScaffoldEvent.ShowSnackbar -> {
                coroutineScope.launch {
                    if (event.actionLabel != null && event.onAction != null) {
                        val result = snackbarHostState.showSnackbar(
                            message = event.message,
                            actionLabel = event.actionLabel,
                            withDismissAction = event.withDismissAction,
                            duration = event.duration
                        )
                        if (result == SnackbarResult.ActionPerformed) {
                            event.onAction()
                        }
                    } else {
                        snackbarHostState.showSnackbar(
                            message = event.message,
                            withDismissAction = event.withDismissAction,
                            duration = event.duration
                        )
                    }
                }
            }
            is ScaffoldEvent.ChangeTitle -> {
                // TODO-han.yin: TBD
            }
        }
    }

    // Main UI hierarchy
    AppNavigationDrawer(
        drawerState = drawerState,
        navigationActions = navigationActions,
        gesturesEnabled = drawerGesturesEnabled,
        currentRoute = currentRoute
    ) {
        // The AppScaffold now uses the config we created
        AppScaffold(
            topBarconfig = scaffoldConfig.topBarConfig,
            bottomBarConfig = scaffoldConfig.bottomBarConfig,
            snackbarHostState = snackbarHostState,
        ) { paddingValues ->
            // AnimatedNavHost inside the scaffold content
            AnimatedNavHost(
                navController = navController,
                startDestination = AppDestinations.MODEL_SELECTION_ROUTE,
                modifier = Modifier.padding(paddingValues)
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
                    )
                }

                // Mode Selection Screen
                composable(AppDestinations.MODEL_LOADING_ROUTE) {
                    ModelLoadingScreen(
                        engineState = engineState,
                        onBenchmarkSelected = { prepareJob ->
                            // Wait for preparation to complete, then navigate if still active
                            coroutineScope.launch {
                                prepareJob.join()
                                if (isActive) { navigationActions.navigateToBenchmark() }
                            }
                        },
                        onConversationSelected = { systemPrompt, prepareJob ->
                            // Wait for preparation to complete, then navigate if still active
                            coroutineScope.launch {
                                prepareJob.join()
                                if (isActive) { navigationActions.navigateToConversation() }
                            }
                        },
                        viewModel = modelLoadingViewModel
                    )
                }

                // Benchmark Screen
                composable(AppDestinations.BENCHMARK_ROUTE) {
                    BenchmarkScreen(
                        onNavigateBack = { navigationActions.navigateUp() },
                        viewModel = benchmarkViewModel
                    )
                }

                // Conversation Screen
                composable(AppDestinations.CONVERSATION_ROUTE) {
                    ConversationScreen(
                        onNavigateBack = { navigationActions.navigateUp() },
                        viewModel = conversationViewModel
                    )
                }

                // Settings General Screen
                composable(AppDestinations.SETTINGS_GENERAL_ROUTE) {
                    SettingsGeneralScreen()
                }

                // Models Management Screen
                composable(AppDestinations.MODELS_MANAGEMENT_ROUTE) {
                    ModelsManagementScreen(
                        onScaffoldEvent = handleScaffoldEvent,
                        viewModel = modelsManagementViewModel
                    )
                }
            }
        }
    }
}
