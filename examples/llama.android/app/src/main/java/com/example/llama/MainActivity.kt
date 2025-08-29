package com.example.llama

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.llama.cpp.InferenceEngine.State
import android.llama.cpp.isUninterruptible
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SnackbarDuration
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.SnackbarResult
import androidx.compose.material3.Surface
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavType
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.example.llama.engine.ModelLoadingMetrics
import com.example.llama.navigation.AppDestinations
import com.example.llama.navigation.NavigationActions
import com.example.llama.ui.scaffold.AnimatedNavHost
import com.example.llama.ui.scaffold.AppNavigationDrawer
import com.example.llama.ui.scaffold.AppScaffold
import com.example.llama.ui.scaffold.ScaffoldConfig
import com.example.llama.ui.scaffold.ScaffoldEvent
import com.example.llama.ui.scaffold.bottombar.BottomBarConfig
import com.example.llama.ui.scaffold.topbar.NavigationIcon
import com.example.llama.ui.scaffold.topbar.TopBarConfig
import com.example.llama.ui.screens.BenchmarkScreen
import com.example.llama.ui.screens.ConversationScreen
import com.example.llama.ui.screens.ModelLoadingScreen
import com.example.llama.ui.screens.ModelSelectionScreen
import com.example.llama.ui.screens.ModelsManagementScreen
import com.example.llama.ui.screens.SettingsGeneralScreen
import com.example.llama.ui.theme.LlamaTheme
import com.example.llama.viewmodel.BenchmarkViewModel
import com.example.llama.viewmodel.ConversationViewModel
import com.example.llama.viewmodel.MainViewModel
import com.example.llama.viewmodel.ModelLoadingViewModel
import com.example.llama.viewmodel.ModelSelectionViewModel
import com.example.llama.viewmodel.ModelsManagementViewModel
import com.example.llama.viewmodel.SettingsViewModel
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.launch

@AndroidEntryPoint
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            val settingsViewModel: SettingsViewModel = hiltViewModel()
            val themeMode by settingsViewModel.themeMode.collectAsState()

            LlamaTheme(themeMode) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    AppContent(settingsViewModel)
                }
            }
        }
    }
}

@Composable
fun AppContent(
    settingsViewModel: SettingsViewModel,
    mainViewModel: MainViewModel = hiltViewModel(),
    modelSelectionViewModel: ModelSelectionViewModel = hiltViewModel(),
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
    val isMonitoringEnabled by settingsViewModel.isMonitoringEnabled.collectAsState()
    val memoryUsage by settingsViewModel.memoryUsage.collectAsState()
    val temperatureInfo by settingsViewModel.temperatureMetrics.collectAsState()
    val useFahrenheit by settingsViewModel.useFahrenheitUnit.collectAsState()
    val storageMetrics by settingsViewModel.storageMetrics.collectAsState()

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
            if (drawerState.currentValue == DrawerValue.Open) true else false
        }
    }
    val openDrawer: () -> Unit = { coroutineScope.launch { drawerState.open() } }

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
            is ScaffoldEvent.ShareText -> {
                val shareIntent = Intent().apply {
                    action = Intent.ACTION_SEND
                    putExtra(Intent.EXTRA_TEXT, event.text)
                    event.title?.let { putExtra(Intent.EXTRA_SUBJECT, it) }
                    type = event.mimeType
                }

                val shareChooser = Intent.createChooser(shareIntent, event.title ?: "Share via")

                // Use the current activity for context
                val context = (navController.context as? Activity)
                    ?: throw IllegalStateException("Activity context required for sharing")

                try {
                    context.startActivity(shareChooser)
                } catch (_: ActivityNotFoundException) {
                    coroutineScope.launch {
                        snackbarHostState.showSnackbar(
                            message = "No app found to share content",
                            duration = SnackbarDuration.Short
                        )
                    }
                } catch (e: Exception) {
                    coroutineScope.launch {
                        snackbarHostState.showSnackbar(
                            message = "Share failed due to ${e.message}",
                            duration = SnackbarDuration.Short
                        )
                    }
                }
            }
        }
    }

    // Create scaffold's top & bottom bar configs based on current route
    val scaffoldConfig = when {
        // Model selection screen
        currentRoute == AppDestinations.MODEL_SELECTION_ROUTE -> {
            // Collect states for bottom bar
            val isSearchActive by modelSelectionViewModel.isSearchActive.collectAsState()
            val sortOrder by modelSelectionViewModel.sortOrder.collectAsState()
            val showSortMenu by modelSelectionViewModel.showSortMenu.collectAsState()
            val activeFilters by modelSelectionViewModel.activeFilters.collectAsState()
            val showFilterMenu by modelSelectionViewModel.showFilterMenu.collectAsState()
            val preselection by modelSelectionViewModel.preselection.collectAsState()

            ScaffoldConfig(
                topBarConfig =
                    if (isSearchActive) TopBarConfig.None()
                    else TopBarConfig.Default(
                        title = "Pick your model",
                        navigationIcon = NavigationIcon.Menu {
                            modelSelectionViewModel.resetPreselection()
                            openDrawer()
                        }
                    ),
                bottomBarConfig = BottomBarConfig.ModelSelection(
                    search = BottomBarConfig.ModelSelection.SearchConfig(
                        isActive = isSearchActive,
                        onToggleSearch = modelSelectionViewModel::toggleSearchState,
                        textFieldState = modelSelectionViewModel.searchFieldState,
                        onSearch = { /* No-op for now */ }
                    ),
                    sorting = BottomBarConfig.ModelSelection.SortingConfig(
                        currentOrder = sortOrder,
                        isMenuVisible = showSortMenu,
                        toggleMenu = modelSelectionViewModel::toggleSortMenu,
                        selectOrder = {
                            modelSelectionViewModel.setSortOrder(it)
                            modelSelectionViewModel.toggleSortMenu(false)
                        }
                    ),
                    filtering = BottomBarConfig.ModelSelection.FilteringConfig(
                        isActive = activeFilters.any { it.value },
                        filters = activeFilters,
                        onToggleFilter = modelSelectionViewModel::toggleFilter,
                        onClearFilters = modelSelectionViewModel::clearFilters,
                        isMenuVisible = showFilterMenu,
                        toggleMenu = modelSelectionViewModel::toggleFilterMenu
                    ),
                    runAction = BottomBarConfig.ModelSelection.RunActionConfig(
                        preselection = preselection,
                        onClickRun = {
                            if (modelSelectionViewModel.selectModel(it)) {
                                modelSelectionViewModel.toggleSearchState(false)
                                modelSelectionViewModel.resetPreselection()
                                navigationActions.navigateToModelLoading()
                            }
                        }
                    )
                )
            )
        }

        // Model loading screen
        currentRoute == AppDestinations.MODEL_LOADING_ROUTE ->
            ScaffoldConfig(
                topBarConfig = TopBarConfig.Performance(
                    title = "Select a mode",
                    navigationIcon = NavigationIcon.Back {
                        modelLoadingViewModel.onBackPressed { navigationActions.navigateUp() }
                    },
                    memoryMetrics = if (isMonitoringEnabled) memoryUsage else null,
                    temperatureInfo = null
                )
            )

        // Benchmark screen
        currentRoute.startsWith(AppDestinations.BENCHMARK_ROUTE) -> {
            val engineState by benchmarkViewModel.engineState.collectAsState()
            val showModelCard by benchmarkViewModel.showModelCard.collectAsState()
            val benchmarkResults by benchmarkViewModel.benchmarkResults.collectAsState()

            ScaffoldConfig(
                topBarConfig = TopBarConfig.Performance(
                    title = "Benchmark",
                    navigationIcon = NavigationIcon.Back {
                        benchmarkViewModel.onBackPressed { navigationActions.navigateUp() }
                    },
                    memoryMetrics = if (isMonitoringEnabled) memoryUsage else null,
                    temperatureInfo = if (isMonitoringEnabled) Pair(temperatureInfo, useFahrenheit) else null
                ),
                bottomBarConfig = BottomBarConfig.Benchmark(
                    engineIdle = !engineState.isUninterruptible,
                    onShare = {
                        benchmarkResults.lastOrNull()?.let {
                            handleScaffoldEvent(ScaffoldEvent.ShareText(it.text))
                        }
                    },
                    onRerun = {
                        if (engineState.isUninterruptible) {
                            handleScaffoldEvent(ScaffoldEvent.ShowSnackbar(
                                message = "Benchmark already in progress!\n" +
                                    "Please wait for the current run to complete."
                            ))
                        } else {
                            benchmarkViewModel.runBenchmark()
                        }
                    },
                    onClear = benchmarkViewModel::clearResults,
                    showModelCard = showModelCard,
                    onToggleModelCard = benchmarkViewModel::toggleModelCard,
                )
            )
        }

        // Conversation screen
        currentRoute.startsWith(AppDestinations.CONVERSATION_ROUTE) -> {
            val showModelCard by conversationViewModel.showModelCard.collectAsState()

            val modelThinkingOrSpeaking =
                engineState is State.ProcessingUserPrompt || engineState is State.Generating

            val showStubMessage = {
                handleScaffoldEvent(ScaffoldEvent.ShowSnackbar(
                    message = "Stub for now, let me know if you want it done :)"
                ))
            }

            ScaffoldConfig(
                topBarConfig = TopBarConfig.Performance(
                    title = "Chat",
                    navigationIcon = NavigationIcon.Back {
                        conversationViewModel.onBackPressed { navigationActions.navigateUp() }
                    },
                    memoryMetrics = if (isMonitoringEnabled) memoryUsage else null,
                    temperatureInfo = if (isMonitoringEnabled) Pair(temperatureInfo, useFahrenheit) else null,
                ),
                bottomBarConfig = BottomBarConfig.Conversation(
                    isEnabled = !modelThinkingOrSpeaking,
                    textFieldState = conversationViewModel.inputFieldState,
                    onSendClick = conversationViewModel::sendMessage,
                    showModelCard = showModelCard,
                    onToggleModelCard = conversationViewModel::toggleModelCard,
                    onAttachPhotoClick = showStubMessage,
                    onAttachFileClick = showStubMessage,
                    onAudioInputClick = showStubMessage,
                )
            )
        }

        // Settings screen
        currentRoute == AppDestinations.SETTINGS_GENERAL_ROUTE ->
            ScaffoldConfig(
                topBarConfig = TopBarConfig.Default(
                    title = "Settings",
                    navigationIcon = NavigationIcon.Back { navigationActions.navigateUp() }
                )
            )

        // Storage management screen
        currentRoute == AppDestinations.MODELS_MANAGEMENT_ROUTE -> {
            // Collect the needed states
            val sortOrder by modelsManagementViewModel.sortOrder.collectAsState()
            val isMultiSelectionMode by modelsManagementViewModel.isMultiSelectionMode.collectAsState()
            val selectedModels by modelsManagementViewModel.selectedModels.collectAsState()
            val showSortMenu by modelsManagementViewModel.showSortMenu.collectAsState()
            val activeFilters by modelsManagementViewModel.activeFilters.collectAsState()
            val showFilterMenu by modelsManagementViewModel.showFilterMenu.collectAsState()
            val showImportModelMenu by modelsManagementViewModel.showImportModelMenu.collectAsState()

            // Create file launcher for importing local models
            val fileLauncher = rememberLauncherForActivityResult(
                contract = ActivityResultContracts.OpenDocument()
            ) { uri -> uri?.let { modelsManagementViewModel.importLocalModelFileSelected(it) } }

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
                    isActive = activeFilters.any { it.value },
                    filters = activeFilters,
                    onToggleFilter = modelsManagementViewModel::toggleFilter,
                    onClearFilters = modelsManagementViewModel::clearFilters,
                    isMenuVisible = showFilterMenu,
                    toggleMenu = modelsManagementViewModel::toggleFilterMenu
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
                        modelsManagementViewModel.queryModelsFromHuggingFace()
                        modelsManagementViewModel.toggleImportMenu(false)
                    }
                )
            )

            ScaffoldConfig(
                topBarConfig = TopBarConfig.Storage(
                    title = "Models Management",
                    navigationIcon = NavigationIcon.Back { navigationActions.navigateUp() },
                    storageMetrics = if (isMonitoringEnabled) storageMetrics else null,
                ),
                bottomBarConfig = bottomBarConfig
            )
        }

        // Fallback for empty screen or unknown routes
        else -> ScaffoldConfig(
            topBarConfig = TopBarConfig.Default(title = "", navigationIcon = NavigationIcon.None)
        )
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
                        onManageModelsClicked = {
                            navigationActions.navigateToModelsManagement()
                        },
                        onConfirmSelection = { modelInfo, ramWarning ->
                            if (modelSelectionViewModel.confirmSelectedModel(modelInfo, ramWarning)) {
                                navigationActions.navigateToModelLoading()
                                modelSelectionViewModel.toggleSearchState(false)
                            }
                        },
                        viewModel = modelSelectionViewModel
                    )
                }

                // Mode Selection Screen
                composable(AppDestinations.MODEL_LOADING_ROUTE) {
                    ModelLoadingScreen(
                        onScaffoldEvent = handleScaffoldEvent,
                        onNavigateBack = { navigationActions.navigateUp() },
                        onNavigateToBenchmark = { navigationActions.navigateToBenchmark(it) },
                        onNavigateToConversation = { navigationActions.navigateToConversation(it) },
                        viewModel = modelLoadingViewModel
                    )
                }

                // Benchmark Screen
                composable(
                    route = AppDestinations.BENCHMARK_ROUTE_WITH_PARAMS,
                    arguments = listOf(
                        navArgument("modelLoadTimeMs") {
                            type = NavType.LongType
                            defaultValue = 0L
                        }
                    )
                ) { backStackEntry ->
                    val modelLoadTimeMs = backStackEntry.arguments?.getLong("modelLoadTimeMs") ?: 0L
                    val metrics = if (modelLoadTimeMs > 0) {
                        ModelLoadingMetrics(modelLoadTimeMs)
                    } else throw IllegalArgumentException("Expecting a valid ModelLoadingMetrics!")

                    BenchmarkScreen(
                        loadingMetrics = metrics,
                        onNavigateBack = { navigationActions.navigateUp() },
                        viewModel = benchmarkViewModel
                    )
                }

                // Conversation Screen
                composable(
                    route = AppDestinations.CONVERSATION_ROUTE_WITH_PARAMS,
                    arguments = listOf(
                        navArgument("modelLoadTimeMs") {
                            type = NavType.LongType
                            defaultValue = 0L
                        },
                        navArgument("promptTimeMs") {
                            type = NavType.LongType
                            defaultValue = 0L
                        }
                    )
                ) { backStackEntry ->
                    val modelLoadTimeMs = backStackEntry.arguments?.getLong("modelLoadTimeMs") ?: 0L
                    val promptTimeMs = backStackEntry.arguments?.getLong("promptTimeMs") ?: 0L
                    val metrics = if (modelLoadTimeMs > 0) {
                        ModelLoadingMetrics(
                            modelLoadingTimeMs = modelLoadTimeMs,
                            systemPromptProcessingTimeMs = if (promptTimeMs > 0) promptTimeMs else null
                        )
                    } else throw IllegalArgumentException("Expecting a valid ModelLoadingMetrics!")

                    ConversationScreen(
                        loadingMetrics = metrics,
                        onNavigateBack = { navigationActions.navigateUp() },
                        viewModel = conversationViewModel
                    )
                }

                // Models Management Screen
                composable(AppDestinations.MODELS_MANAGEMENT_ROUTE) {
                    ModelsManagementScreen(
                        onScaffoldEvent = handleScaffoldEvent,
                        viewModel = modelsManagementViewModel
                    )
                }

                // General Settings Screen
                composable(AppDestinations.SETTINGS_GENERAL_ROUTE) {
                    SettingsGeneralScreen(
                        viewModel = settingsViewModel
                    )
                }
            }
        }
    }
}
