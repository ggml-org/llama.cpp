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
import com.example.llama.ui.screens.ModelsScreen
import com.example.llama.ui.screens.SettingsGeneralScreen
import com.example.llama.ui.theme.LlamaTheme
import com.example.llama.viewmodel.BenchmarkViewModel
import com.example.llama.viewmodel.ConversationViewModel
import com.example.llama.viewmodel.MainViewModel
import com.example.llama.viewmodel.ModelLoadingViewModel
import com.example.llama.viewmodel.ModelScreenUiMode
import com.example.llama.viewmodel.ModelsManagementViewModel
import com.example.llama.viewmodel.ModelsViewModel
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
    modelsViewModel: ModelsViewModel = hiltViewModel(),
    modelsManagementViewModel: ModelsManagementViewModel = hiltViewModel(),
    modelLoadingViewModel: ModelLoadingViewModel = hiltViewModel(),
    benchmarkViewModel: BenchmarkViewModel = hiltViewModel(),
    conversationViewModel: ConversationViewModel = hiltViewModel(),
) {
    val coroutineScope = rememberCoroutineScope()
    val snackbarHostState = remember { SnackbarHostState() }

    // App core states
    val engineState by mainViewModel.engineState.collectAsState()
    val showModelImportTooltip by mainViewModel.showModelImportTooltip.collectAsState()
    val showChatTooltip by mainViewModel.showModelImportTooltip.collectAsState()

    // Model state
    val modelScreenUiMode by modelsViewModel.modelScreenUiMode.collectAsState()

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
        currentRoute == AppDestinations.MODELS_ROUTE -> {
            // Collect states for bottom bar
            val allModels by modelsViewModel.allModels.collectAsState()
            val filteredModels by modelsViewModel.filteredModels.collectAsState()
            val sortOrder by modelsViewModel.sortOrder.collectAsState()
            val showSortMenu by modelsViewModel.showSortMenu.collectAsState()
            val activeFilters by modelsViewModel.activeFilters.collectAsState()
            val showFilterMenu by modelsViewModel.showFilterMenu.collectAsState()
            val preselection by modelsViewModel.preselectedModelToRun.collectAsState()

            val selectedModelsToDelete by modelsManagementViewModel.selectedModelsToDelete.collectAsState()
            val showImportModelMenu by modelsManagementViewModel.showImportModelMenu.collectAsState()

            val hasModelsInstalled = allModels?.isNotEmpty() == true

            // Create file launcher for importing local models
            val fileLauncher = rememberLauncherForActivityResult(
                contract = ActivityResultContracts.OpenDocument()
            ) { uri -> uri?.let { modelsManagementViewModel.importLocalModelFileSelected(it) } }

            ScaffoldConfig(
                topBarConfig =
                    when (modelScreenUiMode) {
                        ModelScreenUiMode.BROWSING ->
                            TopBarConfig.ModelsBrowsing(
                                title = "Installed models",
                                navigationIcon = NavigationIcon.Menu {
                                    modelsViewModel.resetPreselection()
                                    openDrawer()
                                },
                                onToggleManaging = if (hasModelsInstalled) {
                                    { modelsViewModel.toggleMode(ModelScreenUiMode.MANAGING) }
                                } else null,
                            )
                        ModelScreenUiMode.SEARCHING ->
                            TopBarConfig.None()
                        ModelScreenUiMode.MANAGING ->
                            TopBarConfig.ModelsManagement(
                                title = "Managing models",
                                navigationIcon = NavigationIcon.Back {
                                    modelsViewModel.toggleMode(ModelScreenUiMode.BROWSING)
                                },
                                storageMetrics = if (isMonitoringEnabled) storageMetrics else null,
                            )
                        ModelScreenUiMode.DELETING ->
                            TopBarConfig.ModelsDeleting(
                                title = "Deleting models",
                                navigationIcon = NavigationIcon.Quit {
                                    modelsManagementViewModel.resetManagementState()
                                    modelsViewModel.toggleMode(ModelScreenUiMode.MANAGING)
                                },
                            )
                    },
                bottomBarConfig =
                    when (modelScreenUiMode) {
                        ModelScreenUiMode.BROWSING ->
                            BottomBarConfig.Models.Browsing(
                                isSearchingEnabled = hasModelsInstalled,
                                onToggleSearching = {
                                    modelsViewModel.toggleMode(ModelScreenUiMode.SEARCHING)
                                },
                                sorting = BottomBarConfig.Models.Browsing.SortingConfig(
                                    isEnabled = hasModelsInstalled,
                                    currentOrder = sortOrder,
                                    isMenuVisible = showSortMenu,
                                    toggleMenu = modelsViewModel::toggleSortMenu,
                                    selectOrder = {
                                        modelsViewModel.setSortOrder(it)
                                        modelsViewModel.toggleSortMenu(false)
                                    }
                                ),
                                filtering = BottomBarConfig.Models.Browsing.FilteringConfig(
                                    isEnabled = hasModelsInstalled,
                                    filters = activeFilters,
                                    onToggleFilter = modelsViewModel::toggleFilter,
                                    onClearFilters = modelsViewModel::clearFilters,
                                    isMenuVisible = showFilterMenu,
                                    toggleMenu = modelsViewModel::toggleFilterMenu
                                ),
                                runAction = BottomBarConfig.Models.RunActionConfig(
                                    preselectedModelToRun = preselection,
                                    onClickRun = {
                                        if (modelsViewModel.selectModel(it)) {
                                            modelsViewModel.resetPreselection()
                                            navigationActions.navigateToModelLoading()
                                        }
                                    }
                                ),
                            )

                        ModelScreenUiMode.SEARCHING ->
                            BottomBarConfig.Models.Searching(
                                textFieldState = modelsViewModel.searchFieldState,
                                onQuitSearching = {
                                    modelsViewModel.toggleMode(ModelScreenUiMode.BROWSING)
                                },
                                onSearch = { /* No-op for now */ },
                                runAction = BottomBarConfig.Models.RunActionConfig(
                                    preselectedModelToRun = preselection,
                                    onClickRun = {
                                        if (modelsViewModel.selectModel(it)) {
                                            modelsViewModel.resetPreselection()
                                            navigationActions.navigateToModelLoading()
                                        }
                                    }
                                ),
                            )

                        ModelScreenUiMode.MANAGING ->
                            BottomBarConfig.Models.Managing(
                                isDeletionEnabled = filteredModels?.isNotEmpty() == true,
                                onToggleDeleting = {
                                    modelsViewModel.toggleMode(ModelScreenUiMode.DELETING)
                                },
                                sorting = BottomBarConfig.Models.Managing.SortingConfig(
                                    isEnabled = hasModelsInstalled,
                                    currentOrder = sortOrder,
                                    isMenuVisible = showSortMenu,
                                    toggleMenu = { modelsViewModel.toggleSortMenu(it) },
                                    selectOrder = {
                                        modelsViewModel.setSortOrder(it)
                                        modelsViewModel.toggleSortMenu(false)
                                    }
                                ),
                                filtering = BottomBarConfig.Models.Managing.FilteringConfig(
                                    isEnabled = hasModelsInstalled,
                                    filters = activeFilters,
                                    onToggleFilter = modelsViewModel::toggleFilter,
                                    onClearFilters = modelsViewModel::clearFilters,
                                    isMenuVisible = showFilterMenu,
                                    toggleMenu = modelsViewModel::toggleFilterMenu
                                ),
                                importing = BottomBarConfig.Models.Managing.ImportConfig(
                                    showTooltip = showModelImportTooltip,
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
                                ),
                            )

                        ModelScreenUiMode.DELETING ->
                            BottomBarConfig.Models.Deleting(
                                onQuitDeleting = {
                                    modelsViewModel.toggleMode(ModelScreenUiMode.MANAGING)
                                },
                                selectedModels = selectedModelsToDelete,
                                selectAllFilteredModels = {
                                    filteredModels?.let {
                                        modelsManagementViewModel.selectModelsToDelete(it)
                                    }
                                },
                                clearAllSelectedModels = {
                                    modelsManagementViewModel.clearSelectedModelsToDelete()
                                },
                                deleteSelected = {
                                    selectedModelsToDelete.let {
                                        if (it.isNotEmpty()) {
                                            modelsManagementViewModel.batchDeletionClicked(it)
                                        }
                                    }
                                },
                            )
                    }
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
            val showShareFab by benchmarkViewModel.showShareFab.collectAsState()
            val showModelCard by benchmarkViewModel.showModelCard.collectAsState()

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
                    showShareFab = showShareFab,
                    onShare = { benchmarkViewModel.shareResult(handleScaffoldEvent) },
                    onRerun = { benchmarkViewModel.rerunBenchmark(handleScaffoldEvent) },
                    onClear = { benchmarkViewModel.clearResults(handleScaffoldEvent) },
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

            val showStubMessage = null  //            {
//                handleScaffoldEvent(ScaffoldEvent.ShowSnackbar(
//                    message = "Stub for now, let me know if you want it done :)"
//                ))
//            }

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
            onScaffoldEvent = handleScaffoldEvent,
            snackbarHostState = snackbarHostState,
        ) { paddingValues ->
            // AnimatedNavHost inside the scaffold content
            AnimatedNavHost(
                navController = navController,
                startDestination = AppDestinations.MODELS_ROUTE,
                modifier = Modifier.padding(paddingValues)
            ) {
                // Model Selection Screen
                composable(AppDestinations.MODELS_ROUTE) {
                    ModelsScreen(
                        onConfirmSelection = { modelInfo, ramWarning ->
                            if (modelsViewModel.confirmSelectedModel(modelInfo, ramWarning)) {
                                navigationActions.navigateToModelLoading()
                            }
                        },
                        onFirstModelImportSuccess =
                            if (showModelImportTooltip) {
                                { mainViewModel.waiveModelImportTooltip() }
                            } else null,
                        onScaffoldEvent = handleScaffoldEvent,
                        modelsViewModel = modelsViewModel,
                        managementViewModel = modelsManagementViewModel,
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
                        onScaffoldEvent = handleScaffoldEvent,
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
