package com.example.llama.navigation

import androidx.navigation.NavController
import com.example.llama.engine.ModelLoadingMetrics

/**
 * Navigation destinations for the app
 */
object AppDestinations {
    // Primary navigation destinations
    const val MODEL_SELECTION_ROUTE = "model_selection"
    const val MODEL_LOADING_ROUTE = "model_loading"

    const val CONVERSATION_ROUTE = "conversation"
    const val CONVERSATION_ROUTE_WITH_PARAMS = "conversation/{modelLoadTimeMs}/{promptTimeMs}"

    const val BENCHMARK_ROUTE = "benchmark"
    const val BENCHMARK_ROUTE_WITH_PARAMS = "benchmark/{modelLoadTimeMs}"

    // Settings destinations
    const val SETTINGS_GENERAL_ROUTE = "settings_general"
    const val MODELS_MANAGEMENT_ROUTE = "models_management"
}

/**
 * Navigation actions to be performed in the app
 */
class NavigationActions(private val navController: NavController) {

    fun navigateToModelSelection() {
        navController.navigate(AppDestinations.MODEL_SELECTION_ROUTE) {
            // Clear back stack to start fresh
            popUpTo(AppDestinations.MODEL_SELECTION_ROUTE) { inclusive = true }
        }
    }

    fun navigateToModelLoading() {
        navController.navigate(AppDestinations.MODEL_LOADING_ROUTE)
    }

    fun navigateToConversation(metrics: ModelLoadingMetrics) {
        val route = AppDestinations.CONVERSATION_ROUTE
        val modelLoadTimeMs = metrics.modelLoadingTimeMs
        val promptTimeMs = metrics.systemPromptProcessingTimeMs ?: 0
        navController.navigate("$route/$modelLoadTimeMs/$promptTimeMs")
    }

    fun navigateToBenchmark(metrics: ModelLoadingMetrics) {
        val route = AppDestinations.BENCHMARK_ROUTE
        val modelLoadTimeMs = metrics.modelLoadingTimeMs
        navController.navigate("$route/$modelLoadTimeMs")
    }

    fun navigateToSettingsGeneral() {
        navController.navigate(AppDestinations.SETTINGS_GENERAL_ROUTE)
    }

    fun navigateToModelsManagement() {
        navController.navigate(AppDestinations.MODELS_MANAGEMENT_ROUTE)
    }

    fun navigateUp() {
        navController.navigateUp()
    }
}
