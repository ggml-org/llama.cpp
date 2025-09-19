package com.arm.aiplayground.navigation

import androidx.navigation.NavController
import com.arm.aiplayground.engine.ModelLoadingMetrics

/**
 * Navigation destinations for the app
 */
object AppDestinations {
    // Primary navigation destinations
    const val MODELS_ROUTE = "models"
    const val MODEL_LOADING_ROUTE = "model_loading"

    const val CONVERSATION_ROUTE = "conversation"
    const val CONVERSATION_ROUTE_WITH_PARAMS = "conversation/{modelLoadTimeMs}/{promptTimeMs}"

    const val BENCHMARK_ROUTE = "benchmark"
    const val BENCHMARK_ROUTE_WITH_PARAMS = "benchmark/{modelLoadTimeMs}"

    // Settings destinations
    const val SETTINGS_GENERAL_ROUTE = "settings_general"
}

/**
 * Navigation actions to be performed in the app
 */
class NavigationActions(private val navController: NavController) {

    fun navigateToModelSelection() {
        navController.navigate(AppDestinations.MODELS_ROUTE) {
            // Clear back stack to start fresh
            popUpTo(AppDestinations.MODELS_ROUTE) { inclusive = true }
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

    fun navigateUp() {
        navController.navigateUp()
    }
}
