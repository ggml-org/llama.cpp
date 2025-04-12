package com.example.llama.revamp.navigation

import androidx.navigation.NavController

/**
 * Navigation destinations for the app
 */
object AppDestinations {
    // Primary navigation destinations
    const val MODEL_SELECTION_ROUTE = "model_selection"
    const val MODEL_LOADING_ROUTE = "model_loading"
    const val CONVERSATION_ROUTE = "conversation"
    const val BENCHMARK_ROUTE = "benchmark"

    // Settings destinations (moved from tabs to separate routes)
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

    fun navigateToConversation() {
        navController.navigate(AppDestinations.CONVERSATION_ROUTE)
    }

    fun navigateToBenchmark() {
        navController.navigate(AppDestinations.BENCHMARK_ROUTE)
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
