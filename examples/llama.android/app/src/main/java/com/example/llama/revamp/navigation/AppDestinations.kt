package com.example.llama.revamp.navigation

import androidx.navigation.NavController

/**
 * Navigation destinations for the app
 */
object AppDestinations {
    const val MODEL_SELECTION_ROUTE = "model_selection"
    const val MODE_SELECTION_ROUTE = "mode_selection"
    const val CONVERSATION_ROUTE = "conversation"
    const val BENCHMARK_ROUTE = "benchmark"
    const val SETTINGS_ROUTE = "settings"
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

    fun navigateToModeSelection() {
        navController.navigate(AppDestinations.MODE_SELECTION_ROUTE)
    }

    fun navigateToConversation() {
        navController.navigate(AppDestinations.CONVERSATION_ROUTE)
    }

    fun navigateToBenchmark() {
        navController.navigate(AppDestinations.BENCHMARK_ROUTE)
    }

    fun navigateToSettings(tab: String = "GENERAL") {
        navController.navigate("${AppDestinations.SETTINGS_ROUTE}/$tab")
    }

    fun navigateUp() {
        navController.navigateUp()
    }
}
