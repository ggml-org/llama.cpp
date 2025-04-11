package com.example.llama.revamp.util

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.example.llama.revamp.data.preferences.UserPreferences
import com.example.llama.revamp.engine.InferenceEngine
import com.example.llama.revamp.monitoring.PerformanceMonitor
import com.example.llama.revamp.viewmodel.MainViewModel
import com.example.llama.revamp.viewmodel.PerformanceViewModel

/**
 * Utility class to provide ViewModel factories.
 *
 * TODO-han.yin: Replace with Hilt
 */
object ViewModelFactoryProvider {

    /**
     * Creates a factory for PerformanceViewModel.
     */
    fun getPerformanceViewModelFactory(
        performanceMonitor: PerformanceMonitor,
        userPreferences: UserPreferences
    ): ViewModelProvider.Factory {
        return object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>): T {
                if (modelClass.isAssignableFrom(PerformanceViewModel::class.java)) {
                    return PerformanceViewModel(performanceMonitor, userPreferences) as T
                }
                throw IllegalArgumentException("Unknown ViewModel class: ${modelClass.name}")
            }
        }
    }

    /**
     * Creates a factory for MainViewModel.
     */
    fun getMainViewModelFactory(
        inferenceEngine: InferenceEngine
    ): ViewModelProvider.Factory {
        return object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>): T {
                if (modelClass.isAssignableFrom(MainViewModel::class.java)) {
                    return MainViewModel(inferenceEngine) as T
                }
                throw IllegalArgumentException("Unknown ViewModel class: ${modelClass.name}")
            }
        }
    }
}
