package com.example.llama.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.llama.data.source.prefs.AppPreferences
import com.example.llama.engine.InferenceService
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
/**
 * Main ViewModel that expose the core states of [InferenceService] and [AppPreferences]
 */
class MainViewModel @Inject constructor (
    private val appPreferences: AppPreferences,
    private val inferenceService: InferenceService,
) : ViewModel() {

    val engineState = inferenceService.engineState

    // App preferences
    private val _showModelImportTooltip = MutableStateFlow(true)
    val showModelImportTooltip: StateFlow<Boolean> = _showModelImportTooltip.asStateFlow()

    private val _showChatTooltip = MutableStateFlow(true)
    val showChatTooltip: StateFlow<Boolean> = _showChatTooltip.asStateFlow()

    private val _showModelManagementTooltip = MutableStateFlow(true)
    val showModelManagementTooltip: StateFlow<Boolean> = _showModelManagementTooltip.asStateFlow()


    /**
     * Unload the current model and release the resources
     */
    suspend fun unloadModel() = inferenceService.unloadModel()

    init {
        viewModelScope.launch {
            launch {
                appPreferences.userHasImportedFirstModel().collect {
                    _showModelImportTooltip.value = !it
                }
            }
            launch {
                appPreferences.userHasChattedWithModel().collect {
                    _showChatTooltip.value = !it
                }
            }
            launch {
                appPreferences.userHasNavigatedToManagement().collect {
                    _showModelManagementTooltip.value = !it
                }
            }
        }
    }

    fun waiveChatTooltip() {
        viewModelScope.launch {
            appPreferences.setUserHasChattedWithModel(true)
        }
    }
    fun waiveModelImportTooltip() {
        viewModelScope.launch {
            appPreferences.setUserHasImportedFirstModel(true)
        }
    }

    fun waiveModelManagementTooltip() {
        viewModelScope.launch {
            appPreferences.setUserHasNavigatedToManagement(true)
        }
    }
}
