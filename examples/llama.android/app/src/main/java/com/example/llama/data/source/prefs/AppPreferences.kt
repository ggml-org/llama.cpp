package com.example.llama.data.source.prefs

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.preferencesDataStore
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Manages internal preferences for the application.
 */
@Singleton
class AppPreferences @Inject constructor (
    @ApplicationContext private val context: Context
) {
    companion object {
        private const val DATASTORE_APP = "app"
        private val Context.appDataStore: DataStore<Preferences>
            by preferencesDataStore(name = DATASTORE_APP)

        // Preference keys
        private val USER_HAS_IMPORTED_FIRST_MODEL = booleanPreferencesKey("user_has_imported_first_model")
        private val USER_HAS_CHATTED_WITH_MODEL = booleanPreferencesKey("user_has_chatted_with_model")
        private val USER_HAS_NAVIGATED_TO_MANAGEMENT = booleanPreferencesKey("user_has_navigated_to_management")
    }

    /**
     * Gets whether the user has imported his first model
     */
    fun userHasImportedFirstModel(): Flow<Boolean> =
        context.appDataStore.data.map { preferences ->
            preferences[USER_HAS_IMPORTED_FIRST_MODEL] == true
        }

    /**
     * Sets whether the user has completed importing the first model.
     */
    suspend fun setUserHasImportedFirstModel(done: Boolean) = withContext(Dispatchers.IO) {
        context.appDataStore.edit { preferences ->
            preferences[USER_HAS_IMPORTED_FIRST_MODEL] = done
        }
    }

    /**
     * Gets whether the user has chatted with a model
     */
    fun userHasChattedWithModel(): Flow<Boolean> =
        context.appDataStore.data.map { preferences ->
            preferences[USER_HAS_CHATTED_WITH_MODEL] == true
        }

    /**
     * Sets whether the user has completed chatting with a model.
     */
    suspend fun setUserHasChattedWithModel(done: Boolean) = withContext(Dispatchers.IO) {
        context.appDataStore.edit { preferences ->
            preferences[USER_HAS_CHATTED_WITH_MODEL] = done
        }
    }

    /**
     * Gets whether the user has navigated to model management screen.
     */
    fun userHasNavigatedToManagement(): Flow<Boolean> =
        context.appDataStore.data.map { preferences ->
            preferences[USER_HAS_NAVIGATED_TO_MANAGEMENT] == true
        }

    /**
     * Sets whether the user has navigated to model management screen.
     */
    suspend fun setUserHasNavigatedToManagement(done: Boolean) = withContext(Dispatchers.IO) {
        context.appDataStore.edit { preferences ->
            preferences[USER_HAS_NAVIGATED_TO_MANAGEMENT] = done
        }
    }
}
