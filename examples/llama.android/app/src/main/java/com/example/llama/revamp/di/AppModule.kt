package com.example.llama.revamp.di

import android.content.Context
import com.example.llama.revamp.data.preferences.UserPreferences
import com.example.llama.revamp.data.repository.SystemPromptRepository
import com.example.llama.revamp.engine.InferenceEngine
import com.example.llama.revamp.monitoring.PerformanceMonitor
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
object AppModule {
    @Provides
    @Singleton
    fun provideInferenceEngine() = InferenceEngine()

    @Provides
    @Singleton
    fun providePerformanceMonitor(@ApplicationContext context: Context) = PerformanceMonitor(context)

    @Provides
    @Singleton
    fun provideUserPreferences(@ApplicationContext context: Context) = UserPreferences(context)

    @Provides
    @Singleton
    fun provideSystemPromptRepository(@ApplicationContext context: Context) = SystemPromptRepository(context)
}
