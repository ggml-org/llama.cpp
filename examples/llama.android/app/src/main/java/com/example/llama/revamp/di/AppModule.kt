package com.example.llama.revamp.di

import android.content.Context
import com.example.llama.revamp.data.local.AppDatabase
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
    fun provideAppDatabase(@ApplicationContext context: Context) = AppDatabase.getDatabase(context)

    @Provides
    fun providesSystemPromptDao(appDatabase: AppDatabase) = appDatabase.systemPromptDao()
}
