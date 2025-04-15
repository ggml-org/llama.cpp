package com.example.llama.revamp.data.local

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface ModelDao {
    @Query("SELECT * FROM models ORDER BY dateAdded DESC")
    fun getAllModels(): Flow<List<ModelEntity>>  // Changed to Flow

    @Query("SELECT * FROM models WHERE id = :id")
    suspend fun getModelById(id: String): ModelEntity?

    @Query("SELECT * FROM models WHERE id IN (:ids)")
    suspend fun getModelsByIds(ids: List<String>): List<ModelEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertModel(model: ModelEntity)

    @Delete
    suspend fun deleteModel(model: ModelEntity)

    @Delete
    suspend fun deleteModels(models: List<ModelEntity>)

    @Query("UPDATE models SET lastUsed = :timestamp WHERE id = :id")
    suspend fun updateLastUsed(id: String, timestamp: Long)
}
