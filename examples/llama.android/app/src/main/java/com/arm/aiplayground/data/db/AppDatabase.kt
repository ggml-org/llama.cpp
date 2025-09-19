package com.arm.aiplayground.data.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import com.arm.aiplayground.data.db.dao.ModelDao
import com.arm.aiplayground.data.db.dao.SystemPromptDao
import com.arm.aiplayground.data.db.entity.ModelEntity
import com.arm.aiplayground.data.db.entity.SystemPromptEntity
import javax.inject.Singleton

/**
 * Main database for the application.
 */
@Singleton
@Database(
    entities = [ModelEntity::class, SystemPromptEntity::class],
    version = 1,
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {

    abstract fun modelDao(): ModelDao

    abstract fun systemPromptDao(): SystemPromptDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "llama_app_database"
                )
                    .fallbackToDestructiveMigration(false)
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }
}
