package com.example.llama.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "conversations")
data class Conversation(
    @PrimaryKey val id: String,
    val title: String,
    val createdAt: Long = System.currentTimeMillis(),
    val updatedAt: Long = System.currentTimeMillis()
)

@Entity(tableName = "messages")
data class Message(
    @PrimaryKey val id: String,
    val conversationId: String,
    val role: String,          // "user" | "assistant"
    val content: String,
    val timestamp: Long = System.currentTimeMillis()
)
