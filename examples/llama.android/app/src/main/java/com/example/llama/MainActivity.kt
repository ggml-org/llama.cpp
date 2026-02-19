package com.example.llama

import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.activity.addCallback
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.arm.aichat.gguf.GgufMetadata
import com.arm.aichat.gguf.GgufMetadataReader
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.util.UUID

class MainActivity : AppCompatActivity() {

    private lateinit var statusTv: TextView
    private lateinit var messagesRv: RecyclerView
    private lateinit var userInputEt: EditText
    private lateinit var userActionFab: FloatingActionButton

    private lateinit var engine: InferenceEngine
    private var generationJob: Job? = null

    private var isModelReady = false
    private var isGenerating = false
    private var currentModelName: String? = null
    private val messages = mutableListOf<Message>()
    private val lastAssistantMsg = StringBuilder()
    private val messageAdapter = MessageAdapter(messages)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        onBackPressedDispatcher.addCallback { Log.w(TAG, "Ignore back press") }

        statusTv = findViewById(R.id.gguf)
        messagesRv = findViewById(R.id.messages)
        messagesRv.layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        messagesRv.adapter = messageAdapter
        userInputEt = findViewById(R.id.user_input)
        userActionFab = findViewById(R.id.fab)

        // Kaydedilmiş sohbeti yükle
        loadChatHistory()

        lifecycleScope.launch(Dispatchers.Default) {
            engine = AiChat.getInferenceEngine(applicationContext)
        }

        statusTv.text = "Model seçmek için butona bas"

        userActionFab.setOnClickListener {
            when {
                isGenerating -> {
                    // Üretimi durdur
                    generationJob?.cancel()
                }
                isModelReady -> {
                    handleUserInput()
                }
                else -> {
                    showModelPickerDialog()
                }
            }
        }
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menu.add(0, MENU_CHANGE_MODEL, 0, "Model değiştir")
        menu.add(0, MENU_CLEAR_CHAT, 1, "Sohbeti temizle")
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            MENU_CHANGE_MODEL -> {
                generationJob?.cancel()
                if (isModelReady) {
                    try { engine.cleanUp() } catch (e: Exception) { Log.e(TAG, "CleanUp error", e) }
                }
                isModelReady = false
                isGenerating = false
                currentModelName = null
                updateFabIcon()
                statusTv.text = "Model seçmek için butona bas"
                showModelPickerDialog()
                true
            }
            MENU_CLEAR_CHAT -> {
                AlertDialog.Builder(this)
                    .setTitle("Sohbeti temizle")
                    .setMessage("Tüm mesajlar silinecek. Emin misiniz?")
                    .setPositiveButton("Evet") { _, _ ->
                        messages.clear()
                        messageAdapter.notifyDataSetChanged()
                        saveChatHistory()
                    }
                    .setNegativeButton("İptal", null)
                    .show()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    private fun updateFabIcon() {
        when {
            isGenerating -> userActionFab.setImageResource(android.R.drawable.ic_media_pause)
            isModelReady -> userActionFab.setImageResource(R.drawable.outline_send_24)
            else -> userActionFab.setImageResource(android.R.drawable.ic_input_add)
        }
    }

    private fun showModelPickerDialog() {
        val savedModels = getSavedModels()
        if (savedModels.isEmpty()) {
            getContent.launch(arrayOf("*/*"))
            return
        }
        val options = savedModels.map { cleanModelName(it.name) }.toMutableList()
        options.add("+ Yeni model ekle")

        AlertDialog.Builder(this)
            .setTitle("Model seç")
            .setItems(options.toTypedArray()) { _, which ->
                if (which == savedModels.size) {
                    getContent.launch(arrayOf("*/*"))
                } else {
                    loadSavedModel(savedModels[which])
                }
            }
            .show()
    }

    private fun cleanModelName(filename: String): String {
        return filename
            .removeSuffix(FILE_EXTENSION_GGUF)
            .replace(Regex("-\\d{13}$"), "")  // timestamp sonunu kaldır
            .replace("-", " ")
            .replaceFirstChar { it.uppercase() }
    }

    private fun getSavedModels(): List<File> {
        return ensureModelsDirectory()
            .listFiles()
            ?.filter { it.isFile && it.name.endsWith(FILE_EXTENSION_GGUF) }
            ?.sortedByDescending { it.lastModified() }
            ?: emptyList()
    }

    private fun loadSavedModel(modelFile: File) {
        userActionFab.isEnabled = false
        userInputEt.hint = "Model yükleniyor..."
        val displayName = cleanModelName(modelFile.name)
        statusTv.text = "⏳ $displayName yükleniyor..."

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                loadModel(modelFile.name, modelFile)
                withContext(Dispatchers.Main) {
                    currentModelName = displayName
                    isModelReady = true
                    userInputEt.hint = "Mesajınızı yazın..."
                    userInputEt.isEnabled = true
                    userActionFab.isEnabled = true
                    updateFabIcon()
                    statusTv.text = "✓ $displayName"
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Model yüklenemedi: ${e.message}", Toast.LENGTH_LONG).show()
                    userActionFab.isEnabled = true
                    statusTv.text = "Hata! Tekrar dene"
                }
            }
        }
    }

    private val getContent = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        uri?.let { handleSelectedModel(it) }
    }

    private fun handleSelectedModel(uri: Uri) {
        userActionFab.isEnabled = false
        userInputEt.hint = "Dosya okunuyor..."
        statusTv.text = "⏳ Analiz ediliyor..."

        lifecycleScope.launch(Dispatchers.IO) {
            contentResolver.openInputStream(uri)?.use {
                GgufMetadataReader.create().readStructuredMetadata(it)
            }?.let { metadata ->
                val modelName = uri.lastPathSegment
    ?.substringAfterLast('/')
    ?.let { if (it.endsWith(FILE_EXTENSION_GGUF)) it else it + FILE_EXTENSION_GGUF }
    ?: (metadata.filename() + FILE_EXTENSION_GGUF
                val displayName = metadata.basic.nameLabel
                    ?: metadata.basic.name
                    ?: cleanModelName(modelName)

                contentResolver.openInputStream(uri)?.use { input ->
                    ensureModelFile(modelName, input)
                }?.let { modelFile ->
                    loadModel(modelName, modelFile)
                    withContext(Dispatchers.Main) {
                        currentModelName = displayName
                        isModelReady = true
                        userInputEt.hint = "Mesajınızı yazın..."
                        userInputEt.isEnabled = true
                        userActionFab.isEnabled = true
                        updateFabIcon()
                        statusTv.text = "✓ $displayName"
                    }
                }
            }
        }
    }

    private suspend fun ensureModelFile(modelName: String, input: InputStream) =
        withContext(Dispatchers.IO) {
            File(ensureModelsDirectory(), modelName).also { file ->
                if (!file.exists()) {
                    withContext(Dispatchers.Main) {
                        userInputEt.hint = "Kopyalanıyor..."
                        statusTv.text = "⏳ Kopyalanıyor..."
                    }
                    FileOutputStream(file).use { input.copyTo(it) }
                } else {
                    Log.i(TAG, "Model zaten mevcut: $modelName")
                }
            }
        }

    private suspend fun loadModel(modelName: String, modelFile: File) =
        withContext(Dispatchers.IO) {
            withContext(Dispatchers.Main) {
                userInputEt.hint = "Model yükleniyor..."
            }
            engine.loadModel(modelFile.path)
        }

    private fun handleUserInput() {
        val userMsg = userInputEt.text.toString()
        if (userMsg.isEmpty()) {
            Toast.makeText(this, "Mesaj boş!", Toast.LENGTH_SHORT).show()
            return
        }

        userInputEt.text = null
        userInputEt.isEnabled = false
        isGenerating = true
        updateFabIcon()

        messages.add(Message(UUID.randomUUID().toString(), userMsg, true))
        lastAssistantMsg.clear()
        messages.add(Message(UUID.randomUUID().toString(), "", false))
        messageAdapter.notifyItemInserted(messages.size - 1)
        messagesRv.scrollToPosition(messages.size - 1)

        generationJob = lifecycleScope.launch(Dispatchers.Default) {
            engine.sendUserPrompt(userMsg)
                .onCompletion {
                    withContext(Dispatchers.Main) {
                        isGenerating = false
                        userInputEt.isEnabled = true
                        updateFabIcon()
                        saveChatHistory()
                    }
                }.collect { token ->
                    withContext(Dispatchers.Main) {
                        val idx = messages.size - 1
                        if (idx >= 0 && !messages[idx].isUser) {
                            messages.removeAt(idx).copy(
                                content = lastAssistantMsg.append(token).toString()
                            ).let { messages.add(it) }
                            messageAdapter.notifyItemChanged(messages.size - 1)
                            messagesRv.scrollToPosition(messages.size - 1)
                        }
                    }
                }
        }
    }

    // Sohbet geçmişini SharedPreferences'a kaydet
    private fun saveChatHistory() {
        val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        val jsonArray = JSONArray()
        messages.forEach { msg ->
            JSONObject().apply {
                put("id", msg.id)
                put("content", msg.content)
                put("isUser", msg.isUser)
            }.let { jsonArray.put(it) }
        }
        prefs.edit().putString(KEY_CHAT_HISTORY, jsonArray.toString()).apply()
    }

    // Kaydedilmiş sohbeti yükle
    private fun loadChatHistory() {
        val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
        val json = prefs.getString(KEY_CHAT_HISTORY, null) ?: return
        try {
            val jsonArray = JSONArray(json)
            for (i in 0 until jsonArray.length()) {
                val obj = jsonArray.getJSONObject(i)
                messages.add(Message(
                    id = obj.getString("id"),
                    content = obj.getString("content"),
                    isUser = obj.getBoolean("isUser")
                ))
            }
            messageAdapter.notifyDataSetChanged()
            if (messages.isNotEmpty()) {
                messagesRv.scrollToPosition(messages.size - 1)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Sohbet geçmişi yüklenemedi", e)
        }
    }

    private fun ensureModelsDirectory() =
        File(filesDir, DIRECTORY_MODELS).also {
            if (it.exists() && !it.isDirectory) it.delete()
            if (!it.exists()) it.mkdir()
        }

    override fun onStop() {
        generationJob?.cancel()
        super.onStop()
    }

    override fun onDestroy() {
        engine.destroy()
        super.onDestroy()
    }

    companion object {
        private val TAG = MainActivity::class.java.simpleName
        private const val DIRECTORY_MODELS = "models"
        private const val FILE_EXTENSION_GGUF = ".gguf"
        private const val MENU_CHANGE_MODEL = 1
        private const val MENU_CLEAR_CHAT = 2
        private const val PREFS_NAME = "chat_prefs"
        private const val KEY_CHAT_HISTORY = "chat_history"
    }
}

fun GgufMetadata.filename() = when {
    basic.name != null -> {
        basic.name?.let { name ->
            basic.sizeLabel?.let { size -> "$name-$size" } ?: name
        }
    }
    architecture?.architecture != null -> {
        architecture?.architecture?.let { arch ->
            basic.uuid?.let { uuid -> "$arch-$uuid" } ?: "$arch-${System.currentTimeMillis()}"
        }
    }
    else -> "model-${System.currentTimeMillis().toHexString()}"
}
