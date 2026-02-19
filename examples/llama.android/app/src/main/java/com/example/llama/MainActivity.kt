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
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.util.UUID

class MainActivity : AppCompatActivity() {

    private lateinit var ggufTv: TextView
    private lateinit var messagesRv: RecyclerView
    private lateinit var userInputEt: EditText
    private lateinit var userActionFab: FloatingActionButton

    private lateinit var engine: InferenceEngine
    private var generationJob: Job? = null

    private var isModelReady = false
    private var currentModelName: String? = null
    private val messages = mutableListOf<Message>()
    private val lastAssistantMsg = StringBuilder()
    private val messageAdapter = MessageAdapter(messages)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        onBackPressedDispatcher.addCallback { Log.w(TAG, "Ignore back press for simplicity") }

        ggufTv = findViewById(R.id.gguf)
        messagesRv = findViewById(R.id.messages)
        messagesRv.layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        messagesRv.adapter = messageAdapter
        userInputEt = findViewById(R.id.user_input)
        userActionFab = findViewById(R.id.fab)

        lifecycleScope.launch(Dispatchers.Default) {
            engine = AiChat.getInferenceEngine(applicationContext)
        }

        userActionFab.setOnClickListener {
            if (isModelReady) {
                handleUserInput()
            } else {
                showModelPickerDialog()
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
                currentModelName = null
                userActionFab.setImageResource(android.R.drawable.ic_input_add)
                ggufTv.text = "Model seçmek için butona bas"
                showModelPickerDialog()
                true
            }
            MENU_CLEAR_CHAT -> {
                messages.clear()
                messageAdapter.notifyDataSetChanged()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    private fun showModelPickerDialog() {
        val savedModels = getSavedModels()

        if (savedModels.isEmpty()) {
            getContent.launch(arrayOf("*/*"))
            return
        }

        val options = savedModels.map { it.name }.toMutableList()
        options.add("+ Yeni model seç")

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
        ggufTv.text = "⏳ ${modelFile.name}"

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                loadModel(modelFile.name, modelFile)
                withContext(Dispatchers.Main) {
                    currentModelName = modelFile.name
                    isModelReady = true
                    userInputEt.hint = "Mesajınızı yazın..."
                    userInputEt.isEnabled = true
                    userActionFab.setImageResource(R.drawable.outline_send_24)
                    userActionFab.isEnabled = true
                    ggufTv.text = "✓ ${modelFile.name}"
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(
                        this@MainActivity,
                        "Model yüklenemedi: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                    userActionFab.isEnabled = true
                    ggufTv.text = "Hata! Model seçmek için butona bas"
                }
            }
        }
    }

    private val getContent = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        Log.i(TAG, "Selected file uri:\n $uri")
        uri?.let { handleSelectedModel(it) }
    }

    private fun handleSelectedModel(uri: Uri) {
        userActionFab.isEnabled = false
        userInputEt.hint = "GGUF okunuyor..."
        ggufTv.text = "⏳ Dosya analiz ediliyor..."

        lifecycleScope.launch(Dispatchers.IO) {
            contentResolver.openInputStream(uri)?.use {
                GgufMetadataReader.create().readStructuredMetadata(it)
            }?.let { metadata ->
                Log.i(TAG, "GGUF parsed: \n$metadata")
                withContext(Dispatchers.Main) {
                    ggufTv.text = metadata.toString()
                }

                val modelName = metadata.filename() + FILE_EXTENSION_GGUF
                contentResolver.openInputStream(uri)?.use { input ->
                    ensureModelFile(modelName, input)
                }?.let { modelFile ->
                    loadModel(modelName, modelFile)

                    withContext(Dispatchers.Main) {
                        currentModelName = modelName
                        isModelReady = true
                        userInputEt.hint = "Mesajınızı yazın..."
                        userInputEt.isEnabled = true
                        userActionFab.setImageResource(R.drawable.outline_send_24)
                        userActionFab.isEnabled = true
                        ggufTv.text = "✓ $modelName"
                    }
                }
            }
        }
    }

    private suspend fun ensureModelFile(modelName: String, input: InputStream) =
        withContext(Dispatchers.IO) {
            File(ensureModelsDirectory(), modelName).also { file ->
                if (!file.exists()) {
                    Log.i(TAG, "Copying file to $modelName")
                    withContext(Dispatchers.Main) {
                        userInputEt.hint = "Dosya kopyalanıyor..."
                        ggufTv.text = "⏳ Kopyalanıyor: $modelName"
                    }
                    FileOutputStream(file).use { input.copyTo(it) }
                    Log.i(TAG, "Done copying $modelName")
                } else {
                    Log.i(TAG, "File already exists $modelName")
                }
            }
        }

    private suspend fun loadModel(modelName: String, modelFile: File) =
        withContext(Dispatchers.IO) {
            Log.i(TAG, "Loading model $modelName")
            withContext(Dispatchers.Main) {
                userInputEt.hint = "Model yükleniyor..."
                ggufTv.text = "⏳ Yükleniyor: $modelName"
            }
            engine.loadModel(modelFile.path)
        }

    private fun handleUserInput() {
        userInputEt.text.toString().also { userMsg ->
            if (userMsg.isEmpty()) {
                Toast.makeText(this, "Mesaj boş!", Toast.LENGTH_SHORT).show()
            } else {
                userInputEt.text = null
                userInputEt.isEnabled = false
                userActionFab.isEnabled = false

                messages.add(Message(UUID.randomUUID().toString(), userMsg, true))
                lastAssistantMsg.clear()
                messages.add(Message(UUID.randomUUID().toString(), "", false))

                generationJob = lifecycleScope.launch(Dispatchers.Default) {
                    engine.sendUserPrompt(userMsg)
                        .onCompletion {
                            withContext(Dispatchers.Main) {
                                userInputEt.isEnabled = true
                                userActionFab.isEnabled = true
                            }
                        }.collect { token ->
                            withContext(Dispatchers.Main) {
                                val messageCount = messages.size
                                check(messageCount > 0 && !messages[messageCount - 1].isUser)
                                messages.removeAt(messageCount - 1).copy(
                                    content = lastAssistantMsg.append(token).toString()
                                ).let { messages.add(it) }
                                messageAdapter.notifyItemChanged(messages.size - 1)
                                messagesRv.scrollToPosition(messages.size - 1)
                            }
                        }
                }
            }
        }
    }

    private fun ensureModelsDirectory() =
        File(filesDir, DIRECTORY_MODELS).also {
            if (it.exists() && !it.isDirectory) { it.delete() }
            if (!it.exists()) { it.mkdir() }
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
        private const val BENCH_PROMPT_PROCESSING_TOKENS = 512
        private const val BENCH_TOKEN_GENERATION_TOKENS = 128
        private const val BENCH_SEQUENCE = 1
        private const val BENCH_REPETITION = 3
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
