package com.example.llama

import android.app.AlertDialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.ActionBarDrawerToggle
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.drawerlayout.widget.DrawerLayout
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.arm.aichat.InferenceEngine
import com.arm.aichat.internal.InferenceEngineImpl
import com.example.llama.data.AppDatabase
import com.example.llama.data.Conversation
import com.example.llama.data.DbMessage
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.UUID

class MainActivity : AppCompatActivity() {

    private lateinit var drawerLayout: DrawerLayout
    private lateinit var toolbar: Toolbar
    private lateinit var messagesRv: RecyclerView
    private lateinit var messageInput: EditText
    private lateinit var fab: FloatingActionButton
    private lateinit var conversationsRv: RecyclerView
    private lateinit var btnNewChat: Button

    private lateinit var messageAdapter: MessageAdapter
    private lateinit var conversationAdapter: ConversationAdapter
    private lateinit var db: AppDatabase
    private lateinit var engine: InferenceEngine

    private var currentConversationId: String = ""
    private var loadedModelPath: String? = null
    private var isGenerating = false
    private var generationJob: Job? = null

    private val currentMessages = mutableListOf<ChatMessage>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        db = AppDatabase.getInstance(this)
        engine = InferenceEngineImpl.getInstance(this)

        bindViews()
        setupToolbar()
        setupDrawer()
        setupMessageList()
        setupConversationList()
        setupFab()
        setupInput()
        observeConversations()

        lifecycleScope.launch {
            ensureActiveConversation()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        engine.destroy()
    }

    private fun bindViews() {
        drawerLayout    = findViewById(R.id.drawer_layout)
        toolbar         = findViewById(R.id.toolbar)
        messagesRv      = findViewById(R.id.messages)
        messageInput    = findViewById(R.id.message)
        fab             = findViewById(R.id.send)
        conversationsRv = findViewById(R.id.conversations_list)
        btnNewChat      = findViewById(R.id.btn_new_chat)
    }

    private fun setupToolbar() {
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
    }

    private fun setupDrawer() {
        val toggle = ActionBarDrawerToggle(
            this, drawerLayout, toolbar,
            R.string.drawer_open, R.string.drawer_close
        )
        drawerLayout.addDrawerListener(toggle)
        toggle.syncState()

        btnNewChat.setOnClickListener {
            lifecycleScope.launch {
                createNewConversation()
                drawerLayout.closeDrawers()
            }
        }
    }

    private fun setupMessageList() {
        messageAdapter = MessageAdapter { msg ->
            val clip = ClipData.newPlainText("mesaj", msg)
            (getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager)
                .setPrimaryClip(clip)
            Toast.makeText(this, "Panoya kopyalandı", Toast.LENGTH_SHORT).show()
        }
        messagesRv.layoutManager = LinearLayoutManager(this).also { it.stackFromEnd = true }
        messagesRv.adapter = messageAdapter
    }

    private fun setupConversationList() {
        conversationAdapter = ConversationAdapter(
            onSelect = { conv ->
                lifecycleScope.launch {
                    switchConversation(conv.id)
                    drawerLayout.closeDrawers()
                }
            },
            onDelete = { conv -> confirmDeleteConversation(conv) }
        )
        conversationsRv.layoutManager = LinearLayoutManager(this)
        conversationsRv.adapter = conversationAdapter
    }

    private fun setupFab() {
        updateFabIcon()
        fab.setOnClickListener {
            when {
                isGenerating -> stopGeneration()
                loadedModelPath == null -> showModelPickerDialog()
                else -> sendMessage()
            }
        }
    }

    private fun updateFabIcon() {
        val icon = when {
            isGenerating -> android.R.drawable.ic_media_pause
            loadedModelPath == null -> android.R.drawable.ic_menu_add
            else -> android.R.drawable.ic_menu_send
        }
        fab.setImageResource(icon)
    }

    private fun setupInput() {
        messageInput.setOnEditorActionListener { _, _, _ ->
            if (!isGenerating && loadedModelPath != null) sendMessage()
            true
        }
    }

    private fun observeConversations() {
        lifecycleScope.launch {
            db.chatDao().getAllConversations().collectLatest { list ->
                conversationAdapter.activeId = currentConversationId
                conversationAdapter.submitList(list)
            }
        }
    }

    private suspend fun ensureActiveConversation() {
        val prefs = getSharedPreferences("llama_prefs", MODE_PRIVATE)
        val savedId = prefs.getString("active_conversation_id", null)
        currentConversationId = if (savedId != null && conversationExists(savedId)) {
            savedId
        } else {
            createNewConversation()
        }
        loadMessagesForCurrent()
    }

    private suspend fun conversationExists(id: String): Boolean = withContext(Dispatchers.IO) {
        try { db.chatDao().getMessages(id).isNotEmpty() } catch (e: Exception) { false }
    }

    private suspend fun createNewConversation(): String = withContext(Dispatchers.IO) {
        val id = UUID.randomUUID().toString()
        db.chatDao().insertConversation(Conversation(id = id, title = "Yeni Sohbet"))
        withContext(Dispatchers.Main) {
            currentConversationId = id
            saveActiveId(id)
            currentMessages.clear()
            messageAdapter.submitList(emptyList())
            conversationAdapter.activeId = id
            conversationAdapter.notifyDataSetChanged()
            updateToolbarTitle("Yeni Sohbet")
        }
        id
    }

    private suspend fun switchConversation(id: String) {
        if (id == currentConversationId) return
        currentConversationId = id
        saveActiveId(id)
        loadMessagesForCurrent()
        withContext(Dispatchers.Main) {
            conversationAdapter.activeId = id
            conversationAdapter.notifyDataSetChanged()
        }
    }

    private suspend fun loadMessagesForCurrent() = withContext(Dispatchers.IO) {
        val dbMessages = db.chatDao().getMessages(currentConversationId)
        val chatMessages = dbMessages.map {
            ChatMessage(content = it.content, isUser = it.role == "user")
        }
        withContext(Dispatchers.Main) {
            currentMessages.clear()
            currentMessages.addAll(chatMessages)
            messageAdapter.submitList(currentMessages.toList())
            if (currentMessages.isNotEmpty())
                messagesRv.scrollToPosition(currentMessages.size - 1)
            val title = if (chatMessages.isNotEmpty())
                chatMessages.first().content.take(30) else "Yeni Sohbet"
            updateToolbarTitle(title)
        }
    }

    private fun saveActiveId(id: String) {
        getSharedPreferences("llama_prefs", MODE_PRIVATE)
            .edit().putString("active_conversation_id", id).apply()
    }

    private fun updateToolbarTitle(title: String) {
        supportActionBar?.title = title
    }

    private fun sendMessage() {
        val text = messageInput.text.toString().trim()
        if (text.isEmpty()) return
        messageInput.text.clear()

        val userMsg = ChatMessage(content = text, isUser = true)
        currentMessages.add(userMsg)
        messageAdapter.submitList(currentMessages.toList())
        messagesRv.scrollToPosition(currentMessages.size - 1)

        val convId = currentConversationId

        lifecycleScope.launch(Dispatchers.IO) {
            db.chatDao().insertMessage(
                DbMessage(UUID.randomUUID().toString(), convId, "user", text)
            )
            if (currentMessages.size == 1) {
                db.chatDao().updateConversationTitle(convId, text.take(40), System.currentTimeMillis())
            } else {
                db.chatDao().touchConversation(convId, System.currentTimeMillis())
            }
        }

        isGenerating = true
        updateFabIcon()
        var fullResponse = ""

        generationJob = lifecycleScope.launch {
            try {
                engine.sendUserPrompt(text)
                    .collect { token ->
                        fullResponse += token
                        messageAdapter.updateLastAssistantMessage(fullResponse)
                        messagesRv.scrollToPosition(currentMessages.size - 1)
                    }
            } catch (e: Exception) {
                messageAdapter.updateLastAssistantMessage(
                    if (fullResponse.isEmpty()) "[Hata: ${e.message}]" else fullResponse
                )
            } finally {
                if (currentMessages.isNotEmpty() && !currentMessages.last().isUser) {
                    currentMessages[currentMessages.size - 1] = ChatMessage(fullResponse, false)
                } else {
                    currentMessages.add(ChatMessage(fullResponse, false))
                }
                if (fullResponse.isNotEmpty()) {
                    lifecycleScope.launch(Dispatchers.IO) {
                        db.chatDao().insertMessage(
                            DbMessage(UUID.randomUUID().toString(), convId, "assistant", fullResponse)
                        )
                        db.chatDao().touchConversation(convId, System.currentTimeMillis())
                    }
                }
                isGenerating = false
                updateFabIcon()
            }
        }
    }

    private fun stopGeneration() {
        generationJob?.cancel()
        isGenerating = false
        updateFabIcon()
    }

    private fun confirmDeleteConversation(conv: Conversation) {
        AlertDialog.Builder(this)
            .setTitle("Sohbeti Sil")
            .setMessage("\"${conv.title}\" silinsin mi?")
            .setPositiveButton("Sil") { _, _ ->
                lifecycleScope.launch {
                    withContext(Dispatchers.IO) {
                        db.chatDao().deleteMessages(conv.id)
                        db.chatDao().deleteConversation(conv.id)
                    }
                    if (conv.id == currentConversationId) createNewConversation()
                }
            }
            .setNegativeButton("İptal", null)
            .show()
    }

    private fun showModelPickerDialog() {
        val prefs = getSharedPreferences("llama_prefs", MODE_PRIVATE)
        val savedModels = prefs.getStringSet("saved_models", mutableSetOf())!!.toMutableList()

        val options = savedModels.map { it.substringAfterLast("/") }.toMutableList()
        options.add("+ Yeni model ekle")

        AlertDialog.Builder(this)
            .setTitle("Model Seç")
            .setItems(options.toTypedArray()) { _, which ->
                if (which == options.size - 1) showAddModelDialog()
                else loadModel(savedModels[which])
            }
            .show()
    }

    private fun showAddModelDialog() {
        val input = EditText(this).apply { hint = "/storage/emulated/0/models/model.gguf" }
        AlertDialog.Builder(this)
            .setTitle("Model Yolu")
            .setView(input)
            .setPositiveButton("Yükle") { _, _ ->
                val path = input.text.toString().trim()
                if (path.isNotEmpty()) {
                    val prefs = getSharedPreferences("llama_prefs", MODE_PRIVATE)
                    val models = prefs.getStringSet("saved_models", mutableSetOf())!!.toMutableSet()
                    models.add(path)
                    prefs.edit().putStringSet("saved_models", models).apply()
                    loadModel(path)
                }
            }
            .setNegativeButton("İptal", null)
            .show()
    }

    private fun loadModel(path: String) {
        lifecycleScope.launch {
            try {
                Toast.makeText(this@MainActivity, "Model yükleniyor...", Toast.LENGTH_SHORT).show()
                engine.loadModel(path)
                loadedModelPath = path
                updateFabIcon()
                Toast.makeText(
                    this@MainActivity,
                    path.substringAfterLast("/") + " yüklendi",
                    Toast.LENGTH_SHORT
                ).show()
            } catch (e: Exception) {
                Toast.makeText(
                    this@MainActivity,
                    "Model yüklenemedi: ${e.message}",
                    Toast.LENGTH_LONG
                ).show()
            }
        }
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.main_menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_change_model -> { showModelPickerDialog(); true }
            R.id.action_clear_chat  -> { clearCurrentChat(); true }
            else -> super.onOptionsItemSelected(item)
        }
    }

    private fun clearCurrentChat() {
        lifecycleScope.launch {
            withContext(Dispatchers.IO) {
                db.chatDao().deleteMessages(currentConversationId)
                db.chatDao().updateConversationTitle(
                    currentConversationId, "Yeni Sohbet", System.currentTimeMillis()
                )
            }
            currentMessages.clear()
            messageAdapter.submitList(emptyList())
            updateToolbarTitle("Yeni Sohbet")
        }
    }
}
