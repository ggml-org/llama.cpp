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
import com.example.llama.data.AppDatabase
import com.example.llama.data.Conversation
import com.example.llama.data.Message
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.UUID

// ── Mevcut llama.cpp bağlantısını koru ──────────────────────────────────────
// (Llm, InferenceEngine vb. importlar sende mevcut olduğu şekilde bırakılmalı)
// Aşağıdaki import'ları kendi projenize göre uyarlayın:
import com.arm.aichat.InferenceEngine          // veya mevcut import yolun
import com.arm.aichat.InferenceEngineConfig
// ────────────────────────────────────────────────────────────────────────────

class MainActivity : AppCompatActivity() {

    // ── UI ──────────────────────────────────────────────────────────────────
    private lateinit var drawerLayout: DrawerLayout
    private lateinit var toolbar: Toolbar
    private lateinit var messagesRv: RecyclerView
    private lateinit var messageInput: EditText
    private lateinit var fab: FloatingActionButton
    private lateinit var conversationsRv: RecyclerView
    private lateinit var btnNewChat: Button

    // ── Adapter'lar ─────────────────────────────────────────────────────────
    private lateinit var messageAdapter: MessageAdapter
    private lateinit var conversationAdapter: ConversationAdapter

    // ── DB ──────────────────────────────────────────────────────────────────
    private lateinit var db: AppDatabase

    // ── Durum ───────────────────────────────────────────────────────────────
    private var currentConversationId: String = ""
    private var loadedModelPath: String? = null
    private var isGenerating = false
    private var inferenceEngine: InferenceEngine? = null   // kendi tipine göre düzenle

    // Mesaj listesi (mevcut sohbet)
    private val currentMessages = mutableListOf<ChatMessage>()  // MessageAdapter'ın kullandığı tip

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        db = AppDatabase.getInstance(this)

        bindViews()
        setupToolbar()
        setupDrawer()
        setupMessageList()
        setupConversationList()
        setupFab()
        setupInput()
        observeConversations()

        // İlk açılışta aktif sohbeti yükle veya yeni oluştur
        lifecycleScope.launch {
            ensureActiveConversation()
        }
    }

    // ── View bağlama ────────────────────────────────────────────────────────
    private fun bindViews() {
        drawerLayout      = findViewById(R.id.drawer_layout)
        toolbar           = findViewById(R.id.toolbar)
        messagesRv        = findViewById(R.id.messages)
        messageInput      = findViewById(R.id.message)
        fab               = findViewById(R.id.send)
        conversationsRv   = findViewById(R.id.conversations_list)
        btnNewChat        = findViewById(R.id.btn_new_chat)
    }

    // ── Toolbar ─────────────────────────────────────────────────────────────
    private fun setupToolbar() {
        setSupportActionBar(toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
    }

    // ── Drawer ──────────────────────────────────────────────────────────────
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

    // ── Mesaj listesi ───────────────────────────────────────────────────────
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

    // ── Sohbet listesi (drawer) ──────────────────────────────────────────────
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

    // ── FAB (model yok / gönder / durdur) ───────────────────────────────────
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
            loadedModelPath == null -> android.R.drawable.ic_menu_add  // klasör yoksa bunu kullan
            else -> android.R.drawable.ic_menu_send
        }
        fab.setImageResource(icon)
    }

    // ── Input ────────────────────────────────────────────────────────────────
    private fun setupInput() {
        messageInput.setOnEditorActionListener { _, _, _ ->
            if (!isGenerating && loadedModelPath != null) sendMessage()
            true
        }
    }

    // ── DB: sohbet akışını gözlemle ──────────────────────────────────────────
    private fun observeConversations() {
        lifecycleScope.launch {
            db.chatDao().getAllConversations().collectLatest { list ->
                conversationAdapter.activeId = currentConversationId
                conversationAdapter.submitList(list)
            }
        }
    }

    // ── DB: aktif sohbeti yükle veya oluştur ────────────────────────────────
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
        // Basit kontrol: mesaj sayısına bakarak
        try {
            db.chatDao().getMessages(id).isNotEmpty() ||
            db.chatDao().conversationCount() > 0
        } catch (e: Exception) { false }
    }

    private suspend fun createNewConversation(): String = withContext(Dispatchers.IO) {
        val id = UUID.randomUUID().toString()
        val conv = Conversation(
            id = id,
            title = "Yeni Sohbet",
            createdAt = System.currentTimeMillis(),
            updatedAt = System.currentTimeMillis()
        )
        db.chatDao().insertConversation(conv)
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
            ChatMessage(role = it.role, content = it.content)  // MessageAdapter'ın beklediği tip
        }
        withContext(Dispatchers.Main) {
            currentMessages.clear()
            currentMessages.addAll(chatMessages)
            messageAdapter.submitList(currentMessages.toList())
            messagesRv.scrollToPosition(currentMessages.size - 1)
            val title = if (chatMessages.isNotEmpty()) chatMessages.first().content.take(30) else "Yeni Sohbet"
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

    // ── Mesaj gönder ─────────────────────────────────────────────────────────
    private fun sendMessage() {
        val text = messageInput.text.toString().trim()
        if (text.isEmpty()) return
        messageInput.text.clear()

        val userMsg = ChatMessage(role = "user", content = text)
        currentMessages.add(userMsg)
        messageAdapter.submitList(currentMessages.toList())
        messagesRv.scrollToPosition(currentMessages.size - 1)

        lifecycleScope.launch(Dispatchers.IO) {
            // DB'ye kaydet
            db.chatDao().insertMessage(
                Message(UUID.randomUUID().toString(), currentConversationId, "user", text)
            )
            // Sohbet başlığını güncelle (ilk kullanıcı mesajından)
            if (currentMessages.size == 1) {
                db.chatDao().updateConversationTitle(
                    currentConversationId,
                    text.take(40),
                    System.currentTimeMillis()
                )
            } else {
                db.chatDao().touchConversation(currentConversationId, System.currentTimeMillis())
            }
        }

        // ── Buradan sonrasını mevcut inference kodunla entegre et ────────────
        // inferenceEngine?.generate(currentMessages) { token -> ... } gibi
        // Cevap tamamlanınca DB'ye assistant mesajı kaydet:
        // db.chatDao().insertMessage(Message(..., "assistant", fullResponse))
        // ─────────────────────────────────────────────────────────────────────

        isGenerating = true
        updateFabIcon()
        // TODO: kendi llm.generate() çağrını buraya ekle
    }

    private fun stopGeneration() {
        // inferenceEngine?.stop()
        isGenerating = false
        updateFabIcon()
    }

    // ── Sohbet sil ───────────────────────────────────────────────────────────
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
                    if (conv.id == currentConversationId) {
                        createNewConversation()
                    }
                }
            }
            .setNegativeButton("İptal", null)
            .show()
    }

    // ── Model seçici ─────────────────────────────────────────────────────────
    private fun showModelPickerDialog() {
        // Mevcut model picker kodunu buraya taşı
        Toast.makeText(this, "Model seç", Toast.LENGTH_SHORT).show()
    }

    // ── 3 nokta menü ─────────────────────────────────────────────────────────
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
