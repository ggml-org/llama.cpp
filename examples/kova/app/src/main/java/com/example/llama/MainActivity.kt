package com.example.llama

import android.app.Activity
import android.app.AlertDialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
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
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.UUID
import org.json.JSONArray
import org.json.JSONObject
import android.net.Uri
import android.content.ComponentName
import android.content.ServiceConnection
import android.os.IBinder
import android.content.pm.PackageManager
import javax.crypto.Cipher
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.PBEKeySpec
import javax.crypto.spec.SecretKeySpec
import java.security.SecureRandom
// v10 - externalCacheDir ile geçici model kopyalama

class MainActivity : AppCompatActivity() {

    companion object {
        private val logBuffer = ArrayDeque<String>(200)
        var loggingEnabled: Boolean = false

        fun log(tag: String, msg: String) {
            if (!loggingEnabled) return
            val entry = "${java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.getDefault()).format(java.util.Date())} [$tag] $msg"
            android.util.Log.d(tag, msg)
            synchronized(logBuffer) {
                if (logBuffer.size >= 200) logBuffer.removeFirst()
                logBuffer.addLast(entry)
            }
        }

        fun getLogs(): String = synchronized(logBuffer) { logBuffer.joinToString("\n") }
        fun clearLogs() = synchronized(logBuffer) { logBuffer.clear() }

        // Model kaydı: "uri:content://..." veya "/path/to/model.gguf"
        // URI ile kaydedilenler harici, path ile kaydedilenler dahili
        fun isUriEntry(entry: String) = entry.startsWith("uri:")
        fun entryToUri(entry: String): Uri = Uri.parse(entry.removePrefix("uri:"))
        fun entryDisplayName(entry: String): String {
            return if (isUriEntry(entry)) {
                entry.removePrefix("uri:").substringAfterLast("%2F").substringAfterLast("/")
                    .let { if (it.isBlank()) entry.substringAfterLast("/") else it }
            } else {
                entry.substringAfterLast("/")
            }
        }
    }

    private lateinit var drawerLayout: DrawerLayout
    private var selectedTemplate: Int = 0
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
    private var loadedModelPath: String? = null  // savedModels'daki entry string'i
    private var isGenerating = false
    private var generationJob: Job? = null

    // Akıllı kaydırma
    private var autoScroll = true

    // Ayarlar
    private var contextSize: Int = 2048
    private var systemPrompt: String = ""
    private var temperature: Float = 0.8f
    private var topP: Float = 0.95f
    private var topK: Int = 40
    private var noThinking: Boolean = false
    private var autoLoadLastModel: Boolean = false
    private var flashAttn: Boolean = false

    private val currentMessages = mutableListOf<ChatMessage>()

    // Foreground service
    private var generationService: KovaForegroundService? = null
    private var isAppInForeground = true
    private var tokenUpdateCounter = 0

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, service: IBinder) {
            val binder = service as KovaForegroundService.LocalBinder
            generationService = binder.getService()
        }
        override fun onServiceDisconnected(name: ComponentName) {
            generationService = null
        }
    }

    private val backupSaveLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            result.data?.data?.let { uri ->
                pendingBackupUri = uri
                pendingBackupCallback?.invoke(uri)
                pendingBackupCallback = null
            }
        }
    }

    private val backupRestoreLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            result.data?.data?.let { uri -> handleRestoreFile(uri) }
        }
    }

    private var pendingBackupUri: Uri? = null
    private var pendingBackupCallback: ((Uri) -> Unit)? = null

    /**
     * Dosya seçici — kalıcı URI izni alır, modeli KOPYALAMAZ.
     * savedModels'a "uri:content://..." formatında kaydeder.
     */
    private val filePickerLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            result.data?.data?.let { uri ->
                // Kalıcı okuma izni al — uygulama yeniden başlasa bile geçerli
                try {
                    contentResolver.takePersistableUriPermission(
                        uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
                    )
                } catch (e: Exception) {
                    log("Kova", "Kalıcı URI izni alınamadı: ${e.message}")
                }

                val entry = "uri:$uri"
                val prefs = getSharedPreferences("llama_prefs", MODE_PRIVATE)
                val models = prefs.getStringSet("saved_models", mutableSetOf())!!.toMutableSet()
                models.add(entry)
                prefs.edit().putStringSet("saved_models", models).apply()

                log("Kova", "Model URI ile eklendi: $entry")
                showTemplatePickerDialog(entry)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        db = AppDatabase.getInstance(this)
        engine = InferenceEngineImpl.getInstance(this)

        loadSettings()
        cleanupMissingModels()
        bindViews()
        setupToolbar()
        setupDrawer()
        setupMessageList()
        setupConversationList()
        setupFab()
        setupInput()
        observeConversations()

        lifecycleScope.launch { ensureActiveConversation() }

        // Otomatik son model yükleme
        if (autoLoadLastModel) {
            val prefs = getSharedPreferences("llama_prefs", MODE_PRIVATE)
            val lastEntry = prefs.getString("last_loaded_model", null)
            if (lastEntry != null) {
                val savedModels = prefs.getStringSet("saved_models", mutableSetOf())!!
                if (savedModels.contains(lastEntry)) {
                    val modelKey = "template_${entryDisplayName(lastEntry)}"
                    selectedTemplate = prefs.getInt(modelKey, 0)
                    loadModel(lastEntry)
                }
            }
        }

        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            if (checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(arrayOf(android.Manifest.permission.POST_NOTIFICATIONS), 100)
            }
        }
    }

    override fun onResume() { super.onResume(); isAppInForeground = true }
    override fun onPause() { super.onPause(); isAppInForeground = false }

    override fun onDestroy() {
        super.onDestroy()
        deleteTempModelFile()  // Geçici model kopyasını sil
        engine.destroy()
        try { unbindService(serviceConnection) } catch (_: Exception) {}
    }

    private fun loadSettings() {
        val prefs = getSharedPreferences("llama_prefs", MODE_PRIVATE)
        contextSize  = prefs.getInt("context_size", 2048)
        systemPrompt = prefs.getString("system_prompt", "") ?: ""
        temperature  = prefs.getFloat("temperature", 0.8f)
        topP         = prefs.getFloat("top_p", 0.95f)
        topK         = prefs.getInt("top_k", 40)
        noThinking        = prefs.getBoolean("no_thinking", false)
        autoLoadLastModel = prefs.getBoolean("auto_load_last_model", false)
        flashAttn         = prefs.getBoolean("flash_attn", false)
    }

    private fun saveSettings() {
        getSharedPreferences("llama_prefs", MODE_PRIVATE).edit()
            .putInt("context_size", contextSize)
            .putString("system_prompt", systemPrompt)
            .putFloat("temperature", temperature)
            .putFloat("top_p", topP)
            .putInt("top_k", topK)
            .putBoolean("no_thinking", noThinking)
            .putBoolean("auto_load_last_model", autoLoadLastModel)
            .putBoolean("flash_attn", flashAttn)
            .apply()
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
        val toggle = ActionBarDrawerToggle(this, drawerLayout, toolbar, R.string.drawer_open, R.string.drawer_close)
        drawerLayout.addDrawerListener(toggle)
        toggle.syncState()
        btnNewChat.setOnClickListener {
            lifecycleScope.launch { createNewConversation(); drawerLayout.closeDrawers() }
        }
    }

    private fun setupMessageList() {
        messageAdapter = MessageAdapter(
            onCopy = { msg ->
                val clip = ClipData.newPlainText("mesaj", msg)
                (getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager).setPrimaryClip(clip)
                Toast.makeText(this, "Panoya kopyalandı", Toast.LENGTH_SHORT).show()
            },
            onEdit = { position, content -> showEditMessageDialog(position, content) },
            onRegenerate = { _ -> regenerateLastResponse() }
        )
        messagesRv.layoutManager = LinearLayoutManager(this).also { it.stackFromEnd = true }
        messagesRv.adapter = messageAdapter

        messagesRv.addOnScrollListener(object : RecyclerView.OnScrollListener() {
            override fun onScrolled(recyclerView: RecyclerView, dx: Int, dy: Int) {
                if (dy < 0) {
                    autoScroll = false
                } else if (!recyclerView.canScrollVertically(1)) {
                    autoScroll = true
                }
            }
        })
    }

    private fun setupConversationList() {
        conversationAdapter = ConversationAdapter(
            onSelect = { conv ->
                lifecycleScope.launch { switchConversation(conv.id); drawerLayout.closeDrawers() }
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
        fab.setImageResource(when {
            isGenerating -> android.R.drawable.ic_media_pause
            loadedModelPath == null -> android.R.drawable.ic_menu_add
            else -> android.R.drawable.ic_menu_send
        })
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
        currentConversationId = if (savedId != null && conversationExists(savedId)) savedId
                                else createNewConversation()
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
        val chatMessages = dbMessages.map { ChatMessage(content = it.content, isUser = it.role == "user") }
        withContext(Dispatchers.Main) {
            currentMessages.clear()
            currentMessages.addAll(chatMessages)
            messageAdapter.submitList(currentMessages.toList())
            if (currentMessages.isNotEmpty()) messagesRv.scrollToPosition(currentMessages.size - 1)
            updateToolbarTitle(if (chatMessages.isNotEmpty()) chatMessages.first().content.take(30) else "Yeni Sohbet")
        }
    }

    private fun saveActiveId(id: String) {
        getSharedPreferences("llama_prefs", MODE_PRIVATE).edit().putString("active_conversation_id", id).apply()
    }

    private fun updateToolbarTitle(title: String) { supportActionBar?.title = title }

    private fun updateActiveModelSubtitle() {
        val name = loadedModelPath?.let { entryDisplayName(it) } ?: "Model yüklü değil"
        supportActionBar?.subtitle = name
    }

    private fun sendMessage() {
        val text = messageInput.text.toString().trim()
        if (text.isEmpty()) return
        messageInput.text.clear()

        val userMsg = ChatMessage(content = text, isUser = true)
        currentMessages.add(userMsg)
        messageAdapter.submitList(currentMessages.toList())
        autoScroll = true
        messagesRv.scrollToPosition(currentMessages.size - 1)

        val convId = currentConversationId
        lifecycleScope.launch(Dispatchers.IO) {
            db.chatDao().insertMessage(DbMessage(UUID.randomUUID().toString(), convId, "user", text))
            if (currentMessages.size == 1) {
                db.chatDao().updateConversationTitle(convId, text.take(40), System.currentTimeMillis())
            } else {
                db.chatDao().touchConversation(convId, System.currentTimeMillis())
            }
        }
        sendMessageContent(currentMessages.toList())
    }

    private fun stopGeneration() {
        generationJob?.cancel()
        generationJob = null
        isGenerating = false
        updateFabIcon()
        generationService?.onGenerationCancelled()
        generationService = null
        try { unbindService(serviceConnection) } catch (_: Exception) {}
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
            .setNegativeButton("İptal", null).show()
    }

    // ── Model listesi ─────────────────────────────────────────────────────────────

    private fun showModelPickerDialog() {
        val prefs = getSharedPreferences("llama_prefs", MODE_PRIVATE)
        val savedModels = prefs.getStringSet("saved_models", mutableSetOf())!!.toMutableList()
        val options = savedModels.map { entry ->
            val name = entryDisplayName(entry)
            if (entry == loadedModelPath) "✓ $name" else name
        }.toMutableList()
        options.add("+ Yeni model ekle")
        AlertDialog.Builder(this).setTitle("Model Seç")
            .setItems(options.toTypedArray()) { _, which ->
                if (which == options.size - 1) showAddModelDialog()
                else showModelActionDialog(savedModels[which])
            }.show()
    }

    private fun showModelActionDialog(entry: String) {
        AlertDialog.Builder(this).setTitle(entryDisplayName(entry))
            .setItems(arrayOf("Yükle", "Kaldır")) { _, which ->
                when (which) { 0 -> showTemplatePickerDialog(entry); 1 -> confirmRemoveModel(entry) }
            }.setNegativeButton("İptal", null).show()
    }

    private fun confirmRemoveModel(entry: String) {
        val name = entryDisplayName(entry)
        AlertDialog.Builder(this).setTitle("Modeli Kaldır")
            .setMessage("\"$name\" listeden kaldırılsın mı?")
            .setPositiveButton("Kaldır") { _, _ ->
                if (loadedModelPath == entry) {
                    Toast.makeText(this, "Önce başka bir model yükleyin.", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                val prefs = getSharedPreferences("llama_prefs", MODE_PRIVATE)
                val models = prefs.getStringSet("saved_models", mutableSetOf())!!.toMutableSet()
                models.remove(entry)
                prefs.edit().putStringSet("saved_models", models).apply()

                // Dahili kopyaysa fiziksel olarak sil; URI ise sadece listeden çıkar
                if (!isUriEntry(entry)) {
                    val file = java.io.File(entry)
                    if (file.exists() && file.absolutePath.startsWith(filesDir.absolutePath)) file.delete()
                }
                Toast.makeText(this, "\"$name\" kaldırıldı", Toast.LENGTH_SHORT).show()
            }.setNegativeButton("İptal", null).show()
    }

    private fun showTemplatePickerDialog(entry: String) {
        val templates = arrayOf("Otomatik (GGUF'tan)", "Aya / Command-R", "ChatML", "Gemma", "Llama 3")
        val prefs = getSharedPreferences("llama_prefs", MODE_PRIVATE)
        val modelKey = "template_${entryDisplayName(entry)}"
        val savedTemplate = prefs.getInt(modelKey, 0)
        AlertDialog.Builder(this).setTitle("Sohbet Şablonu Seçin")
            .setSingleChoiceItems(templates, savedTemplate) { dialog, which ->
                selectedTemplate = which
                prefs.edit().putInt(modelKey, which).apply()
                dialog.dismiss()
                loadModel(entry)
            }.setNegativeButton("İptal", null).show()
    }

    /** Dosya seçici — kalıcı URI izni ile SD kartta istenen konumdan model seç */
    private fun showAddModelDialog() {
        filePickerLauncher.launch(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
            // Kalıcı izin için flag
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
        })
    }

    // ── Model yükleme ────────────────────────────────────────────────────────────

    // Şu an yüklü geçici kopya (uygulama kapanınca silinecek)
    private var activeTempModelFile: java.io.File? = null

    private fun deleteTempModelFile() {
        activeTempModelFile?.let { f ->
            if (f.exists()) {
                f.delete()
                log("Kova", "Geçici model silindi: ${f.name}")
            }
        }
        activeTempModelFile = null
    }

    /**
     * Modeli yükle.
     * entry = "uri:content://..." → externalCacheDir'e kopyala → yükle → kapanınca sil
     * entry = "/path/..."        → doğrudan yol (kopyalama yok)
     */
    private fun loadModel(entry: String) {
        lifecycleScope.launch {
            var progressDialog: android.app.ProgressDialog? = null
            try {
                val modelName = entryDisplayName(entry)

                if (engine.state.value is InferenceEngine.State.ModelReady ||
                    engine.state.value is InferenceEngine.State.Error) {
                    engine.cleanUp()
                }
                var waited = 0
                while (engine.state.value !is InferenceEngine.State.Initialized && waited < 100) {
                    delay(100); waited++
                }

                (engine as? com.arm.aichat.internal.InferenceEngineImpl)
                    ?.applySettings(contextSize, temperature, topP, topK, flashAttn)

                val pathToLoad: String

                if (isUriEntry(entry)) {
                    val uri = entryToUri(entry)

                    // URI erişimini başlamadan önce doğrula
                    val pfd = try {
                        contentResolver.openFileDescriptor(uri, "r")
                    } catch (e: Exception) { null }

                    if (pfd == null) {
                        AlertDialog.Builder(this@MainActivity)
                            .setTitle("⚠️ Model Erişilemiyor")
                            .setMessage("\"$modelName\" dosyasına erişim izni yok veya dosya taşınmış.\n\nBu genellikle yedekten geri yüklenen modellerde olur. Modeli listeden kaldırıp tekrar ekleyin.")
                            .setPositiveButton("Listeden Kaldır") { _, _ ->
                                val prefs = getSharedPreferences("llama_prefs", MODE_PRIVATE)
                                val models = prefs.getStringSet("saved_models", mutableSetOf())!!.toMutableSet()
                                models.remove(entry)
                                prefs.edit().putStringSet("saved_models", models).apply()
                                Toast.makeText(this@MainActivity, "\"$modelName\" listeden kaldırıldı. Tekrar ekleyebilirsiniz.", Toast.LENGTH_LONG).show()
                            }
                            .setNegativeButton("İptal", null).show()
                        return@launch
                    }

                    val fd = pfd.fd
                    log("Kova", "fd yöntemi deneniyor: fd=$fd model=$modelName")

                    // fd ile doğrudan yüklemeyi dene
                    var fdSuccess = false
                    try {
                        (engine as? com.arm.aichat.internal.InferenceEngineImpl)
                            ?.loadModelFromFd(fd, modelName)
                        fdSuccess = true
                        loadedModelPath = entry
                        log("Kova", "fd ile yükleme başarılı")
                    } catch (e: Exception) {
                        log("Kova", "fd yöntemi başarısız (${e.message}), kopyalama yöntemine geçiliyor...")
                    } finally {
                        pfd.close()
                    }

                    if (fdSuccess) {
                        // fd ile yüklendi, sistem promptu ve UI güncelle
                        if (systemPrompt.isNotEmpty() && selectedTemplate == 0) {
                            try { engine.setSystemPrompt(systemPrompt) } catch (_: Exception) {}
                        }
                        log("Kova", "Model yüklendi (fd): $modelName template=$selectedTemplate flashAttn=$flashAttn")
                        updateFabIcon()
                        updateActiveModelSubtitle()
                        getSharedPreferences("llama_prefs", MODE_PRIVATE).edit()
                            .putString("last_loaded_model", entry).apply()
                        Toast.makeText(this@MainActivity, "$modelName yüklendi", Toast.LENGTH_SHORT).show()
                        return@launch
                    }

                    // fd başarısız — kopyalama yöntemine geç
                    val cacheDir = externalCacheDir ?: cacheDir
                    val tempFile = java.io.File(cacheDir, "model_active_$modelName")

                    val docFile = androidx.documentfile.provider.DocumentFile
                        .fromSingleUri(this@MainActivity, uri)
                    val originalSize = docFile?.length() ?: 0L

                    if (tempFile.exists() && originalSize > 0 && tempFile.length() == originalSize) {
                        log("Kova", "Model zaten önbellekte, kopyalama atlandı: ${tempFile.name}")
                        activeTempModelFile = tempFile
                        pathToLoad = tempFile.absolutePath
                    } else {
                        deleteTempModelFile()

                        progressDialog = android.app.ProgressDialog(this@MainActivity).apply {
                            setTitle("Model hazırlanıyor")
                            setMessage("$modelName kopyalanıyor...")
                            isIndeterminate = false
                            max = 100
                            setProgressStyle(android.app.ProgressDialog.STYLE_HORIZONTAL)
                            setCancelable(false)
                            show()
                        }

                        log("Kova", "Kopyalama başlıyor: $modelName")

                        withContext(Dispatchers.IO) {
                            contentResolver.openInputStream(uri)?.use { input ->
                                var copiedBytes = 0L
                                tempFile.outputStream().use { output ->
                                    val buf = ByteArray(8 * 1024 * 1024)
                                    var n: Int
                                    while (input.read(buf).also { n = it } != -1) {
                                        output.write(buf, 0, n)
                                        copiedBytes += n
                                        if (originalSize > 0) {
                                            val progress = (copiedBytes * 100 / originalSize).toInt()
                                            withContext(Dispatchers.Main) {
                                                progressDialog?.progress = progress
                                            }
                                        }
                                    }
                                }
                            } ?: throw Exception("Dosya açılamadı: $uri")
                        }

                        progressDialog?.dismiss()
                        progressDialog = null
                        activeTempModelFile = tempFile
                        pathToLoad = tempFile.absolutePath
                        log("Kova", "Kopyalama tamamlandı: ${tempFile.length()} bytes")
                    }

                } else {
                    pathToLoad = entry
                }

                Toast.makeText(this@MainActivity, "Model yükleniyor...", Toast.LENGTH_SHORT).show()
                engine.loadModel(pathToLoad)
                loadedModelPath = entry

                if (systemPrompt.isNotEmpty() && selectedTemplate == 0) {
                    try {
                        engine.setSystemPrompt(systemPrompt)
                        log("Kova", "Sistem promptu gönderildi")
                    } catch (e: Exception) {
                        log("Kova", "setSystemPrompt hatası: ${e.message}")
                    }
                }

                log("Kova", "Model yüklendi: $modelName template=$selectedTemplate flashAttn=$flashAttn")
                updateFabIcon()
                updateActiveModelSubtitle()
                // Son yüklü modeli kaydet (otomatik yükleme için)
                getSharedPreferences("llama_prefs", MODE_PRIVATE).edit()
                    .putString("last_loaded_model", entry).apply()
                Toast.makeText(this@MainActivity, "$modelName yüklendi", Toast.LENGTH_SHORT).show()

            } catch (e: Exception) {
                progressDialog?.dismiss()
                progressDialog = null
                deleteTempModelFile()
                Toast.makeText(this@MainActivity, "Model yüklenemedi: ${e.message}", Toast.LENGTH_LONG).show()
                log("Kova", "Model yükleme hatası: ${e.message}")
            }
        }
    }

    /**
     * Uygulama başlangıcında mevcut olmayan dahili modelleri temizle.
     * URI tabanlı modellere dokunma — dosya hâlâ orada olabilir.
     */
    private fun cleanupMissingModels() {
        val prefs = getSharedPreferences("llama_prefs", MODE_PRIVATE)
        val models = prefs.getStringSet("saved_models", mutableSetOf())!!.toMutableSet()
        val valid = models.filter { entry ->
            if (isUriEntry(entry)) true  // URI'leri kontrol etme, izin geçerli olabilir
            else java.io.File(entry).exists()
        }.toMutableSet()
        if (valid.size != models.size) prefs.edit().putStringSet("saved_models", valid).apply()
    }

    // ── Menü ─────────────────────────────────────────────────────────────────────

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.main_menu, menu); return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_change_model -> { showModelPickerDialog(); true }
            R.id.action_clear_chat  -> { clearCurrentChat(); true }
            R.id.action_settings    -> { showSettingsDialog(); true }
            R.id.action_backup      -> { backupChats(); true }
            R.id.action_restore     -> { showRestorePicker(); true }
            R.id.action_logs        -> { showLogsDialog(); true }
            else -> super.onOptionsItemSelected(item)
        }
    }

    private fun clearCurrentChat() {
        lifecycleScope.launch {
            withContext(Dispatchers.IO) {
                db.chatDao().deleteMessages(currentConversationId)
                db.chatDao().updateConversationTitle(currentConversationId, "Yeni Sohbet", System.currentTimeMillis())
            }
            currentMessages.clear()
            messageAdapter.submitList(emptyList())
            updateToolbarTitle("Yeni Sohbet")
        }
    }

    // ─── AYARLAR DİALOGU ────────────────────────────────────────────────────────

    private fun showSettingsDialog() {
        val ctx = this
        val dp = resources.displayMetrics.density
        val scrollView = ScrollView(ctx)
        val layout = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            setPadding((16*dp).toInt(), (16*dp).toInt(), (16*dp).toInt(), (16*dp).toInt())
        }
        scrollView.addView(layout)

        fun sectionTitle(text: String) = TextView(ctx).apply {
            this.text = text; textSize = 14f
            setTypeface(null, android.graphics.Typeface.BOLD)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = (12*dp).toInt(); bottomMargin = (4*dp).toInt() }
        }

        layout.addView(sectionTitle("Context Window (token)"))
        val ctxGroup = RadioGroup(ctx).apply { orientation = RadioGroup.HORIZONTAL }
        val ctxOptions = listOf(2048, 4096, 8192)
        val ctxRadios = ctxOptions.map { size ->
            RadioButton(ctx).apply { text = size.toString(); id = View.generateViewId(); isChecked = (size == contextSize) }
        }
        ctxRadios.forEach { ctxGroup.addView(it) }
        layout.addView(ctxGroup)

        layout.addView(sectionTitle("Sistem Prompt"))
        val systemPromptInput = EditText(ctx).apply {
            hint = "Örn: Sen yardımcı bir asistansın."
            setText(systemPrompt); minLines = 3; maxLines = 6; gravity = android.view.Gravity.TOP
        }
        layout.addView(systemPromptInput)

        layout.addView(sectionTitle("Temperature: %.2f".format(temperature)))
        val tempLabel = layout.getChildAt(layout.childCount - 1) as TextView
        val tempBar = SeekBar(ctx).apply {
            max = 200; progress = (temperature * 100).toInt()
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar, p: Int, fromUser: Boolean) { tempLabel.text = "Temperature: %.2f".format(p / 100f) }
                override fun onStartTrackingTouch(sb: SeekBar) {}
                override fun onStopTrackingTouch(sb: SeekBar) {}
            })
        }
        layout.addView(tempBar)

        layout.addView(sectionTitle("Top-P: %.2f".format(topP)))
        val topPLabel = layout.getChildAt(layout.childCount - 1) as TextView
        val topPBar = SeekBar(ctx).apply {
            max = 100; progress = (topP * 100).toInt()
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar, p: Int, fromUser: Boolean) { topPLabel.text = "Top-P: %.2f".format(p / 100f) }
                override fun onStartTrackingTouch(sb: SeekBar) {}
                override fun onStopTrackingTouch(sb: SeekBar) {}
            })
        }
        layout.addView(topPBar)

        layout.addView(sectionTitle("Top-K: $topK"))
        val topKLabel = layout.getChildAt(layout.childCount - 1) as TextView
        val topKBar = SeekBar(ctx).apply {
            max = 200; progress = topK
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar, p: Int, fromUser: Boolean) { topKLabel.text = "Top-K: ${maxOf(1, p)}" }
                override fun onStartTrackingTouch(sb: SeekBar) {}
                override fun onStopTrackingTouch(sb: SeekBar) {}
            })
        }
        layout.addView(topKBar)

        layout.addView(sectionTitle("Qwen3 Ayarı"))
        val noThinkingRow = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = android.view.Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
                .apply { bottomMargin = (4*dp).toInt() }
        }
        val noThinkingLabel = TextView(ctx).apply {
            text = "💭 Düşünme modunu kapat (/no_think)"; textSize = 13f
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        @Suppress("DEPRECATION")
        val noThinkingSwitch = Switch(ctx).apply { isChecked = noThinking }
        noThinkingRow.addView(noThinkingLabel); noThinkingRow.addView(noThinkingSwitch)
        layout.addView(noThinkingRow)
        layout.addView(TextView(ctx).apply {
            text = "Gereksiz <think> bloklarını önler. Sadece Qwen3 için."; textSize = 11f; alpha = 0.6f
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
                .apply { bottomMargin = (8*dp).toInt() }
        })

        layout.addView(sectionTitle("Model Yükleme"))
        val autoLoadRow = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = android.view.Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
                .apply { bottomMargin = (4*dp).toInt() }
        }
        val autoLoadLabel = TextView(ctx).apply {
            text = "🚀 Son modeli otomatik yükle"; textSize = 13f
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        @Suppress("DEPRECATION")
        val autoLoadSwitch = Switch(ctx).apply { isChecked = autoLoadLastModel }
        autoLoadRow.addView(autoLoadLabel); autoLoadRow.addView(autoLoadSwitch)
        layout.addView(autoLoadRow)
        layout.addView(TextView(ctx).apply {
            text = "Uygulama açılınca son yüklü model otomatik hazırlanır."; textSize = 11f; alpha = 0.6f
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
                .apply { bottomMargin = (8*dp).toInt() }
        })

        layout.addView(sectionTitle("Performans"))
        val flashAttnRow = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = android.view.Gravity.CENTER_VERTICAL
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
                .apply { bottomMargin = (4*dp).toInt() }
        }
        val flashAttnLabel = TextView(ctx).apply {
            text = "⚡ Flash Attention"; textSize = 13f
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        @Suppress("DEPRECATION")
        val flashAttnSwitch = Switch(ctx).apply { isChecked = flashAttn }
        flashAttnRow.addView(flashAttnLabel); flashAttnRow.addView(flashAttnSwitch)
        layout.addView(flashAttnRow)
        layout.addView(TextView(ctx).apply {
            text = "Bazı modellerde üretim hızını artırır. Değişiklik sonraki model yüklemede etkinleşir."; textSize = 11f; alpha = 0.6f
            layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
                .apply { bottomMargin = (8*dp).toInt() }
        })

        AlertDialog.Builder(this).setTitle("⚙️ Ayarlar").setView(scrollView)
            .setPositiveButton("Kaydet") { _, _ ->
                val checkedId = ctxGroup.checkedRadioButtonId
                if (checkedId != -1) {
                    val idx = ctxRadios.indexOfFirst { it.id == checkedId }
                    if (idx >= 0) contextSize = ctxOptions[idx]
                }
                systemPrompt = systemPromptInput.text.toString().trim()
                temperature  = tempBar.progress / 100f
                topP         = topPBar.progress / 100f
                topK         = maxOf(1, topKBar.progress)
                noThinking   = noThinkingSwitch.isChecked
                autoLoadLastModel = autoLoadSwitch.isChecked
                flashAttn    = flashAttnSwitch.isChecked
                saveSettings()
                Toast.makeText(this, "Ayarlar kaydedildi", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("İptal", null).show()
    }

    // ── Düzenleme / Yeniden Oluştur ──────────────────────────────────────────────

    private fun showEditMessageDialog(position: Int, currentContent: String) {
        if (isGenerating) { Toast.makeText(this, "Yanıt üretilirken düzenleme yapılamaz", Toast.LENGTH_SHORT).show(); return }
        val input = android.widget.EditText(this).apply {
            setText(currentContent); setSelection(currentContent.length); setPadding(48, 24, 48, 24)
        }
        android.app.AlertDialog.Builder(this).setTitle("Mesajı Düzenle").setView(input)
            .setPositiveButton("Gönder") { _, _ ->
                val newText = input.text.toString().trim()
                if (newText.isNotEmpty() && newText != currentContent) editAndResend(position, newText)
            }.setNegativeButton("İptal", null).show()
    }

    private fun editAndResend(position: Int, newContent: String) {
        val convId = currentConversationId
        while (currentMessages.size > position) currentMessages.removeAt(currentMessages.size - 1)
        currentMessages.add(ChatMessage(content = newContent, isUser = true))
        messageAdapter.submitList(currentMessages.toList())
        autoScroll = true
        lifecycleScope.launch(Dispatchers.IO) {
            db.chatDao().deleteMessages(convId)
            currentMessages.forEachIndexed { idx, msg ->
                db.chatDao().insertMessage(com.example.llama.data.DbMessage(
                    id = java.util.UUID.randomUUID().toString(), conversationId = convId,
                    role = if (msg.isUser) "user" else "assistant", content = msg.content,
                    timestamp = System.currentTimeMillis() + idx
                ))
            }
        }
        sendMessageContent(currentMessages.toList())
    }

    private fun regenerateLastResponse() {
        if (isGenerating) { Toast.makeText(this, "Yanıt üretilirken yeniden oluşturulamaz", Toast.LENGTH_SHORT).show(); return }
        if (loadedModelPath == null) { Toast.makeText(this, "Önce bir model yükleyin", Toast.LENGTH_SHORT).show(); return }
        if (currentMessages.isNotEmpty() && !currentMessages.last().isUser) currentMessages.removeAt(currentMessages.size - 1)
        if (currentMessages.isEmpty() || !currentMessages.last().isUser) return
        messageAdapter.submitList(currentMessages.toList())
        autoScroll = true
        val convId = currentConversationId
        lifecycleScope.launch(Dispatchers.IO) {
            val dbMessages = db.chatDao().getMessages(convId)
            val lastAssistant = dbMessages.lastOrNull { it.role == "assistant" }
            lastAssistant?.let { msg ->
                db.chatDao().deleteMessages(convId)
                dbMessages.filter { it.id != msg.id }.forEach { db.chatDao().insertMessage(it) }
            }
        }
        sendMessageContent(currentMessages.toList())
    }

    // ── Prompt oluşturma ─────────────────────────────────────────────────────────

    private fun buildFormattedPrompt(messages: List<ChatMessage>): String {
        val sp = systemPrompt

        if (selectedTemplate == 0) {
            val lastUserText = messages.lastOrNull { it.isUser }?.content ?: return ""
            return if (noThinking) "/no_think\n\n$lastUserText" else lastUserText
        }

        val sb = StringBuilder()
        when (selectedTemplate) {
            1 -> {
                sb.append("<BOS_TOKEN>")
                if (sp.isNotEmpty())
                    sb.append("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>$sp<|END_OF_TURN_TOKEN|>")
                for (msg in messages) {
                    if (msg.isUser) {
                        val txt = if (noThinking) "/no_think\n\n${msg.content}" else msg.content
                        sb.append("<|START_OF_TURN_TOKEN|><|USER_TOKEN|>$txt<|END_OF_TURN_TOKEN|>")
                    } else {
                        sb.append("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>${msg.content}<|END_OF_TURN_TOKEN|>")
                    }
                }
                sb.append("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")
            }
            2 -> {
                if (sp.isNotEmpty()) sb.append("<|im_start|>system\n$sp<|im_end|>\n")
                for (msg in messages) {
                    if (msg.isUser) {
                        val txt = if (noThinking) "/no_think\n\n${msg.content}" else msg.content
                        sb.append("<|im_start|>user\n$txt<|im_end|>\n")
                    } else {
                        sb.append("<|im_start|>assistant\n${msg.content}<|im_end|>\n")
                    }
                }
                sb.append("<|im_start|>assistant\n")
            }
            3 -> {
                var systemInjected = false
                for (msg in messages) {
                    if (msg.isUser) {
                        val prefix = if (!systemInjected && sp.isNotEmpty()) { systemInjected = true; "$sp\n\n" } else ""
                        sb.append("<start_of_turn>user\n$prefix${msg.content}<end_of_turn>\n")
                    } else {
                        sb.append("<start_of_turn>model\n${msg.content}<end_of_turn>\n")
                    }
                }
                sb.insert(0, "<bos>")
                sb.append("<start_of_turn>model\n")
            }
            4 -> {
                sb.append("<|begin_of_text|>")
                if (sp.isNotEmpty())
                    sb.append("<|start_header_id|>system<|end_header_id|>\n\n$sp<|eot_id|>")
                for (msg in messages) {
                    if (msg.isUser) {
                        sb.append("<|start_header_id|>user<|end_header_id|>\n\n${msg.content}<|eot_id|>")
                    } else {
                        sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n${msg.content}<|eot_id|>")
                    }
                }
                sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            }
        }
        return sb.toString()
    }

    // ── Üretim ───────────────────────────────────────────────────────────────────

    private fun sendMessageContent(messages: List<ChatMessage>) {
        if (loadedModelPath == null) { Toast.makeText(this, "Önce bir model yükleyin", Toast.LENGTH_SHORT).show(); return }

        val serviceIntent = Intent(this, KovaForegroundService::class.java)
        startService(serviceIntent)
        bindService(serviceIntent, serviceConnection, BIND_AUTO_CREATE)

        val convId = currentConversationId
        isGenerating = true
        tokenUpdateCounter = 0
        updateFabIcon()
        var fullResponse = ""
        var tokenCount = 0
        var generationStartTime = 0L

        val formattedText = buildFormattedPrompt(messages)
        val lastUserText = messages.lastOrNull { it.isUser }?.content ?: ""
        log("Kova", "sendMessageContent: template=$selectedTemplate turns=${messages.size} " +
            "noThinking=$noThinking lastUser='${lastUserText.take(40)}' promptLen=${formattedText.length}")

        generationJob = lifecycleScope.launch {
            var waited = 0
            while (generationService == null && waited < 20) { kotlinx.coroutines.delay(50); waited++ }
            generationService?.onGenerationStarted()

            try {
                engine.sendUserPrompt(formattedText, predictLength = contextSize)
                    .collect { token ->
                        val cleaned = token
                            .replace("<|END_OF_TURN_TOKEN|>", "")
                            .replace("<|START_OF_TURN_TOKEN|>", "")
                            .replace("<|USER_TOKEN|>", "")
                            .replace("<|CHATBOT_TOKEN|>", "")
                            .replace("<|START_RESPONSE|>", "")
                            .replace("<|END_RESPONSE|>", "")
                            .replace("<end_of_turn>", "")
                            .replace("<start_of_turn>", "")
                            .replace("<|eot_id|>", "")
                            .replace("<|im_end|>", "")
                        if (tokenCount == 0) generationStartTime = System.currentTimeMillis()
                        fullResponse += cleaned
                        tokenCount++
                        val newIndex = messageAdapter.updateLastAssistantMessage(fullResponse)
                        if (autoScroll) messagesRv.scrollToPosition(newIndex)
                        tokenUpdateCounter++
                        if (tokenUpdateCounter % 20 == 0) generationService?.onTokenUpdate(fullResponse)
                    }
            } catch (e: Exception) {
                messageAdapter.updateLastAssistantMessage(
                    if (fullResponse.isEmpty()) "[Hata: ${e.message}]" else fullResponse
                )
            } finally {
                val elapsedSec = (System.currentTimeMillis() - generationStartTime) / 1000f
                val tps = if (elapsedSec > 0f && tokenCount > 0) tokenCount / elapsedSec else null

                if (currentMessages.isNotEmpty() && !currentMessages.last().isUser) {
                    currentMessages[currentMessages.size - 1] = ChatMessage(fullResponse, false, tps)
                } else {
                    currentMessages.add(ChatMessage(fullResponse, false, tps))
                }
                // t/s'i son mesajda göster
                messageAdapter.updateLastAssistantMessage(fullResponse, tps)
                if (fullResponse.isNotEmpty()) {
                    lifecycleScope.launch(Dispatchers.IO) {
                        db.chatDao().insertMessage(com.example.llama.data.DbMessage(
                            java.util.UUID.randomUUID().toString(), convId, "assistant", fullResponse, System.currentTimeMillis()
                        ))
                        db.chatDao().touchConversation(convId, System.currentTimeMillis())
                    }
                }
                log("Kova", "Üretim bitti: ${fullResponse.length} karakter, '${fullResponse.take(60).replace("\n"," ")}'")
                isGenerating = false
                autoScroll = true
                updateFabIcon()
                generationService?.onGenerationFinished(fullResponse, isAppInForeground)
                generationService = null
                try { unbindService(serviceConnection) } catch (_: Exception) {}
            }
        }
    }

    // ── Yedekleme / Geri Yükleme ─────────────────────────────────────────────────

    private suspend fun buildBackupJson(
        inclConvs: Boolean = true,
        inclSettings: Boolean = true
    ): String {
        val prefs = getSharedPreferences("llama_prefs", MODE_PRIVATE)
        val root = JSONObject()
        root.put("version", 3); root.put("exportedAt", System.currentTimeMillis())

        // ── Ayarlar bloğu ──
        if (inclSettings) {
            val settingsObj = JSONObject().apply {
                put("context_size", contextSize)
                put("system_prompt", systemPrompt)
                put("temperature", temperature.toDouble())
                put("top_p", topP.toDouble())
                put("top_k", topK)
                put("no_thinking", noThinking)
                put("auto_load_last_model", autoLoadLastModel)
                put("flash_attn", flashAttn)
                put("last_loaded_model", prefs.getString("last_loaded_model", null) ?: "")
            }
            root.put("settings", settingsObj)
        }

        // ── Sohbetler ──
        if (inclConvs) {
            val conversations = db.chatDao().getAllConversationsList()
            val allMessages   = db.chatDao().getAllMessages()
            val convsArray = JSONArray()
            for (conv in conversations) {
                val convObj = JSONObject()
                convObj.put("id", conv.id); convObj.put("title", conv.title); convObj.put("updatedAt", conv.updatedAt)
                val msgsArray = JSONArray()
                allMessages.filter { it.conversationId == conv.id }.forEach { msg ->
                    msgsArray.put(JSONObject().apply {
                        put("id", msg.id); put("role", msg.role); put("content", msg.content); put("timestamp", msg.timestamp)
                    })
                }
                convObj.put("messages", msgsArray); convsArray.put(convObj)
            }
            root.put("conversations", convsArray)
        }

        return root.toString(2)
    }

    private fun encryptBackup(jsonText: String, password: String): ByteArray {
        val rng = SecureRandom()
        val salt = ByteArray(16).also { rng.nextBytes(it) }
        val iv   = ByteArray(12).also { rng.nextBytes(it) }
        val factory = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256")
        val keyBytes = factory.generateSecret(PBEKeySpec(password.toCharArray(), salt, 310_000, 256)).encoded
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, SecretKeySpec(keyBytes, "AES"), GCMParameterSpec(128, iv))
        return "KOVA".toByteArray() + salt + iv + cipher.doFinal(jsonText.toByteArray(Charsets.UTF_8))
    }

    private fun decryptBackup(data: ByteArray, password: String): String {
        require(data.size > 32) { "Geçersiz yedek dosyası" }
        require(String(data.slice(0..3).toByteArray()) == "KOVA") { "Bu dosya Kova yedek dosyası değil" }
        val salt = data.slice(4..19).toByteArray()
        val iv   = data.slice(20..31).toByteArray()
        val enc  = data.slice(32 until data.size).toByteArray()
        val keyBytes = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256")
            .generateSecret(PBEKeySpec(password.toCharArray(), salt, 310_000, 256)).encoded
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, SecretKeySpec(keyBytes, "AES"), GCMParameterSpec(128, iv))
        return String(cipher.doFinal(enc), Charsets.UTF_8)
    }

    private fun isEncryptedBackup(data: ByteArray): Boolean =
        data.size >= 4 && String(data.slice(0..3).toByteArray()) == "KOVA"

    private fun backupChats() {
        val dp = resources.displayMetrics.density
        val scrollView = ScrollView(this)
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding((20*dp).toInt(), (16*dp).toInt(), (20*dp).toInt(), (8*dp).toInt())
        }
        scrollView.addView(layout)

        // Bölüm başlığı
        layout.addView(TextView(this).apply {
            text = "Neleri yedeklemek istiyorsunuz?"; textSize = 13f; alpha = 0.7f
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = (8*dp).toInt() }
        })

        fun makeCheckBox(label: String, checked: Boolean = true): android.widget.CheckBox =
            android.widget.CheckBox(this).apply {
                text = label; isChecked = checked
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { bottomMargin = (4*dp).toInt() }
            }

        val cbConversations = makeCheckBox("💬 Sohbetler")
        val cbSettings      = makeCheckBox("⚙️ Ayarlar (sistem mesajı, temperature vb.)")
        layout.addView(cbConversations)
        layout.addView(cbSettings)

        // Ayırıcı
        layout.addView(View(this).apply {
            setBackgroundColor(0x22888888)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, (1*dp).toInt()
            ).apply { topMargin = (12*dp).toInt(); bottomMargin = (12*dp).toInt() }
        })

        layout.addView(TextView(this).apply {
            text = "Şifreleme (isteğe bağlı)"; textSize = 13f; alpha = 0.7f
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = (6*dp).toInt() }
        })
        val passwordInput = android.widget.EditText(this).apply {
            hint = "Şifre — boş bırakırsanız şifresiz kaydedilir"
            inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
        }
        layout.addView(passwordInput)

        AlertDialog.Builder(this).setTitle("💾 Yedekleme").setView(scrollView)
            .setPositiveButton("Yedekle") { _, _ ->
                val inclConvs     = cbConversations.isChecked
                val inclSettings  = cbSettings.isChecked
                if (!inclConvs && !inclSettings) {
                    Toast.makeText(this, "En az bir seçenek işaretleyin", Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }
                val password    = passwordInput.text.toString()
                val isEncrypted = password.isNotEmpty()
                val suffix      = buildString {
                    if (inclConvs)    append("s")
                    if (inclSettings) append("a")
                }
                val fileName = "kova_yedek_${suffix}_${System.currentTimeMillis()}.${if (isEncrypted) "kova" else "json"}"
                pendingBackupCallback = { uri ->
                    performBackupToUri(uri, password, isEncrypted, inclConvs, inclSettings)
                }
                backupSaveLauncher.launch(Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = if (isEncrypted) "application/octet-stream" else "application/json"
                    putExtra(Intent.EXTRA_TITLE, fileName)
                })
            }.setNegativeButton("İptal", null).show()
    }

    private fun performBackupToUri(
        uri: Uri, password: String, encrypt: Boolean,
        inclConvs: Boolean, inclSettings: Boolean
    ) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val jsonText = buildBackupJson(inclConvs, inclSettings)
                val conversations = if (inclConvs) db.chatDao().getAllConversationsList() else emptyList()
                contentResolver.openOutputStream(uri)?.use { out ->
                    if (encrypt) out.write(encryptBackup(jsonText, password))
                    else out.write(jsonText.toByteArray(Charsets.UTF_8))
                } ?: throw Exception("Dosya yazılamadı")
                withContext(Dispatchers.Main) {
                    val parts = buildList {
                        if (inclConvs)    add("${conversations.size} sohbet")
                        if (inclSettings) add("ayarlar")
                    }
                    Toast.makeText(this@MainActivity,
                        "${parts.joinToString(", ")} yedeklendi${if (encrypt) " (AES-256 şifreli)" else ""}",
                        Toast.LENGTH_LONG).show()
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Yedekleme hatası: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun showRestorePicker() {
        backupRestoreLauncher.launch(Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE); type = "*/*"
        })
    }

    private fun handleRestoreFile(uri: Uri) {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val bytes = contentResolver.openInputStream(uri)?.readBytes() ?: throw Exception("Dosya okunamadı")
                if (isEncryptedBackup(bytes)) {
                    withContext(Dispatchers.Main) {
                        val passInput = android.widget.EditText(this@MainActivity).apply {
                            hint = "Yedekleme şifresi"
                            inputType = android.text.InputType.TYPE_CLASS_TEXT or android.text.InputType.TYPE_TEXT_VARIATION_PASSWORD
                            setPadding(48, 24, 48, 24)
                        }
                        AlertDialog.Builder(this@MainActivity).setTitle("🔐 Şifreli Yedek")
                            .setMessage("Bu yedek şifrelenmiş. Şifreyi girin:").setView(passInput)
                            .setPositiveButton("Çöz") { _, _ ->
                                val pass = passInput.text.toString()
                                if (pass.isEmpty()) { Toast.makeText(this@MainActivity, "Şifre boş olamaz", Toast.LENGTH_SHORT).show(); return@setPositiveButton }
                                lifecycleScope.launch(Dispatchers.IO) {
                                    try {
                                        val jsonText = decryptBackup(bytes, pass)
                                        withContext(Dispatchers.Main) { showRestoreSelectionDialog(jsonText) }
                                    } catch (e: Exception) {
                                        withContext(Dispatchers.Main) { Toast.makeText(this@MainActivity, "Şifre çözme hatası.", Toast.LENGTH_LONG).show() }
                                    }
                                }
                            }.setNegativeButton("İptal", null).show()
                    }
                } else {
                    val jsonText = bytes.toString(Charsets.UTF_8)
                    withContext(Dispatchers.Main) { showRestoreSelectionDialog(jsonText) }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) { Toast.makeText(this@MainActivity, "Dosya okuma hatası: ${e.message}", Toast.LENGTH_LONG).show() }
            }
        }
    }

    private fun showRestoreSelectionDialog(jsonText: String) {
        val root = try { JSONObject(jsonText) } catch (e: Exception) {
            Toast.makeText(this, "Geçersiz yedek dosyası", Toast.LENGTH_LONG).show(); return
        }
        val version     = root.optInt("version", 1)
        val hasConvs    = root.has("conversations") && root.getJSONArray("conversations").length() > 0
        val hasSettings = version >= 3 && root.has("settings")

        val convCount = if (hasConvs) root.getJSONArray("conversations").length() else 0

        val dp = resources.displayMetrics.density
        val scrollView = ScrollView(this)
        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding((20*dp).toInt(), (16*dp).toInt(), (20*dp).toInt(), (8*dp).toInt())
        }
        scrollView.addView(layout)

        // Dosya içeriği özeti
        layout.addView(TextView(this).apply {
            text = buildString {
                append("Bu yedek dosyasında:\n")
                if (hasConvs)    append("  • $convCount sohbet\n")
                if (hasSettings) append("  • Ayarlar\n")
                if (!hasConvs && !hasSettings) append("  • (Tanınmayan format)\n")
            }
            textSize = 13f
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = (12*dp).toInt() }
        })

        layout.addView(TextView(this).apply {
            text = "Neleri geri yüklemek istiyorsunuz?"; textSize = 13f; alpha = 0.7f
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = (8*dp).toInt() }
        })

        fun makeCheckBox(label: String, enabled: Boolean): android.widget.CheckBox =
            android.widget.CheckBox(this).apply {
                text = label; isChecked = enabled; isEnabled = enabled
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { bottomMargin = (4*dp).toInt() }
            }

        val cbConvs    = makeCheckBox("💬 Sohbetler ($convCount adet)", hasConvs)
        val cbSettings = makeCheckBox("⚙️ Ayarlar", hasSettings)
        layout.addView(cbConvs)
        layout.addView(cbSettings)

        // Sohbet birleştirme seçeneği (sadece sohbetler varsa göster)
        var mergeConvs = false
        if (hasConvs) {
            layout.addView(View(this).apply {
                setBackgroundColor(0x22888888)
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, (1*dp).toInt()
                ).apply { topMargin = (12*dp).toInt(); bottomMargin = (12*dp).toInt() }
            })
            layout.addView(TextView(this).apply {
                text = "Sohbet geri yükleme yöntemi:"; textSize = 13f; alpha = 0.7f
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { bottomMargin = (6*dp).toInt() }
            })
            val rg = RadioGroup(this).apply { orientation = RadioGroup.VERTICAL }
            val rbOverwrite = RadioButton(this).apply { text = "Mevcut sohbetlerin üzerine yaz"; id = View.generateViewId(); isChecked = true }
            val rbMerge     = RadioButton(this).apply { text = "Mevcut sohbetlere ekle (birleştir)"; id = View.generateViewId() }
            rg.addView(rbOverwrite); rg.addView(rbMerge)
            rg.setOnCheckedChangeListener { _, checkedId -> mergeConvs = (checkedId == rbMerge.id) }
            layout.addView(rg)
        }

        AlertDialog.Builder(this).setTitle("📂 Geri Yükleme Seçenekleri").setView(scrollView)
            .setPositiveButton("Geri Yükle") { _, _ ->
                val doConvs    = cbConvs.isChecked
                val doSettings = cbSettings.isChecked
                if (!doConvs && !doSettings) {
                    Toast.makeText(this, "En az bir seçenek işaretleyin", Toast.LENGTH_SHORT).show()
                    return@setPositiveButton
                }
                lifecycleScope.launch(Dispatchers.IO) {
                    importJsonBackup(jsonText, doConvs, doSettings, mergeConvs)
                }
            }.setNegativeButton("İptal", null).show()
    }

    // ── Uygulama Logları ─────────────────────────────────────────────────────────

    private fun showLogsDialog() {
        val display = getLogs().ifBlank { "Henüz log yok.\nBir model yükleyip mesaj gönderin." }
        val tv = android.widget.TextView(this).apply {
            text = display; textSize = 10f; setTextIsSelectable(true)
            typeface = android.graphics.Typeface.MONOSPACE
            val pad = (8 * resources.displayMetrics.density).toInt(); setPadding(pad, pad, pad, pad)
        }
        val scroll = android.widget.ScrollView(this).apply {
            addView(tv); post { fullScroll(android.widget.ScrollView.FOCUS_DOWN) }
        }
        AlertDialog.Builder(this).setTitle("🔍 Uygulama Logları").setView(scroll)
            .setPositiveButton("Kapat", null)
            .setNeutralButton("Kopyala") { _, _ ->
                val clip = ClipData.newPlainText("log", display)
                (getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager).setPrimaryClip(clip)
                Toast.makeText(this, "Loglar kopyalandı", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton(if (loggingEnabled) "🟢 Loglamayı Kapat" else "🔴 Loglamayı Aç") { _, _ ->
                loggingEnabled = !loggingEnabled
                Toast.makeText(this, if (loggingEnabled) "Loglama açıldı" else "Loglama kapatıldı", Toast.LENGTH_SHORT).show()
            }.show()
    }

    private suspend fun importJsonBackup(
        jsonText: String,
        doConvs: Boolean = true,
        doSettings: Boolean = true,
        mergeConvs: Boolean = false
    ) {
        try {
            val root    = JSONObject(jsonText)
            val version = root.optInt("version", 1)
            val prefs   = getSharedPreferences("llama_prefs", MODE_PRIVATE)
            var settingsRestored = false
            var convCount        = 0
            var msgCount         = 0

            // ── Ayarlar ──
            if (doSettings && version >= 3 && root.has("settings")) {
                val s      = root.getJSONObject("settings")
                val editor = prefs.edit()
                if (s.has("context_size"))         editor.putInt("context_size", s.getInt("context_size"))
                if (s.has("system_prompt"))        editor.putString("system_prompt", s.getString("system_prompt"))
                if (s.has("temperature"))          editor.putFloat("temperature", s.getDouble("temperature").toFloat())
                if (s.has("top_p"))                editor.putFloat("top_p", s.getDouble("top_p").toFloat())
                if (s.has("top_k"))                editor.putInt("top_k", s.getInt("top_k"))
                if (s.has("no_thinking"))          editor.putBoolean("no_thinking", s.getBoolean("no_thinking"))
                if (s.has("auto_load_last_model")) editor.putBoolean("auto_load_last_model", s.getBoolean("auto_load_last_model"))
                if (s.has("flash_attn"))           editor.putBoolean("flash_attn", s.getBoolean("flash_attn"))
                if (s.has("last_loaded_model") && s.getString("last_loaded_model").isNotEmpty())
                    editor.putString("last_loaded_model", s.getString("last_loaded_model"))
                editor.apply()
                withContext(Dispatchers.Main) { loadSettings() }
                settingsRestored = true
            }

            // ── Sohbetler ──
            if (doConvs && root.has("conversations")) {
                val convsArray = root.getJSONArray("conversations")
                if (!mergeConvs) {
                    db.chatDao().deleteAllMessages()
                    db.chatDao().deleteAllConversations()
                }
                for (i in 0 until convsArray.length()) {
                    val convObj = convsArray.getJSONObject(i)
                    val convId  = convObj.getString("id")
                    if (mergeConvs) {
                        val exists = try { db.chatDao().getMessages(convId).isNotEmpty() } catch (_: Exception) { false }
                        val finalId = if (exists) java.util.UUID.randomUUID().toString() else convId
                        db.chatDao().insertConversation(com.example.llama.data.Conversation(
                            id = finalId, title = convObj.getString("title"), updatedAt = convObj.getLong("updatedAt")
                        ))
                        convCount++
                        val msgsArray = convObj.getJSONArray("messages")
                        for (j in 0 until msgsArray.length()) {
                            val msgObj = msgsArray.getJSONObject(j)
                            db.chatDao().insertMessage(com.example.llama.data.DbMessage(
                                id = java.util.UUID.randomUUID().toString(), conversationId = finalId,
                                role = msgObj.getString("role"), content = msgObj.getString("content"),
                                timestamp = msgObj.getLong("timestamp")
                            ))
                            msgCount++
                        }
                    } else {
                        db.chatDao().insertConversation(com.example.llama.data.Conversation(
                            id = convId, title = convObj.getString("title"), updatedAt = convObj.getLong("updatedAt")
                        ))
                        convCount++
                        val msgsArray = convObj.getJSONArray("messages")
                        for (j in 0 until msgsArray.length()) {
                            val msgObj = msgsArray.getJSONObject(j)
                            db.chatDao().insertMessage(com.example.llama.data.DbMessage(
                                id = msgObj.getString("id"), conversationId = convId,
                                role = msgObj.getString("role"), content = msgObj.getString("content"),
                                timestamp = msgObj.getLong("timestamp")
                            ))
                            msgCount++
                        }
                    }
                }
            }

            withContext(Dispatchers.Main) {
                if (doConvs) {
                    currentMessages.clear()
                    messageAdapter.submitList(emptyList())
                    lifecycleScope.launch { ensureActiveConversation() }
                }
                val parts = buildList {
                    if (doConvs && convCount > 0) add("$convCount sohbet ($msgCount mesaj)${if (mergeConvs) " eklendi" else " geri yüklendi"}")
                    if (settingsRestored) add("ayarlar geri yüklendi")
                }
                AlertDialog.Builder(this@MainActivity).setTitle("✅ Geri Yükleme Tamamlandı")
                    .setMessage(parts.joinToString("\n• ", prefix = "• "))
                    .setPositiveButton("Tamam", null).show()
            }
        } catch (e: Exception) {
            withContext(Dispatchers.Main) {
                Toast.makeText(this@MainActivity, "Geri yükleme hatası: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }
    }
}
