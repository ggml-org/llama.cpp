package com.example.llama

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import io.noties.markwon.Markwon

data class ChatMessage(
    val content: String,
    val isUser: Boolean,
    val tokensPerSecond: Float? = null
)

class MessageAdapter(
    private val onCopy: (String) -> Unit,
    private val onEdit: (Int, String) -> Unit,
    private val onRegenerate: (Int) -> Unit
) : RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    private val messages = mutableListOf<ChatMessage>()
    private var markwon: Markwon? = null

    // Hangi pozisyonlardaki thinking bloğu açık — notifyItemChanged sonrası da korunur
    private val expandedPositions = mutableSetOf<Int>()

    companion object {
        private const val VIEW_TYPE_USER = 1
        private const val VIEW_TYPE_ASSISTANT = 2
    }

    // --- Thinking block ayrıştırma ---

    private data class ParsedMessage(
        val thinkContent: String?,
        val visibleContent: String
    )

    private fun parseThinking(raw: String): ParsedMessage {
        val completeRegex = Regex("""<think>(.*?)</think>""", RegexOption.DOT_MATCHES_ALL)
        val completeMatch = completeRegex.find(raw)
        if (completeMatch != null) {
            val thinkContent = completeMatch.groupValues[1].trim()
            val visible = raw.removeRange(completeMatch.range).trim()
            return ParsedMessage(thinkContent, visible)
        }
        val openIdx = raw.indexOf("<think>")
        if (openIdx != -1) {
            val thinkContent = raw.substring(openIdx + 7).trim()
            val visible = raw.substring(0, openIdx).trim()
            return ParsedMessage("$thinkContent▌", visible)
        }
        return ParsedMessage(null, raw)
    }

    // --- Adapter ---

    fun submitList(newMessages: List<ChatMessage>) {
        messages.clear()
        messages.addAll(newMessages)
        expandedPositions.clear()
        notifyDataSetChanged()
    }

    // Geriye dönük uyumluluk için bırakıldı — artık kullanılmıyor.
    var isStreaming: Boolean = false
    var markdownThisUpdate: Boolean = false

    fun updateLastAssistantMessage(text: String, tps: Float? = null): Int {
        if (messages.isNotEmpty() && !messages.last().isUser) {
            messages[messages.size - 1] = ChatMessage(content = text, isUser = false, tokensPerSecond = tps)
            notifyItemChanged(messages.size - 1)
        } else {
            messages.add(ChatMessage(content = text, isUser = false, tokensPerSecond = tps))
            notifyItemInserted(messages.size - 1)
        }
        return messages.size - 1
    }

    override fun getItemViewType(position: Int) =
        if (messages[position].isUser) VIEW_TYPE_USER else VIEW_TYPE_ASSISTANT

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        if (markwon == null) markwon = Markwon.create(parent.context)
        val inflater = LayoutInflater.from(parent.context)
        return if (viewType == VIEW_TYPE_USER)
            UserViewHolder(inflater.inflate(R.layout.item_message_user, parent, false))
        else
            AssistantViewHolder(inflater.inflate(R.layout.item_message_assistant, parent, false))
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val message = messages[position]

        if (holder is UserViewHolder) {
            holder.itemView.findViewById<TextView>(R.id.msg_content).text = message.content
            holder.itemView.findViewById<Button>(R.id.btn_copy).setOnClickListener {
                onCopy(message.content)
            }
            holder.itemView.findViewById<Button>(R.id.btn_edit).setOnClickListener {
                onEdit(position, message.content)
            }

        } else if (holder is AssistantViewHolder) {
            val parsed = parseThinking(message.content)

            // --- Thinking card ---
            if (parsed.thinkContent != null) {
                holder.thinkingSection.visibility = View.VISIBLE
                holder.thinkingContent.text = parsed.thinkContent

                val isExpanded = expandedPositions.contains(position)
                holder.thinkingContent.visibility = if (isExpanded) View.VISIBLE else View.GONE
                holder.thinkingChevron.text = if (isExpanded) "▴" else "▾"

                holder.thinkingHeader.setOnClickListener {
                    val nowExpanded = expandedPositions.contains(position)
                    if (nowExpanded) {
                        expandedPositions.remove(position)
                        holder.thinkingContent.visibility = View.GONE
                        holder.thinkingChevron.text = "▾"
                    } else {
                        expandedPositions.add(position)
                        holder.thinkingContent.visibility = View.VISIBLE
                        holder.thinkingChevron.text = "▴"
                    }
                }
            } else {
                holder.thinkingSection.visibility = View.GONE
            }

            // --- Asıl mesaj ---
            val textView = holder.itemView.findViewById<TextView>(R.id.msg_content)
            val displayText = parsed.visibleContent.ifEmpty {
                if (parsed.thinkContent != null) "" else "…"
            }

            val isLastMessage = position == messages.size - 1
            // Her durumda Markdown render — üretim sırasında da, bitince de.
            markwon?.setMarkdown(textView, displayText) ?: run { textView.text = displayText }
            textView.setTextIsSelectable(true)

            // --- t/s göstergesi ---
            val tps = message.tokensPerSecond
            if (tps != null && tps > 0f) {
                holder.txtTps.visibility = View.VISIBLE
                holder.txtTps.text = "%.2f t/s".format(tps)
            } else {
                holder.txtTps.visibility = View.GONE
            }

            holder.itemView.findViewById<Button>(R.id.btn_copy).setOnClickListener {
                onCopy(parsed.visibleContent.ifEmpty { message.content })
            }
            holder.itemView.findViewById<Button>(R.id.btn_regenerate).setOnClickListener {
                onRegenerate(position)
            }
        }
    }

    override fun getItemCount() = messages.size

    class UserViewHolder(view: View) : RecyclerView.ViewHolder(view)

    class AssistantViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val thinkingSection: LinearLayout = view.findViewById(R.id.thinking_section)
        val thinkingHeader: LinearLayout = view.findViewById(R.id.thinking_header)
        val thinkingContent: TextView = view.findViewById(R.id.thinking_content)
        val thinkingChevron: TextView = view.findViewById(R.id.thinking_chevron)
        val txtTps: TextView = view.findViewById(R.id.txt_tps)
    }
}
