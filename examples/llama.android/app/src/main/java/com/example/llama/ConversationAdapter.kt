package com.example.llama

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.example.llama.data.Conversation

class ConversationAdapter(
    private val onSelect: (Conversation) -> Unit,
    private val onDelete: (Conversation) -> Unit
) : RecyclerView.Adapter<ConversationAdapter.VH>() {

    private val items = mutableListOf<Conversation>()
    var activeId: String? = null

    inner class VH(view: View) : RecyclerView.ViewHolder(view) {
        val title: TextView = view.findViewById(R.id.conv_title)
        val btnDelete: ImageButton = view.findViewById(R.id.btn_delete_conv)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH =
        VH(LayoutInflater.from(parent.context).inflate(R.layout.item_conversation, parent, false))

    override fun onBindViewHolder(holder: VH, position: Int) {
        val conv = items[position]
        holder.title.text = conv.title
        holder.itemView.isSelected = conv.id == activeId
        holder.itemView.alpha = if (conv.id == activeId) 1f else 0.85f
        holder.itemView.setOnClickListener { onSelect(conv) }
        holder.btnDelete.setOnClickListener { onDelete(conv) }
    }

    override fun getItemCount() = items.size

    fun submitList(list: List<Conversation>) {
        items.clear()
        items.addAll(list)
        notifyDataSetChanged()
    }
}
