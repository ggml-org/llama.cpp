```mermaid
sequenceDiagram
    participant UI as 🧩 ChatSidebar / ChatScreen
    participant convStore as 🗄️ conversationsStore
    participant chatStore as 🗄️ chatStore
    participant DbSvc as ⚙️ DatabaseService
    participant IDB as 💾 IndexedDB

    Note over convStore: State:<br/>conversations: DatabaseConversation[]<br/>activeConversation: DatabaseConversation | null<br/>activeMessages: DatabaseMessage[]<br/>isInitialized: boolean<br/>pendingMcpServerOverrides: Map&lt;string, McpServerOverride&gt;

    %% ═══════════════════════════════════════════════════════════════════════════
    Note over UI,IDB: 🚀 INITIALIZATION
    %% ═══════════════════════════════════════════════════════════════════════════

    Note over convStore: Auto-initialized in constructor (browser only)
    convStore->>convStore: initialize()
    activate convStore
    convStore->>convStore: loadConversations()
    convStore->>DbSvc: getAllConversations()
    DbSvc->>IDB: SELECT * FROM conversations ORDER BY lastModified DESC
    IDB-->>DbSvc: Conversation[]
    DbSvc-->>convStore: conversations
    convStore->>convStore: conversations = $state(data)
    convStore->>convStore: isInitialized = true
    deactivate convStore

    %% ═══════════════════════════════════════════════════════════════════════════
    Note over UI,IDB: ➕ CREATE CONVERSATION
    %% ═══════════════════════════════════════════════════════════════════════════

    UI->>convStore: createConversation(name?)
    activate convStore
    convStore->>DbSvc: createConversation(name || "New Chat")
    DbSvc->>IDB: INSERT INTO conversations
    IDB-->>DbSvc: conversation {id, name, lastModified, currNode: ""}
    DbSvc-->>convStore: conversation
    convStore->>convStore: conversations.unshift(conversation)
    convStore->>convStore: activeConversation = $state(conversation)
    convStore->>convStore: activeMessages = $state([])

    alt pendingMcpServerOverrides has entries
        loop each pending override
            convStore->>DbSvc: Store MCP server override for new conversation
        end
        convStore->>convStore: clearPendingMcpServerOverrides()
    end
    deactivate convStore

    %% ═══════════════════════════════════════════════════════════════════════════
    Note over UI,IDB: 📂 LOAD CONVERSATION
    %% ═══════════════════════════════════════════════════════════════════════════

    UI->>convStore: loadConversation(convId)
    activate convStore
    convStore->>DbSvc: getConversation(convId)
    DbSvc->>IDB: SELECT * FROM conversations WHERE id = ?
    IDB-->>DbSvc: conversation
    convStore->>convStore: activeConversation = $state(conversation)

    convStore->>convStore: refreshActiveMessages()
    convStore->>DbSvc: getConversationMessages(convId)
    DbSvc->>IDB: SELECT * FROM messages WHERE convId = ?
    IDB-->>DbSvc: allMessages[]
    convStore->>convStore: filterByLeafNodeId(allMessages, currNode)
    Note right of convStore: Filter to show only current branch path
    convStore->>convStore: activeMessages = $state(filtered)

    Note right of convStore: Route (+page.svelte) then calls:<br/>chatStore.syncLoadingStateForChat(convId)
    deactivate convStore

    %% ═══════════════════════════════════════════════════════════════════════════
    Note over UI,IDB: 🌳 MESSAGE BRANCHING MODEL
    %% ═══════════════════════════════════════════════════════════════════════════

    Note over IDB: Message Tree Structure:<br/>- Each message has parent (null for root)<br/>- Each message has children[] array<br/>- Conversation.currNode points to active leaf<br/>- filterByLeafNodeId() traverses from root to currNode

    rect rgb(240, 240, 255)
        Note over convStore: Example Branch Structure:
        Note over convStore: root → user1 → assistant1 → user2 → assistant2a (currNode)<br/>                                    ↘ assistant2b (alt branch)
    end

    %% ═══════════════════════════════════════════════════════════════════════════
    Note over UI,IDB: ↔️ BRANCH NAVIGATION
    %% ═══════════════════════════════════════════════════════════════════════════

    UI->>convStore: navigateToSibling(msgId, direction)
    activate convStore
    convStore->>convStore: Find message in activeMessages
    convStore->>convStore: Get parent message
    convStore->>convStore: Find sibling in parent.children[]
    convStore->>convStore: findLeafNode(siblingId, allMessages)
    Note right of convStore: Navigate to leaf of sibling branch
    convStore->>convStore: updateCurrentNode(leafId)
    convStore->>DbSvc: updateCurrentNode(convId, leafId)
    DbSvc->>IDB: UPDATE conversations SET currNode = ?
    convStore->>convStore: refreshActiveMessages()
    deactivate convStore

    %% ═══════════════════════════════════════════════════════════════════════════
    Note over UI,IDB: 📝 UPDATE CONVERSATION
    %% ═══════════════════════════════════════════════════════════════════════════

    UI->>convStore: updateConversationName(convId, newName)
    activate convStore
    convStore->>DbSvc: updateConversation(convId, {name: newName})
    DbSvc->>IDB: UPDATE conversations SET name = ?
    convStore->>convStore: Update in conversations array
    deactivate convStore

    Note over convStore: Auto-title update (after first response):
    convStore->>convStore: updateConversationTitleWithConfirmation()
    convStore->>convStore: titleUpdateConfirmationCallback?()
    Note right of convStore: Shows dialog if title would change

    %% ═══════════════════════════════════════════════════════════════════════════
    Note over UI,IDB: 🗑️ DELETE CONVERSATION
    %% ═══════════════════════════════════════════════════════════════════════════

    UI->>convStore: deleteConversation(convId)
    activate convStore
    convStore->>DbSvc: deleteConversation(convId)
    DbSvc->>IDB: DELETE FROM conversations WHERE id = ?
    DbSvc->>IDB: DELETE FROM messages WHERE convId = ?
    convStore->>convStore: conversations.filter(c => c.id !== convId)
    alt deleted active conversation
        convStore->>convStore: clearActiveConversation()
    end
    deactivate convStore

    UI->>convStore: deleteAll()
    activate convStore
    convStore->>DbSvc: Delete all conversations and messages
    convStore->>convStore: conversations = []
    convStore->>convStore: clearActiveConversation()
    deactivate convStore

    %% ═══════════════════════════════════════════════════════════════════════════
    Note over UI,IDB: � MCP SERVER PER-CHAT OVERRIDES
    %% ═══════════════════════════════════════════════════════════════════════════

    Note over convStore: Conversations can override which MCP servers are enabled.
    Note over convStore: Uses pendingMcpServerOverrides before conversation<br/>is created, then persists to conversation metadata.

    UI->>convStore: setMcpServerOverride(convId, serverName, override)
    Note right of convStore: override = {enabled: boolean}

    UI->>convStore: toggleMcpServerForChat(convId, serverName, enabled)
    activate convStore
    convStore->>convStore: setMcpServerOverride(convId, serverName, {enabled})
    deactivate convStore

    UI->>convStore: isMcpServerEnabledForChat(convId, serverName)
    Note right of convStore: Check override → fall back to global MCP config

    UI->>convStore: getAllMcpServerOverrides(convId)
    Note right of convStore: Returns all overrides for a conversation

    UI->>convStore: removeMcpServerOverride(convId, serverName)
    UI->>convStore: getMcpServerOverride(convId, serverName)

    %% ═══════════════════════════════════════════════════════════════════════════
    Note over UI,IDB: 📤 EXPORT / 📥 IMPORT
    %% ═══════════════════════════════════════════════════════════════════════════

    UI->>convStore: exportAllConversations()
    activate convStore
    convStore->>DbSvc: getAllConversations()
    loop each conversation
        convStore->>DbSvc: getConversationMessages(convId)
    end
    convStore->>convStore: triggerDownload(JSON blob)
    deactivate convStore

    UI->>convStore: importConversations(file)
    activate convStore
    convStore->>convStore: Parse JSON file
    convStore->>convStore: importConversationsData(parsed)
    convStore->>DbSvc: importConversations(parsed)
    Note right of DbSvc: Skips duplicate conversations<br/>(checks existing by ID)
    DbSvc->>IDB: INSERT conversations + messages (skip existing)
    convStore->>convStore: loadConversations()
    deactivate convStore
```
