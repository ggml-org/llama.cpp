// coversations is stored in localStorage
// format: { [convId]: { id: string, lastModified: number, messages: [...] } }

import { CONFIG_DEFAULT } from '../Config';
import { Conversation, Message, TimingReport } from './types';
import Dexie, { Table } from 'dexie';

const event = new EventTarget();

type CallbackConversationChanged = (convId: string) => void;
let onConversationChangedHandlers: [
  CallbackConversationChanged,
  EventListener,
][] = [];
const dispatchConversationChange = (convId: string) => {
  event.dispatchEvent(
    new CustomEvent('conversationChange', { detail: { convId } })
  );
};

const db = new Dexie('LlamacppWebui') as Dexie & {
  conversations: Table<Conversation>;
  messages: Table<Message>;
};

// https://dexie.org/docs/Version/Version.stores()
db.version(1).stores({
  // Unlike SQL, you don’t need to specify all properties but only the one you wish to index.
  conversations: '&id, lastModified',
  messages: '&id, convId, [convId+id], timestamp',
});

// convId is a string prefixed with 'conv-'
const StorageUtils = {
  /**
   * manage conversations
   */
  async getAllConversations(): Promise<Conversation[]> {
    await migrationLStoIDB().catch(console.error); // noop if already migrated
    return (await db.conversations.toArray()).sort(
      (a, b) => b.lastModified - a.lastModified
    );
  },
  /**
   * can return null if convId does not exist
   */
  async getOneConversation(convId: string): Promise<Conversation | null> {
    return (await db.conversations.where('id').equals(convId).first()) ?? null;
  },
  /**
   * get all message nodes in a conversation
   */
  async getMessages(convId: string): Promise<Message[]> {
    return await db.messages.where({ convId }).toArray();
  },
  /**
   * use in conjunction with getMessages to filter messages by leafNodeId
   * includeRoot: whether to include the root node in the result
   * if node with leafNodeId does not exist, return the path with the latest timestamp
   */
  filterByLeafNodeId(
    msgs: Readonly<Message[]>,
    leafNodeId: Message['id'],
    includeRoot: boolean
  ): Readonly<Message[]> {
    const res: Message[] = [];
    const nodeMap = new Map<Message['id'], Message>();
    for (const msg of msgs) {
      nodeMap.set(msg.id, msg);
    }
    let startNode: Message | undefined = nodeMap.get(leafNodeId);
    if (!startNode) {
      // if not found, we return the path with the latest timestamp
      let latestTime = -1;
      for (const msg of msgs) {
        if (msg.timestamp > latestTime) {
          startNode = msg;
          latestTime = msg.timestamp;
        }
      }
    }
    // traverse the path from leafNodeId to root
    // startNode can never be undefined here
    let currNode: Message | undefined = startNode;
    while (currNode) {
      if (currNode.type !== 'root' || (currNode.type === 'root' && includeRoot))
        res.push(currNode);
      currNode = nodeMap.get(currNode.parent ?? -1);
    }
    res.sort((a, b) => a.timestamp - b.timestamp);
    return res;
  },
  /**
   * create a new conversation with a default root node
   */
  async createConversation(name: string): Promise<Conversation> {
    const now = Date.now();
    const msgId = now;
    const conv: Conversation = {
      id: `conv-${now}`,
      lastModified: now,
      currNode: msgId,
      name,
    };
    await db.conversations.add(conv);
    // create a root node
    await db.messages.add({
      id: msgId,
      convId: conv.id,
      type: 'root',
      timestamp: now,
      role: 'system',
      content: '',
      parent: -1,
      children: [],
    });
    return conv;
  },
  /**
   * update the name of a conversation
   */
  async updateConversationName(convId: string, name: string): Promise<void> {
    await db.conversations.update(convId, {
      name,
      lastModified: Date.now(),
    });
    dispatchConversationChange(convId);
  },
  /**
   * if convId does not exist, throw an error
   */
  async appendMsg(
    msg: Exclude<Message, 'parent' | 'children'>,
    parentNodeId: Message['id']
  ): Promise<void> {
    await this.appendMsgChain([msg], parentNodeId);
  },

  /**
   * Adds chain of messages to the DB, usually
   * produced by tool calling.
   */
  async appendMsgChain(
    messages: Exclude<Message, 'parent' | 'children'>[],
    parentNodeId: Message['id']
  ): Promise<void> {
    if (messages.length === 0) return;

    const { convId } = messages[0];

    // Verify conversation exists
    const conv = await this.getOneConversation(convId);
    if (!conv) {
      throw new Error(`Conversation ${convId} does not exist`);
    }

    // Verify starting parent exists
    const startParent = await db.messages
      .where({ convId, id: parentNodeId })
      .first();
    if (!startParent) {
      throw new Error(
        `Starting parent message ${parentNodeId} does not exist in conversation ${convId}`
      );
    }

    // Get the last message ID for updating the conversation
    const lastMsgId = messages[messages.length - 1].id;

    try {
      // Process all messages in a single transaction
      await db.transaction('rw', db.messages, db.conversations, () => {
        // First message connects to startParentId
        let parentId = parentNodeId;
        const parentChildren = [...startParent.children];

        for (let i = 0; i < messages.length; i++) {
          const msg = messages[i];

          // Add this message to its parent's children
          if (i === 0) {
            // First message - update the starting parent
            parentChildren.push(msg.id);
            db.messages.update(parentId, { children: parentChildren });
          } else {
            // Other messages - previous message is the parent
            db.messages.update(parentId, { children: [msg.id] });
          }

          // Add the message
          db.messages.add({
            ...msg,
            parent: parentId,
            children: [], // Will be updated if this message has children
          });

          // Next message's parent is this message
          parentId = msg.id;
        }

        // Update the conversation
        db.conversations.update(convId, {
          lastModified: Date.now(),
          currNode: lastMsgId,
        });
      });
    } catch (error) {
      console.error('Error saving message chain:', error);
      throw error;
    }

    dispatchConversationChange(convId);
  },

  /**
   * remove conversation by id
   */
  async remove(convId: string): Promise<void> {
    await db.transaction('rw', db.conversations, db.messages, async () => {
      await db.conversations.delete(convId);
      await db.messages.where({ convId }).delete();
    });
    dispatchConversationChange(convId);
  },

  // event listeners
  onConversationChanged(callback: CallbackConversationChanged) {
    const fn = (e: Event) => callback((e as CustomEvent).detail.convId);
    onConversationChangedHandlers.push([callback, fn]);
    event.addEventListener('conversationChange', fn);
  },
  offConversationChanged(callback: CallbackConversationChanged) {
    const fn = onConversationChangedHandlers.find(([cb, _]) => cb === callback);
    if (fn) {
      event.removeEventListener('conversationChange', fn[1]);
    }
    onConversationChangedHandlers = [];
  },

  // manage config
  getConfig(): typeof CONFIG_DEFAULT {
    const savedVal = JSON.parse(localStorage.getItem('config') || '{}');
    // to prevent breaking changes in the future, we always provide default value for missing keys
    return {
      ...CONFIG_DEFAULT,
      ...savedVal,
    };
  },
  setConfig(config: typeof CONFIG_DEFAULT) {
    localStorage.setItem('config', JSON.stringify(config));
  },
  getTheme(): string {
    return localStorage.getItem('theme') || 'auto';
  },
  setTheme(theme: string) {
    if (theme === 'auto') {
      localStorage.removeItem('theme');
    } else {
      localStorage.setItem('theme', theme);
    }
  },

  /**
   * Import a demo conversation from JSON data
   */
  async importDemoConversation(demoConv: {
    demo?: boolean;
    id: string;
    lastModified: number;
    messages: Message[];
  }): Promise<void> {
    if (demoConv.messages.length === 0) return;

    // Create a new conversation with a fresh ID
    const conv = await this.createConversation('Demo Conversation');

    // Get the root message to use as parent
    const rootMessages = await this.getMessages(conv.id);
    const rootMessage = rootMessages.find((msg) => msg.type === 'root');
    if (!rootMessage) {
      throw new Error('Failed to find root message in created conversation');
    }

    // Transform demo messages to the format expected by appendMsgChain
    const now = Date.now();
    const transformedMessages = demoConv.messages
      .filter(
        (msg) =>
          msg && msg.role && msg.content !== undefined && msg.id !== undefined
      )
      .map((msg, index) => ({
        id: now + index,
        convId: conv.id,
        type: 'text' as const,
        tool_calls: msg.tool_calls || [],
        timestamp: now + index,
        role: msg.role,
        content: msg.content,
        timings: msg.timings,
      })) as Exclude<Message, 'parent' | 'children'>[];

    // Use appendMsgChain to add all messages at once
    await this.appendMsgChain(transformedMessages, rootMessage.id);
  },
};

export default StorageUtils;

// Migration from localStorage to IndexedDB

// these are old types, LS prefix stands for LocalStorage
interface LSConversation {
  id: string; // format: `conv-{timestamp}`
  lastModified: number; // timestamp from Date.now()
  messages: LSMessage[];
}
interface LSMessage {
  id: number;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timings?: TimingReport;
}
async function migrationLStoIDB() {
  if (localStorage.getItem('migratedToIDB')) return;
  const res: LSConversation[] = [];
  for (const key in localStorage) {
    if (key.startsWith('conv-')) {
      res.push(JSON.parse(localStorage.getItem(key) ?? '{}'));
    }
  }
  if (res.length === 0) return;
  await db.transaction('rw', db.conversations, db.messages, async () => {
    let migratedCount = 0;
    for (const conv of res) {
      const { id: convId, lastModified, messages } = conv;
      const firstMsg = messages[0];
      const lastMsg = messages.at(-1);
      if (messages.length < 2 || !firstMsg || !lastMsg) {
        console.log(
          `Skipping conversation ${convId} with ${messages.length} messages`
        );
        continue;
      }
      const name = firstMsg.content ?? '(no messages)';
      await db.conversations.add({
        id: convId,
        lastModified,
        currNode: lastMsg.id,
        name,
      });
      const rootId = messages[0].id - 2;
      await db.messages.add({
        id: rootId,
        convId: convId,
        type: 'root',
        timestamp: rootId,
        role: 'system',
        content: '',
        parent: -1,
        children: [firstMsg.id],
      });
      for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        await db.messages.add({
          ...msg,
          type: 'text',
          convId: convId,
          timestamp: msg.id,
          parent: i === 0 ? rootId : messages[i - 1].id,
          children: i === messages.length - 1 ? [] : [messages[i + 1].id],
        });
      }
      migratedCount++;
      console.log(
        `Migrated conversation ${convId} with ${messages.length} messages`
      );
    }
    console.log(
      `Migrated ${migratedCount} conversations from localStorage to IndexedDB`
    );
    localStorage.setItem('migratedToIDB', '1');
  });
}
