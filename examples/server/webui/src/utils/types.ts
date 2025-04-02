export interface TimingReport {
  prompt_n: number;
  prompt_ms: number;
  predicted_n: number;
  predicted_ms: number;
}

/**
 * What is conversation "branching"? It is a feature that allows the user to edit an old message in the history, while still keeping the conversation flow.
 * Inspired by ChatGPT / Claude / Hugging Chat where you edit a message, a new branch of the conversation is created, and the old message is still visible.
 *
 * We use the same node-based structure like other chat UIs, where each message has a parent and children. A "root" message is the first message in a conversation, which will not be displayed in the UI.
 *
 * root
 *  ├── message 1
 *  │      └── message 2
 *  │             └── message 3
 *  └── message 4
 *        └── message 5
 *
 * In the above example, assuming that user wants to edit message 2, a new branch will be created:
 *
 *          ├── message 2
 *          │   └── message 3
 *          └── message 6
 *
 * Message 2 and 6 are siblings, and message 6 is the new branch.
 *
 * We only need to know the last node (aka leaf) to get the current branch. In the above example, message 5 is the leaf of branch containing message 4 and 5.
 *
 * For the implementation:
 * - StorageUtils.getMessages() returns list of all nodes
 * - StorageUtils.filterByLeafNodeId() filters the list of nodes from a given leaf node
 */

// Note: the term "message" and "node" are used interchangeably in this context
export interface Message {
  id: number;
  convId: string;
  type: 'text' | 'root';
  timestamp: number; // timestamp from Date.now()
  role: 'user' | 'assistant' | 'system';
  content: string;
  timings?: TimingReport;
  extra?: MessageExtra[];
  // node based system for branching
  parent: Message['id'];
  children: Message['id'][];
}

type MessageExtra = MessageExtraTextFile | MessageExtraContext; // TODO: will add more in the future

export interface MessageExtraTextFile {
  type: 'textFile';
  name: string;
  content: string;
}

export interface MessageExtraContext {
  type: 'context';
  content: string;
}

export type APIMessage = Pick<Message, 'role' | 'content'>;

export interface Conversation {
  id: string; // format: `conv-{timestamp}`
  lastModified: number; // timestamp from Date.now()
  currNode: Message['id']; // the current message node being viewed
  name: string;
}

export interface ViewingChat {
  conv: Readonly<Conversation>;
  messages: Readonly<Message[]>;
}

export type PendingMessage = Omit<Message, 'content'> & {
  content: string | null;
};

export enum CanvasType {
  PY_INTERPRETER,
}

export interface CanvasPyInterpreter {
  type: CanvasType.PY_INTERPRETER;
  content: string;
}

export type CanvasData = CanvasPyInterpreter;
