/**
 * Message branching utilities for conversation tree navigation.
 *
 * Conversation branching allows users to edit messages and create alternate paths
 * while preserving the original conversation flow. Each message has parent/children
 * relationships forming a tree structure.
 *
 * Example tree:
 * root
 *  ├── message 1 (user)
 *  │      └── message 2 (assistant)
 *  │             ├── message 3 (user)
 *  │             └── message 6 (user) ← new branch
 *  └── message 4 (user)
 *        └── message 5 (assistant)
 */

/**
 * Filters messages to get the conversation path from root to a specific leaf node.
 * If the leafNodeId doesn't exist, returns the path with the latest timestamp.
 *
 * @param messages - All messages in the conversation
 * @param leafNodeId - The target leaf node ID to trace back from
 * @param includeRoot - Whether to include root messages in the result
 * @returns Array of messages from root to leaf, sorted by timestamp
 */
export function filterByLeafNodeId(
	messages: readonly DatabaseMessage[],
	leafNodeId: string,
	includeRoot: boolean = false
): readonly DatabaseMessage[] {
	const result: DatabaseMessage[] = [];
	const nodeMap = new Map<string, DatabaseMessage>();

	// Build node map for quick lookups
	for (const msg of messages) {
		nodeMap.set(msg.id, msg);
	}

	// Find the starting node (leaf node or latest if not found)
	let startNode: DatabaseMessage | undefined = nodeMap.get(leafNodeId);
	if (!startNode) {
		// If leaf node not found, use the message with latest timestamp
		let latestTime = -1;
		for (const msg of messages) {
			if (msg.timestamp > latestTime) {
				startNode = msg;
				latestTime = msg.timestamp;
			}
		}
	}

	// Traverse from leaf to root, collecting messages
	let currentNode: DatabaseMessage | undefined = startNode;
	while (currentNode) {
		// Include message if it's not root, or if we want to include root
		if (currentNode.type !== 'root' || includeRoot) {
			result.push(currentNode);
		}

		// Stop traversal if parent is null (reached root)
		if (currentNode.parent === null) {
			break;
		}
		currentNode = nodeMap.get(currentNode.parent);
	}

	// Sort by timestamp to get chronological order (root to leaf)
	result.sort((a, b) => a.timestamp - b.timestamp);
	return result;
}

/**
 * Finds the leaf node (message with no children) for a given message branch.
 * Traverses down the tree following the last child until reaching a leaf.
 *
 * @param messages - All messages in the conversation
 * @param messageId - Starting message ID to find leaf for
 * @returns The leaf node ID, or the original messageId if no children
 */
export function findLeafNode(messages: readonly DatabaseMessage[], messageId: string): string {
	const nodeMap = new Map<string, DatabaseMessage>();

	// Build node map for quick lookups
	for (const msg of messages) {
		nodeMap.set(msg.id, msg);
	}

	let currentNode: DatabaseMessage | undefined = nodeMap.get(messageId);
	while (currentNode && currentNode.children.length > 0) {
		// Follow the last child (most recent branch)
		const lastChildId = currentNode.children[currentNode.children.length - 1];
		currentNode = nodeMap.get(lastChildId);
	}

	return currentNode?.id ?? messageId;
}

/**
 * Finds all descendant messages (children, grandchildren, etc.) of a given message.
 * This is used for cascading deletion to remove all messages in a branch.
 *
 * @param messages - All messages in the conversation
 * @param messageId - The root message ID to find descendants for
 * @returns Array of all descendant message IDs
 */
export function findDescendantMessages(
	messages: readonly DatabaseMessage[],
	messageId: string
): string[] {
	const nodeMap = new Map<string, DatabaseMessage>();

	// Build node map for quick lookups
	for (const msg of messages) {
		nodeMap.set(msg.id, msg);
	}

	const descendants: string[] = [];
	const queue: string[] = [messageId];

	while (queue.length > 0) {
		const currentId = queue.shift()!;
		const currentNode = nodeMap.get(currentId);

		if (currentNode) {
			// Add all children to the queue and descendants list
			for (const childId of currentNode.children) {
				descendants.push(childId);
				queue.push(childId);
			}
		}
	}

	return descendants;
}

/**
 * Gets sibling information for a message, including all sibling IDs and current position.
 * Siblings are messages that share the same parent.
 *
 * @param messages - All messages in the conversation
 * @param messageId - The message to get sibling info for
 * @returns Sibling information including leaf node IDs for navigation
 */
export function getMessageSiblings(
	messages: readonly DatabaseMessage[],
	messageId: string
): ChatMessageSiblingInfo | null {
	const nodeMap = new Map<string, DatabaseMessage>();

	// Build node map for quick lookups
	for (const msg of messages) {
		nodeMap.set(msg.id, msg);
	}

	const message = nodeMap.get(messageId);
	if (!message) {
		return null;
	}

	// Handle null parent (root message) case
	if (message.parent === null) {
		// No parent means this is likely a root node with no siblings
		return {
			message,
			siblingIds: [messageId],
			currentIndex: 0,
			totalSiblings: 1
		};
	}

	const parentNode = nodeMap.get(message.parent);
	if (!parentNode) {
		// Parent not found - treat as single message
		return {
			message,
			siblingIds: [messageId],
			currentIndex: 0,
			totalSiblings: 1
		};
	}

	// Get all sibling IDs (including self)
	const siblingIds = parentNode.children;

	// Convert sibling message IDs to their corresponding leaf node IDs
	// This allows navigation between different conversation branches
	const siblingLeafIds = siblingIds.map((siblingId: string) => findLeafNode(messages, siblingId));

	// Find current message's position among siblings
	const currentIndex = siblingIds.indexOf(messageId);

	return {
		message,
		siblingIds: siblingLeafIds,
		currentIndex,
		totalSiblings: siblingIds.length
	};
}

/**
 * Creates a display-ready list of messages with sibling information for UI rendering.
 * This is the main function used by chat components to render conversation branches.
 *
 * @param messages - All messages in the conversation
 * @param leafNodeId - Current leaf node being viewed
 * @returns Array of messages with sibling navigation info
 */
export function getMessageDisplayList(
	messages: readonly DatabaseMessage[],
	leafNodeId: string
): ChatMessageSiblingInfo[] {
	// Get the current conversation path
	const currentPath = filterByLeafNodeId(messages, leafNodeId, true);
	const result: ChatMessageSiblingInfo[] = [];

	// Add sibling info for each message in the current path
	for (const message of currentPath) {
		if (message.type === 'root') {
			continue; // Skip root messages in display
		}

		const siblingInfo = getMessageSiblings(messages, message.id);
		if (siblingInfo) {
			result.push(siblingInfo);
		}
	}

	return result;
}

/**
 * Checks if a message has multiple siblings (indicating branching at that point).
 *
 * @param messages - All messages in the conversation
 * @param messageId - The message to check
 * @returns True if the message has siblings
 */
export function hasMessageSiblings(
	messages: readonly DatabaseMessage[],
	messageId: string
): boolean {
	const siblingInfo = getMessageSiblings(messages, messageId);
	return siblingInfo ? siblingInfo.totalSiblings > 1 : false;
}

/**
 * Gets the next sibling message ID for navigation.
 *
 * @param messages - All messages in the conversation
 * @param messageId - Current message ID
 * @returns Next sibling's leaf node ID, or null if at the end
 */
export function getNextSibling(
	messages: readonly DatabaseMessage[],
	messageId: string
): string | null {
	const siblingInfo = getMessageSiblings(messages, messageId);
	if (!siblingInfo || siblingInfo.currentIndex >= siblingInfo.totalSiblings - 1) {
		return null;
	}

	return siblingInfo.siblingIds[siblingInfo.currentIndex + 1];
}

/**
 * Gets the previous sibling message ID for navigation.
 *
 * @param messages - All messages in the conversation
 * @param messageId - Current message ID
 * @returns Previous sibling's leaf node ID, or null if at the beginning
 */
export function getPreviousSibling(
	messages: readonly DatabaseMessage[],
	messageId: string
): string | null {
	const siblingInfo = getMessageSiblings(messages, messageId);
	if (!siblingInfo || siblingInfo.currentIndex <= 0) {
		return null;
	}

	return siblingInfo.siblingIds[siblingInfo.currentIndex - 1];
}
