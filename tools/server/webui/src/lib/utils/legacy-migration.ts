/**
 * @deprecated Legacy migration utility — remove at some point in the future once all users have migrated to the new structured agentic message format.
 *
 * Converts old marker-based agentic messages to the new structured format
 * with separate messages per turn.
 *
 * Old format: Single assistant message with markers in content:
 *   <<<reasoning_content_start>>>...<<<reasoning_content_end>>>
 *   <<<AGENTIC_TOOL_CALL_START>>>...<<<AGENTIC_TOOL_CALL_END>>>
 *
 * New format: Separate messages per turn:
 *   - assistant (content + reasoningContent + toolCalls)
 *   - tool (toolCallId + content)
 *   - assistant (next turn)
 *   - ...
 */

import { LEGACY_AGENTIC_REGEX, LEGACY_REASONING_TAGS } from '$lib/constants';
import { DatabaseService } from '$lib/services/database.service';
import { MessageRole, MessageType } from '$lib/enums';
import type { DatabaseMessage } from '$lib/types/database';

const MIGRATION_DONE_KEY = 'llama-webui-migration-v2-done';

/**
 * @deprecated Part of legacy migration — remove with the migration module.
 * Check if migration has been performed.
 */
export function isMigrationNeeded(): boolean {
	try {
		return !localStorage.getItem(MIGRATION_DONE_KEY);
	} catch {
		return false;
	}
}

/**
 * Mark migration as done.
 */
function markMigrationDone(): void {
	try {
		localStorage.setItem(MIGRATION_DONE_KEY, String(Date.now()));
	} catch {
		// Ignore localStorage errors
	}
}

/**
 * Check if a message has legacy markers in its content.
 */
function hasLegacyMarkers(message: DatabaseMessage): boolean {
	if (!message.content) return false;
	return LEGACY_AGENTIC_REGEX.HAS_LEGACY_MARKERS.test(message.content);
}

/**
 * Extract reasoning content from legacy marker format.
 */
function extractLegacyReasoning(content: string): { reasoning: string; cleanContent: string } {
	let reasoning = '';
	let cleanContent = content;

	// Extract all reasoning blocks
	const re = new RegExp(LEGACY_AGENTIC_REGEX.REASONING_EXTRACT.source, 'g');
	let match;
	while ((match = re.exec(content)) !== null) {
		reasoning += match[1];
	}

	// Remove reasoning tags from content
	cleanContent = cleanContent
		.replace(new RegExp(LEGACY_AGENTIC_REGEX.REASONING_BLOCK.source, 'g'), '')
		.replace(LEGACY_AGENTIC_REGEX.REASONING_OPEN, '');

	return { reasoning, cleanContent };
}

/**
 * Parse legacy content with tool call markers into structured turns.
 */
interface ParsedTurn {
	textBefore: string;
	toolCalls: Array<{
		name: string;
		args: string;
		result: string;
	}>;
}

function parseLegacyToolCalls(content: string): ParsedTurn[] {
	const turns: ParsedTurn[] = [];
	const regex = new RegExp(LEGACY_AGENTIC_REGEX.COMPLETED_TOOL_CALL.source, 'g');

	let lastIndex = 0;
	let currentTurn: ParsedTurn = { textBefore: '', toolCalls: [] };
	let match;

	while ((match = regex.exec(content)) !== null) {
		const textBefore = content.slice(lastIndex, match.index).trim();

		// If there's text between tool calls and we already have tool calls,
		// that means a new turn started (text after tool results = new LLM turn)
		if (textBefore && currentTurn.toolCalls.length > 0) {
			turns.push(currentTurn);
			currentTurn = { textBefore, toolCalls: [] };
		} else if (textBefore && currentTurn.toolCalls.length === 0) {
			currentTurn.textBefore = textBefore;
		}

		currentTurn.toolCalls.push({
			name: match[1],
			args: match[2],
			result: match[3].replace(/^\n+|\n+$/g, '')
		});

		lastIndex = match.index + match[0].length;
	}

	// Any remaining text after the last tool call
	const remainingText = content.slice(lastIndex).trim();

	if (currentTurn.toolCalls.length > 0) {
		turns.push(currentTurn);
	}

	// If there's text after all tool calls, it's the final assistant response
	if (remainingText) {
		// Remove any partial/open markers
		const cleanRemaining = remainingText
			.replace(LEGACY_AGENTIC_REGEX.AGENTIC_TOOL_CALL_OPEN, '')
			.trim();
		if (cleanRemaining) {
			turns.push({ textBefore: cleanRemaining, toolCalls: [] });
		}
	}

	// If no tool calls found at all, return the original content as a single turn
	if (turns.length === 0) {
		turns.push({ textBefore: content.trim(), toolCalls: [] });
	}

	return turns;
}

/**
 * Migrate a single conversation's messages from legacy format to new format.
 */
async function migrateConversation(convId: string): Promise<number> {
	const allMessages = await DatabaseService.getConversationMessages(convId);
	let migratedCount = 0;

	for (const message of allMessages) {
		if (message.role !== MessageRole.ASSISTANT) continue;
		if (!hasLegacyMarkers(message)) {
			// Still check for reasoning-only markers (no tool calls)
			if (message.content?.includes(LEGACY_REASONING_TAGS.START)) {
				const { reasoning, cleanContent } = extractLegacyReasoning(message.content);
				await DatabaseService.updateMessage(message.id, {
					content: cleanContent.trim(),
					reasoningContent: reasoning || undefined
				});
				migratedCount++;
			}
			continue;
		}

		// Has agentic markers - full migration needed
		const { reasoning, cleanContent } = extractLegacyReasoning(message.content);
		const turns = parseLegacyToolCalls(cleanContent);

		// Parse existing toolCalls JSON to try to match IDs
		let existingToolCalls: Array<{ id: string; function?: { name: string; arguments: string } }> =
			[];
		if (message.toolCalls) {
			try {
				existingToolCalls = JSON.parse(message.toolCalls);
			} catch {
				// Ignore
			}
		}

		// First turn uses the existing message
		const firstTurn = turns[0];
		if (!firstTurn) continue;

		// Match tool calls from the first turn to existing IDs
		const firstTurnToolCalls = firstTurn.toolCalls.map((tc, i) => {
			const existing =
				existingToolCalls.find((e) => e.function?.name === tc.name) || existingToolCalls[i];
			return {
				id: existing?.id || `legacy_tool_${i}`,
				type: 'function' as const,
				function: { name: tc.name, arguments: tc.args }
			};
		});

		// Update the existing message for the first turn
		await DatabaseService.updateMessage(message.id, {
			content: firstTurn.textBefore,
			reasoningContent: reasoning || undefined,
			toolCalls: firstTurnToolCalls.length > 0 ? JSON.stringify(firstTurnToolCalls) : ''
		});

		let currentParentId = message.id;
		let toolCallIdCounter = existingToolCalls.length;

		// Create tool result messages for the first turn
		for (let i = 0; i < firstTurn.toolCalls.length; i++) {
			const tc = firstTurn.toolCalls[i];
			const toolCallId = firstTurnToolCalls[i]?.id || `legacy_tool_${i}`;

			const toolMsg = await DatabaseService.createMessageBranch(
				{
					convId,
					type: MessageType.TEXT,
					role: MessageRole.TOOL,
					content: tc.result,
					toolCallId,
					timestamp: message.timestamp + i + 1,
					toolCalls: '',
					children: []
				},
				currentParentId
			);
			currentParentId = toolMsg.id;
		}

		// Create messages for subsequent turns
		for (let turnIdx = 1; turnIdx < turns.length; turnIdx++) {
			const turn = turns[turnIdx];

			const turnToolCalls = turn.toolCalls.map((tc, i) => {
				const idx = toolCallIdCounter + i;
				const existing = existingToolCalls[idx];
				return {
					id: existing?.id || `legacy_tool_${idx}`,
					type: 'function' as const,
					function: { name: tc.name, arguments: tc.args }
				};
			});
			toolCallIdCounter += turn.toolCalls.length;

			// Create assistant message for this turn
			const assistantMsg = await DatabaseService.createMessageBranch(
				{
					convId,
					type: MessageType.TEXT,
					role: MessageRole.ASSISTANT,
					content: turn.textBefore,
					timestamp: message.timestamp + turnIdx * 100,
					toolCalls: turnToolCalls.length > 0 ? JSON.stringify(turnToolCalls) : '',
					children: [],
					model: message.model
				},
				currentParentId
			);
			currentParentId = assistantMsg.id;

			// Create tool result messages for this turn
			for (let i = 0; i < turn.toolCalls.length; i++) {
				const tc = turn.toolCalls[i];
				const toolCallId = turnToolCalls[i]?.id || `legacy_tool_${toolCallIdCounter + i}`;

				const toolMsg = await DatabaseService.createMessageBranch(
					{
						convId,
						type: MessageType.TEXT,
						role: MessageRole.TOOL,
						content: tc.result,
						toolCallId,
						timestamp: message.timestamp + turnIdx * 100 + i + 1,
						toolCalls: '',
						children: []
					},
					currentParentId
				);
				currentParentId = toolMsg.id;
			}
		}

		// Re-parent any children of the original message to the last created message
		// (the original message's children list was the next user message or similar)
		if (message.children.length > 0 && currentParentId !== message.id) {
			for (const childId of message.children) {
				// Skip children we just created (they were already properly parented)
				const child = allMessages.find((m) => m.id === childId);
				if (!child) continue;
				// Only re-parent non-tool messages that were original children
				if (child.role !== MessageRole.TOOL) {
					await DatabaseService.updateMessage(childId, { parent: currentParentId });
					// Add to new parent's children
					const newParent = await DatabaseService.getConversationMessages(convId).then((msgs) =>
						msgs.find((m) => m.id === currentParentId)
					);
					if (newParent && !newParent.children.includes(childId)) {
						await DatabaseService.updateMessage(currentParentId, {
							children: [...newParent.children, childId]
						});
					}
				}
			}
			// Clear re-parented children from the original message
			await DatabaseService.updateMessage(message.id, { children: [] });
		}

		migratedCount++;
	}

	return migratedCount;
}

/**
 * @deprecated Part of legacy migration — remove with the migration module.
 * Run the full migration across all conversations.
 * This should be called once at app startup if migration is needed.
 */
export async function runLegacyMigration(): Promise<void> {
	if (!isMigrationNeeded()) return;

	console.log('[Migration] Starting legacy message format migration...');

	try {
		const conversations = await DatabaseService.getAllConversations();
		let totalMigrated = 0;

		for (const conv of conversations) {
			const count = await migrateConversation(conv.id);
			totalMigrated += count;
		}

		if (totalMigrated > 0) {
			console.log(
				`[Migration] Migrated ${totalMigrated} messages across ${conversations.length} conversations`
			);
		} else {
			console.log('[Migration] No legacy messages found, marking as done');
		}

		markMigrationDone();
	} catch (error) {
		console.error('[Migration] Failed to migrate legacy messages:', error);
		// Still mark as done to avoid infinite retry loops
		markMigrationDone();
	}
}
