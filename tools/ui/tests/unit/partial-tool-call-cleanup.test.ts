import { describe, it, expect } from 'vitest';
import { MessageRole } from '$lib/enums';
import { deriveAgenticSections } from '$lib/utils/agentic';
import type { DatabaseMessage } from '$lib/types/database';

function makeAssistant(overrides: Partial<DatabaseMessage> = {}): DatabaseMessage {
	return {
		id: overrides.id ?? 'ast-1',
		convId: 'conv-1',
		type: 'text',
		timestamp: Date.now(),
		role: MessageRole.ASSISTANT,
		content: overrides.content ?? '',
		parent: null,
		children: [],
		...overrides
	} as DatabaseMessage;
}

// Mirrors the filter inside ChatService.convertDbMessageToApiChatMessageData:
// a partial tool call captured mid-stream must not survive into the next request
// payload. The fix in chatStore.savePartialResponseIfNeeded clears toolCalls to ''
// on Stop/Send immediately, mirroring what the agentic flow already does in
// onAssistantTurnComplete(...undefined).
function buildApiToolCalls(message: DatabaseMessage): unknown[] | undefined {
	if (!message.toolCalls) return undefined;
	try {
		const parsed = JSON.parse(message.toolCalls);
		return Array.isArray(parsed) && parsed.length > 0 ? parsed : undefined;
	} catch {
		return undefined;
	}
}

describe('partial tool call cleanup', () => {
	// Reproduces the broken payload from the user's screenshot: model was
	// streaming a tool call whose arguments JSON was cut mid-string. The outer
	// envelope still parses, but the arguments themselves are invalid JSON and
	// the server rejects the request.
	it('marks a partial tool call payload as unsafe to re-send', () => {
		const message = makeAssistant({
			content: 'partial reasoning',
			toolCalls: JSON.stringify([
				{
					id: 'call_1',
					type: 'function',
					function: {
						name: 'exec_shell_command',
						arguments: '{"command":`grep -n \\"read_to\\" ` /Users'
					}
				}
			])
		});

		const apiToolCalls = buildApiToolCalls(message);

		// The bug: even though arguments are invalid, the outer array parses and
		// the request gets sent. Function arguments must be parseable JSON on their
		// own for the server to execute the tool.
		expect(apiToolCalls).toBeDefined();
		const args = (apiToolCalls![0] as { function: { arguments: string } }).function.arguments;
		expect(() => JSON.parse(args)).toThrow();
	});

	// After Stop, savePartialResponseIfNeeded clears toolCalls and the agentic
	// flow does the same in its silent-return detection. The next request reads
	// toolCalls = '' and the conversion drops the field entirely so the server
	// never sees the half-streamed call.
	it('drops tool_calls from the API request after toolCalls is cleared', () => {
		const clearedMessage = makeAssistant({
			content: 'partial reasoning',
			toolCalls: ''
		});

		const apiToolCalls = buildApiToolCalls(clearedMessage);
		expect(apiToolCalls).toBeUndefined();
	});

	// The cleanup path keeps the partial reasoning content visible in the UI;
	// only the tool_calls field is reset. deriveAgenticSections should still
	// surface the reasoning as interrupted (no content / no tool calls behind
	// it) without resurrecting the dead tool call block.
	it('keeps reasoning content visible after cleanup, without a tool call block', () => {
		const cleared = makeAssistant({
			content: '',
			reasoningContent: 'thinking about read_to',
			toolCalls: ''
		});

		const sections = deriveAgenticSections(cleared);
		expect(sections).toHaveLength(1);
		expect(sections[0].type).toBe('reasoning');
		expect(sections.some((s) => s.type.includes('tool_call'))).toBe(false);
	});
});
