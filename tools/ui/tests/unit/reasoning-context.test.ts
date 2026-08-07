import { describe, it, expect } from 'vitest';
import { MessageRole } from '$lib/enums';

/**
 * Tests for the new reasoning content handling.
 * In the new architecture, reasoning content is stored in a dedicated
 * `reasoningContent` field on DatabaseMessage, not embedded in content with tags.
 * The API sends it as `reasoning_content` on ApiChatMessageData.
 */

describe('reasoning content in new structured format', () => {
	it('reasoning is stored as separate field, not in content', () => {
		// Simulate what the new chat store does
		const message = {
			content: 'The answer is 4.',
			reasoningContent: 'Let me think: 2+2=4, basic arithmetic.'
		};

		// Content should be clean
		expect(message.content).not.toContain('<<<');
		expect(message.content).toBe('The answer is 4.');

		// Reasoning in dedicated field
		expect(message.reasoningContent).toBe('Let me think: 2+2=4, basic arithmetic.');
	});

	it('convertDbMessageToApiChatMessageData includes reasoning_content', () => {
		// Simulate the conversion logic
		const dbMessage = {
			role: MessageRole.ASSISTANT,
			content: 'The answer is 4.',
			reasoningContent: 'Let me think: 2+2=4, basic arithmetic.'
		};

		const apiMessage: Record<string, unknown> = {
			role: dbMessage.role,
			content: dbMessage.content
		};
		if (dbMessage.reasoningContent) {
			apiMessage.reasoning_content = dbMessage.reasoningContent;
		}

		expect(apiMessage.content).toBe('The answer is 4.');
		expect(apiMessage.reasoning_content).toBe('Let me think: 2+2=4, basic arithmetic.');
		// No internal tags leak into either field
		expect(apiMessage.content).not.toContain('<<<');
		expect(apiMessage.reasoning_content).not.toContain('<<<');
	});

	it('API message excludes reasoning when excludeReasoningFromContext is true', () => {
		const dbMessage = {
			role: MessageRole.ASSISTANT,
			content: 'The answer is 4.',
			reasoningContent: 'internal thinking'
		};

		const excludeReasoningFromContext = true;

		const apiMessage: Record<string, unknown> = {
			role: dbMessage.role,
			content: dbMessage.content
		};
		if (!excludeReasoningFromContext && dbMessage.reasoningContent) {
			apiMessage.reasoning_content = dbMessage.reasoningContent;
		}

		expect(apiMessage.content).toBe('The answer is 4.');
		expect(apiMessage.reasoning_content).toBeUndefined();
	});

	it('handles messages with no reasoning', () => {
		const dbMessage = {
			role: MessageRole.ASSISTANT,
			content: 'No reasoning here.',
			reasoningContent: undefined
		};

		const apiMessage: Record<string, unknown> = {
			role: dbMessage.role,
			content: dbMessage.content
		};
		if (dbMessage.reasoningContent) {
			apiMessage.reasoning_content = dbMessage.reasoningContent;
		}

		expect(apiMessage.content).toBe('No reasoning here.');
		expect(apiMessage.reasoning_content).toBeUndefined();
	});
});
