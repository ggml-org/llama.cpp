import { describe, it, expect } from 'vitest';
import { deriveAgenticSections, hasAgenticContent } from '$lib/utils/agentic';
import { AgenticSectionType, MessageRole } from '$lib/enums';
import type { DatabaseMessage } from '$lib/types/database';
import type { ApiChatCompletionToolCall } from '$lib/types/api';

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

function makeToolMsg(overrides: Partial<DatabaseMessage> = {}): DatabaseMessage {
	return {
		id: overrides.id ?? 'tool-1',
		convId: 'conv-1',
		type: 'text',
		timestamp: Date.now(),
		role: MessageRole.TOOL,
		content: overrides.content ?? 'tool result',
		parent: null,
		children: [],
		toolCallId: overrides.toolCallId ?? 'call_1',
		...overrides
	} as DatabaseMessage;
}

describe('deriveAgenticSections', () => {
	it('returns empty array for assistant with no content', () => {
		const msg = makeAssistant({ content: '' });
		const sections = deriveAgenticSections(msg);
		expect(sections).toEqual([]);
	});

	it('returns text section for simple assistant message', () => {
		const msg = makeAssistant({ content: 'Hello world' });
		const sections = deriveAgenticSections(msg);
		expect(sections).toHaveLength(1);
		expect(sections[0].type).toBe(AgenticSectionType.TEXT);
		expect(sections[0].content).toBe('Hello world');
	});

	it('returns reasoning + text for message with reasoning', () => {
		const msg = makeAssistant({
			content: 'Answer is 4.',
			reasoningContent: 'Let me think...'
		});
		const sections = deriveAgenticSections(msg);
		expect(sections).toHaveLength(2);
		expect(sections[0].type).toBe(AgenticSectionType.REASONING);
		expect(sections[0].content).toBe('Let me think...');
		expect(sections[1].type).toBe(AgenticSectionType.TEXT);
	});

	it('single turn: assistant with tool calls and results', () => {
		const msg = makeAssistant({
			content: 'Let me check.',
			toolCalls: JSON.stringify([
				{
					id: 'call_1',
					type: 'function',
					function: { name: 'search', arguments: '{"q":"test"}' }
				}
			])
		});
		const toolResult = makeToolMsg({
			toolCallId: 'call_1',
			content: 'Found 3 results'
		});
		const sections = deriveAgenticSections(msg, [toolResult]);
		expect(sections).toHaveLength(2);
		expect(sections[0].type).toBe(AgenticSectionType.TEXT);
		expect(sections[1].type).toBe(AgenticSectionType.TOOL_CALL);
		expect(sections[1].toolName).toBe('search');
		expect(sections[1].toolResult).toBe('Found 3 results');
	});

	it('single turn: pending tool call without result', () => {
		const msg = makeAssistant({
			toolCalls: JSON.stringify([
				{ id: 'call_1', type: 'function', function: { name: 'bash', arguments: '{}' } }
			])
		});
		const sections = deriveAgenticSections(msg, [], [], true);
		expect(sections).toHaveLength(1);
		expect(sections[0].type).toBe(AgenticSectionType.TOOL_CALL_PENDING);
		expect(sections[0].toolName).toBe('bash');
	});

	it('multi-turn: two assistant turns grouped as one session', () => {
		const assistant1 = makeAssistant({
			id: 'ast-1',
			content: 'Turn 1 text',
			toolCalls: JSON.stringify([
				{
					id: 'call_1',
					type: 'function',
					function: { name: 'search', arguments: '{"q":"foo"}' }
				}
			])
		});
		const tool1 = makeToolMsg({ id: 'tool-1', toolCallId: 'call_1', content: 'result 1' });
		const assistant2 = makeAssistant({
			id: 'ast-2',
			content: 'Final answer based on results.'
		});

		// toolMessages contains both tool result and continuation assistant
		const sections = deriveAgenticSections(assistant1, [tool1, assistant2]);
		expect(sections).toHaveLength(3);
		// Turn 1
		expect(sections[0].type).toBe(AgenticSectionType.TEXT);
		expect(sections[0].content).toBe('Turn 1 text');
		expect(sections[1].type).toBe(AgenticSectionType.TOOL_CALL);
		expect(sections[1].toolName).toBe('search');
		expect(sections[1].toolResult).toBe('result 1');
		// Turn 2 (final)
		expect(sections[2].type).toBe(AgenticSectionType.TEXT);
		expect(sections[2].content).toBe('Final answer based on results.');
	});

	it('multi-turn: three turns with tool calls', () => {
		const assistant1 = makeAssistant({
			id: 'ast-1',
			content: '',
			toolCalls: JSON.stringify([
				{
					id: 'call_1',
					type: 'function',
					function: { name: 'list_files', arguments: '{}' }
				}
			])
		});
		const tool1 = makeToolMsg({ id: 'tool-1', toolCallId: 'call_1', content: 'file1 file2' });
		const assistant2 = makeAssistant({
			id: 'ast-2',
			content: 'Reading file1...',
			toolCalls: JSON.stringify([
				{
					id: 'call_2',
					type: 'function',
					function: { name: 'read_file', arguments: '{"path":"file1"}' }
				}
			])
		});
		const tool2 = makeToolMsg({
			id: 'tool-2',
			toolCallId: 'call_2',
			content: 'contents of file1'
		});
		const assistant3 = makeAssistant({
			id: 'ast-3',
			content: 'Here is the analysis.',
			reasoningContent: 'The file contains...'
		});

		const sections = deriveAgenticSections(assistant1, [tool1, assistant2, tool2, assistant3]);
		// Turn 1: tool_call (no text since content is empty)
		// Turn 2: text + tool_call
		// Turn 3: reasoning + text
		expect(sections).toHaveLength(5);
		expect(sections[0].type).toBe(AgenticSectionType.TOOL_CALL);
		expect(sections[0].toolName).toBe('list_files');
		expect(sections[1].type).toBe(AgenticSectionType.TEXT);
		expect(sections[1].content).toBe('Reading file1...');
		expect(sections[2].type).toBe(AgenticSectionType.TOOL_CALL);
		expect(sections[2].toolName).toBe('read_file');
		expect(sections[3].type).toBe(AgenticSectionType.REASONING);
		expect(sections[4].type).toBe(AgenticSectionType.TEXT);
		expect(sections[4].content).toBe('Here is the analysis.');
	});

	it('returns REASONING_PENDING when streaming with only reasoning content', () => {
		const msg = makeAssistant({
			reasoningContent: 'Let me think about this...'
		});
		const sections = deriveAgenticSections(msg, [], [], true);
		expect(sections).toHaveLength(1);
		expect(sections[0].type).toBe(AgenticSectionType.REASONING_PENDING);
		expect(sections[0].content).toBe('Let me think about this...');
	});

	it('returns REASONING (not pending) when streaming but text content has appeared', () => {
		const msg = makeAssistant({
			content: 'The answer is',
			reasoningContent: 'Let me think...'
		});
		const sections = deriveAgenticSections(msg, [], [], true);
		expect(sections).toHaveLength(2);
		expect(sections[0].type).toBe(AgenticSectionType.REASONING);
		expect(sections[1].type).toBe(AgenticSectionType.TEXT);
	});

	it('returns REASONING (not pending) when not streaming', () => {
		const msg = makeAssistant({
			reasoningContent: 'Let me think...'
		});
		const sections = deriveAgenticSections(msg, [], [], false);
		expect(sections).toHaveLength(1);
		expect(sections[0].type).toBe(AgenticSectionType.REASONING);
	});

	it('multi-turn: streaming tool calls on last turn', () => {
		const assistant1 = makeAssistant({
			toolCalls: JSON.stringify([
				{ id: 'call_1', type: 'function', function: { name: 'search', arguments: '{}' } }
			])
		});
		const tool1 = makeToolMsg({ toolCallId: 'call_1', content: 'result' });
		const assistant2 = makeAssistant({ id: 'ast-2', content: '' });

		const streamingToolCalls: ApiChatCompletionToolCall[] = [
			{ id: 'call_2', type: 'function', function: { name: 'write_file', arguments: '{"pa' } }
		];

		const sections = deriveAgenticSections(assistant1, [tool1, assistant2], streamingToolCalls);
		// Turn 1: tool_call
		// Turn 2 (streaming): streaming tool call
		expect(sections.some((s) => s.type === AgenticSectionType.TOOL_CALL)).toBe(true);
		expect(sections.some((s) => s.type === AgenticSectionType.TOOL_CALL_STREAMING)).toBe(true);
	});
});

describe('hasAgenticContent', () => {
	it('returns false for plain assistant', () => {
		const msg = makeAssistant({ content: 'Just text' });
		expect(hasAgenticContent(msg)).toBe(false);
	});

	it('returns true when message has toolCalls', () => {
		const msg = makeAssistant({
			toolCalls: JSON.stringify([
				{ id: 'call_1', type: 'function', function: { name: 'test', arguments: '{}' } }
			])
		});
		expect(hasAgenticContent(msg)).toBe(true);
	});

	it('returns true when toolMessages are provided', () => {
		const msg = makeAssistant();
		const tool = makeToolMsg();
		expect(hasAgenticContent(msg, [tool])).toBe(true);
	});

	it('returns false for empty toolCalls JSON', () => {
		const msg = makeAssistant({ toolCalls: '[]' });
		expect(hasAgenticContent(msg)).toBe(false);
	});
});
