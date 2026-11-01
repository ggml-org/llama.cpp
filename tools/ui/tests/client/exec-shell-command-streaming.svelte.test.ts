// Guards the exec_shell_command title while the call is still being
// generated: the command must stream into the terminal title from the
// first character (args arrive as partial JSON), with a blinking cursor
// while the args are incomplete, and the conversation cwd standing in
// for the x-tool-cwd value that only arrives with the result message.

import ChatMessageToolCallBlockExecShellCommand from '$lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlockExecShellCommand.svelte';
import ChatMessageAgenticContent from '$lib/components/app/chat/ChatMessages/ChatMessageAgenticContent.svelte';
import { SETTINGS_KEYS } from '$lib/constants';
import { AgenticSectionType, BuiltInTool, MessageRole, MessageType } from '$lib/enums';
import { conversationsStore, settingsStore } from '$lib/stores';
import type { AgenticSection, DatabaseMessage } from '$lib/types';
import { tick } from 'svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

const CONV_ID = 'conv-1';
const TOOL_CALL_ID = 'call-1';
const TOOL_NAME = BuiltInTool.SERVER_EXEC_SHELL_COMMAND;
const CWD = '/home/user/project';

function messageWithArgs(args: string): DatabaseMessage {
	return {
		children: [],
		content: '',
		convId: CONV_ID,
		id: 'assistant-1',
		parent: null,
		role: MessageRole.ASSISTANT,
		timestamp: 0,
		toolCalls: JSON.stringify([
			{ function: { arguments: args, name: TOOL_NAME }, id: TOOL_CALL_ID }
		]),
		type: MessageType.TEXT
	};
}

function toolResultMessage(content: string, toolCwd?: string): DatabaseMessage {
	return {
		children: [],
		content,
		convId: CONV_ID,
		id: 'tool-1',
		parent: null,
		role: MessageRole.TOOL,
		timestamp: 0,
		toolCallId: TOOL_CALL_ID,
		toolCalls: '',
		toolCwd,
		type: MessageType.TEXT
	};
}

function triggerText(container: Element): string {
	return container.querySelector('[data-slot="collapsible-trigger"]')?.textContent ?? '';
}

function hasCursor(container: Element): boolean {
	return container.querySelector('.exec-cursor') !== null;
}

describe('exec_shell_command streaming title', () => {
	beforeEach(() => {
		settingsStore.updateMultipleConfig({
			[SETTINGS_KEYS.ALWAYS_SHOW_TOOL_CALL_CONTENT]: false,
			[SETTINGS_KEYS.SHOW_TOOL_CALL_IN_PROGRESS]: true
		});
		conversationsStore.activeConversation = {
			currNode: null,
			cwd: CWD,
			id: CONV_ID,
			lastModified: 0,
			name: 'test'
		};
	});

	afterEach(() => {
		conversationsStore.activeConversation = null;
		conversationsStore.activeMessages = [];
	});

	it('streams the partial command into the title with a cursor', async () => {
		const { container } = render(ChatMessageAgenticContent, {
			isLastAssistantMessage: true,
			isStreaming: true,
			message: messageWithArgs('{"command":"git sta'),
			toolMessages: []
		});

		await tick();

		const title = triggerText(container);

		expect(title).toContain('git sta');
		expect(title).toContain('$');
		expect(title).toContain(CWD);
		expect(hasCursor(container)).toBe(true);
	});

	it('drops the cursor once the args JSON is complete', async () => {
		const screen = render(ChatMessageAgenticContent, {
			isLastAssistantMessage: true,
			isStreaming: true,
			message: messageWithArgs('{"command":"git sta'),
			toolMessages: []
		});

		await tick();

		expect(hasCursor(screen.container)).toBe(true);

		screen.rerender({
			isLastAssistantMessage: true,
			isStreaming: true,
			message: messageWithArgs('{"command":"git status"}'),
			toolMessages: []
		});
		await tick();

		expect(triggerText(screen.container)).toContain('git status');
		expect(hasCursor(screen.container)).toBe(false);
	});

	it('switches the prompt from the conversation cwd to the persisted one', async () => {
		const screen = render(ChatMessageAgenticContent, {
			isLastAssistantMessage: true,
			isStreaming: true,
			message: messageWithArgs('{"command":"git status"}'),
			toolMessages: []
		});

		await tick();

		expect(triggerText(screen.container)).toContain(CWD);

		// Result message lands: its x-tool-cwd takes over the prompt.
		screen.rerender({
			isLastAssistantMessage: true,
			isStreaming: true,
			message: messageWithArgs('{"command":"git status"}'),
			toolMessages: [toolResultMessage('', '/other/dir')]
		});
		await tick();

		expect(triggerText(screen.container)).toContain('/other/dir');
		expect(triggerText(screen.container)).not.toContain(CWD);
	});

	it('shows the executing label only after the command finishes streaming', async () => {
		const screen = render(ChatMessageAgenticContent, {
			isLastAssistantMessage: true,
			isStreaming: true,
			message: messageWithArgs('{"command":"git sta'),
			toolMessages: []
		});

		await tick();

		// While the args stream, the title cursor alone signals activity.
		expect(screen.container.textContent).not.toContain('Executing command...');

		screen.rerender({
			isLastAssistantMessage: true,
			isStreaming: true,
			message: messageWithArgs('{"command":"git status"}'),
			toolMessages: []
		});
		await tick();

		expect(screen.container.textContent).toContain('Executing command...');
	});

	it('shows Executing command... while executing before the first output chunk', async () => {
		// The result message is created empty right before execution, so
		// the block must not go blank while the process produces nothing.
		const section: AgenticSection = {
			content: '',
			toolArgs: '{"command":"sleep 1"}',
			toolCallId: TOOL_CALL_ID,
			toolName: TOOL_NAME,
			toolResult: '',
			type: AgenticSectionType.TOOL_CALL
		};
		const { container } = render(ChatMessageToolCallBlockExecShellCommand, {
			isExecuting: true,
			isStreaming: true,
			open: true,
			section
		});

		await tick();

		expect(container.textContent).toContain('Executing command...');
	});

	it('updates the title when the store mutates toolCalls in place, as the stream does', async () => {
		// The live flow never swaps the message object: onToolCallsStreaming
		// writes growing toolCalls strings into the same store row per delta.
		conversationsStore.activeMessages = [messageWithArgs('')];

		const screen = render(ChatMessageAgenticContent, {
			isLastAssistantMessage: true,
			isStreaming: true,
			message: conversationsStore.activeMessages[0]!,
			toolMessages: []
		});

		await tick();

		const idx = conversationsStore.findMessageIndex('assistant-1');

		for (const args of ['{"command', '{"command":"git', '{"command":"git sta']) {
			conversationsStore.updateMessageAtIndex(idx, {
				toolCalls: JSON.stringify([
					{ function: { arguments: args, name: TOOL_NAME }, id: TOOL_CALL_ID }
				])
			});
			await tick();

			expect(triggerText(screen.container)).toContain(args.slice('{"command":"'.length));
			expect(hasCursor(screen.container)).toBe(true);
		}
	});
});

describe('exec_shell_command interrupted call', () => {
	beforeEach(() => {
		settingsStore.updateMultipleConfig({
			[SETTINGS_KEYS.ALWAYS_SHOW_TOOL_CALL_CONTENT]: true,
			[SETTINGS_KEYS.SHOW_TOOL_CALL_IN_PROGRESS]: true
		});
	});

	it('marks a call without a result as interrupted once streaming stops', async () => {
		const { container } = render(ChatMessageAgenticContent, {
			isLastAssistantMessage: true,
			isStreaming: false,
			message: messageWithArgs('{"command":"git status"}'),
			toolMessages: []
		});

		await tick();

		expect(triggerText(container)).toContain('interrupted');
		expect(container.textContent).toContain('Command interrupted');
		expect(container.textContent).not.toContain('Executing command...');
	});

	it('marks a command stopped before any output as interrupted', async () => {
		// exec_shell_command creates its result message before running, so a
		// stop mid-run leaves an empty result rather than none.
		const { container } = render(ChatMessageAgenticContent, {
			isLastAssistantMessage: true,
			isStreaming: false,
			message: messageWithArgs('{"command":"sleep 40"}'),
			toolMessages: [toolResultMessage('', CWD)]
		});

		await tick();

		expect(triggerText(container)).toContain('interrupted');
		expect(container.textContent).toContain('Command interrupted');
	});

	it('does not keep spinning in an earlier message while a newer reply streams', async () => {
		const { container } = render(ChatMessageAgenticContent, {
			isLastAssistantMessage: false,
			isStreaming: true,
			message: messageWithArgs('{"command":"git status"}'),
			toolMessages: []
		});

		await tick();

		expect(triggerText(container)).toContain('interrupted');
		expect(container.textContent).not.toContain('Executing command...');
	});
});
