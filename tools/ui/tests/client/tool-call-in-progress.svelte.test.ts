// Guards the two Settings > Display toggles that drive the default expansion of
// a tool call block.
//
// "Show tool call in progress" mirrors "Show thought in progress": a call is
// expanded by default only while it is in flight and folds back on its own once
// the result lands. "Always show tool call content" is the one that keeps
// finished calls open and wins on its own. A manual toggle always overrides the
// default, in both directions.

import ChatMessageAgenticContent from '$lib/components/app/chat/ChatMessages/ChatMessageAgenticContent.svelte';
import { COLLAPSIBLE_TRANSITION_MS, SETTINGS_KEYS } from '$lib/constants';
import { MessageRole, MessageType } from '$lib/enums';
import { settingsStore } from '$lib/stores';
import type { DatabaseMessage } from '$lib/types';
import { tick } from 'svelte';
import { beforeEach, describe, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

const TOOL_NAME = 'custom_tool';
const TOOL_CALL_ID = 'call-1';
// Both markers sit inside the collapsible body, which CollapsibleContentBlock
// only mounts while open - so their presence is exactly "the block is expanded".
// The collapse is animated and the body unmounts once it ends, so assertions
// that a block closed wait for the transition first.
const PENDING_BODY = 'Waiting for result...';
const RESULT_BODY = 'marker-tool-output';
const TRIGGER_SELECTOR = '[data-slot="collapsible-trigger"]';
// The collapse transition plus a frame of slack.
const COLLAPSE_SETTLE_MS = COLLAPSIBLE_TRANSITION_MS + 50;

function assistantMessage(): DatabaseMessage {
	return {
		children: [],
		content: '',
		convId: 'conv-1',
		id: 'assistant-1',
		parent: null,
		role: MessageRole.ASSISTANT,
		timestamp: 0,
		toolCalls: JSON.stringify([
			{ function: { arguments: '{"path":"a.txt"}', name: TOOL_NAME }, id: TOOL_CALL_ID }
		]),
		type: MessageType.TEXT
	};
}

function toolResultMessage(): DatabaseMessage {
	return {
		...assistantMessage(),
		content: RESULT_BODY,
		id: 'tool-1',
		role: MessageRole.TOOL,
		toolCallId: TOOL_CALL_ID
	};
}

function setSettings({
	alwaysShow,
	inProgress
}: {
	alwaysShow: boolean;
	inProgress: boolean;
}): void {
	settingsStore.updateMultipleConfig({
		[SETTINGS_KEYS.ALWAYS_SHOW_TOOL_CALL_CONTENT]: alwaysShow,
		[SETTINGS_KEYS.SHOW_TOOL_CALL_IN_PROGRESS]: inProgress
	});
}

function textOf(container: Element): string {
	return container.textContent ?? '';
}

async function waitForCollapse(): Promise<void> {
	await new Promise((resolve) => setTimeout(resolve, COLLAPSE_SETTLE_MS));
	await tick();
}

async function collapseBlock(screen: { container: Element }): Promise<void> {
	const trigger = screen.container.querySelector(TRIGGER_SELECTOR);

	if (!(trigger instanceof HTMLButtonElement)) throw new Error('trigger not found');

	await userEvent.click(trigger);
	await tick();
}

describe('Show tool call in progress', () => {
	beforeEach(() => {
		setSettings({ alwaysShow: false, inProgress: false });
	});

	it('expands a call that is still executing when the setting is on', async () => {
		setSettings({ alwaysShow: false, inProgress: true });

		const { container } = render(ChatMessageAgenticContent, {
			isLastAssistantMessage: true,
			isStreaming: true,
			message: assistantMessage(),
			toolMessages: []
		});

		await tick();

		expect(textOf(container)).toContain(PENDING_BODY);
	});

	it('keeps the call collapsed while executing when the setting is off', async () => {
		const { container } = render(ChatMessageAgenticContent, {
			isLastAssistantMessage: true,
			isStreaming: true,
			message: assistantMessage(),
			toolMessages: []
		});

		await tick();

		// The header is still there, only the body stays hidden.
		expect(textOf(container)).toContain(TOOL_NAME);
		expect(textOf(container)).not.toContain(PENDING_BODY);
	});

	it('folds the call back when its result arrives', async () => {
		setSettings({ alwaysShow: false, inProgress: true });

		const screen = render(ChatMessageAgenticContent, {
			isLastAssistantMessage: true,
			isStreaming: true,
			message: assistantMessage(),
			toolMessages: []
		});

		await tick();

		// Open while it waits for the result.
		expect(textOf(screen.container)).toContain(PENDING_BODY);

		// Same block, now finished: it must close on its own, no toggle given.
		screen.rerender({
			isLastAssistantMessage: true,
			isStreaming: true,
			message: assistantMessage(),
			toolMessages: [toolResultMessage()]
		});
		await waitForCollapse();

		expect(textOf(screen.container)).toContain(TOOL_NAME);
		expect(textOf(screen.container)).not.toContain(PENDING_BODY);
		expect(textOf(screen.container)).not.toContain(RESULT_BODY);
	});

	it('keeps the manual choice once the result arrives', async () => {
		setSettings({ alwaysShow: false, inProgress: true });

		const screen = render(ChatMessageAgenticContent, {
			isLastAssistantMessage: true,
			isStreaming: true,
			message: assistantMessage(),
			toolMessages: []
		});

		await tick();

		// Collapsed by hand while running, so the result must not reopen it.
		await collapseBlock(screen);

		screen.rerender({
			isLastAssistantMessage: true,
			isStreaming: true,
			message: assistantMessage(),
			toolMessages: [toolResultMessage()]
		});
		await waitForCollapse();

		expect(textOf(screen.container)).toContain(TOOL_NAME);
		expect(textOf(screen.container)).not.toContain(RESULT_BODY);
	});

	it('still expands a finished call when "always show" is on', async () => {
		setSettings({ alwaysShow: true, inProgress: false });

		const { container } = render(ChatMessageAgenticContent, {
			isLastAssistantMessage: true,
			isStreaming: true,
			message: assistantMessage(),
			toolMessages: [toolResultMessage()]
		});

		await tick();

		expect(textOf(container)).toContain(RESULT_BODY);
	});

	it('lets a manual collapse win over the in-progress default', async () => {
		setSettings({ alwaysShow: false, inProgress: true });

		const screen = render(ChatMessageAgenticContent, {
			isLastAssistantMessage: true,
			isStreaming: true,
			message: assistantMessage(),
			toolMessages: []
		});

		await tick();

		expect(textOf(screen.container)).toContain(PENDING_BODY);

		await collapseBlock(screen);
		await waitForCollapse();

		expect(textOf(screen.container)).not.toContain(PENDING_BODY);
	});

	it('lets a manual expand win over the collapsed default', async () => {
		const screen = render(ChatMessageAgenticContent, {
			isLastAssistantMessage: true,
			isStreaming: true,
			message: assistantMessage(),
			toolMessages: []
		});

		await tick();

		expect(textOf(screen.container)).not.toContain(PENDING_BODY);

		await collapseBlock(screen);

		expect(textOf(screen.container)).toContain(PENDING_BODY);
	});
});
