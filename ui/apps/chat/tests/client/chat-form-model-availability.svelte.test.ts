// Regression: when the conversation's last assistant message names a model that is no
// longer available, picking an available model in the chat form must clear the submit
// button's error state. The form's selection always decides it.

import ChatFormTestWrapper from './components/ChatFormTestWrapper.svelte';
import { MessageRole, ServerRole } from '#lib/enums/index.ts';
import { conversationsStore } from '#lib/stores/conversations/index.svelte.ts';
import { serverStore } from '#lib/stores/llama-server.svelte.ts';
import { modelsStore } from '#lib/stores/models/index.svelte.ts';
import type { DatabaseMessage, ModelOption } from '#lib/types/index.ts';
import { tick } from 'svelte';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';

const AVAILABLE: ModelOption = {
	capabilities: [],
	id: 'org/available:Q4_K_M',
	model: 'org/available:Q4_K_M',
	name: 'org/available:Q4_K_M'
};
const GONE = 'org/gone:Q4_K_M';

function assistantMessage(model: string): DatabaseMessage {
	return {
		children: [],
		content: 'Hello',
		convId: 'c1',
		id: 'm1',
		model,
		parent: null,
		role: MessageRole.ASSISTANT,
		timestamp: 0,
		type: 'text'
	} as DatabaseMessage;
}

function sendButton(container: HTMLElement) {
	const button = container.querySelector<HTMLButtonElement>('button[type="submit"]');

	if (!button) throw new Error('send button not rendered');

	return button;
}

const isErrorState = (button: HTMLButtonElement) => button.classList.contains('bg-surface-danger');

beforeEach(() => {
	serverStore.role = ServerRole.ROUTER;
	modelsStore.models = [AVAILABLE];
	modelsStore.clearSelection();
	conversationsStore.activeMessages = [assistantMessage(GONE)];
});

afterEach(() => {
	conversationsStore.activeMessages = [];
	modelsStore.clearSelection();
	modelsStore.models = [];
});

describe('ChatForm model availability', () => {
	it('shows the error state while the only model is the conversation one that is gone', async () => {
		const { container } = render(ChatFormTestWrapper, { initialValue: 'Hi' });

		await tick();

		const button = sendButton(container);

		expect(isErrorState(button)).toBe(true);
		expect(button.disabled).toBe(true);
	});

	it('clears the error state once an available model is selected in the form', async () => {
		const { container } = render(ChatFormTestWrapper, { initialValue: 'Hi' });

		await tick();
		await modelsStore.selectModelById(AVAILABLE.id);
		await tick();

		const button = sendButton(container);

		expect(isErrorState(button)).toBe(false);
		expect(button.disabled).toBe(false);
	});
});
