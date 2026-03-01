import { describe, it, expect, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';
import { tick } from 'svelte';
import type { DatabaseMessage } from '$lib/types';
import TestChatMessageWrapper from './components/TestChatMessageWrapper.svelte';

function createUserMessage(): DatabaseMessage {
	return {
		id: 'u-edit-1',
		convId: 'conv-edit-1',
		type: 'text',
		role: 'user',
		content: 'Original user message',
		thinking: '',
		toolCalls: '',
		parent: '-1',
		children: [],
		timestamp: Date.now()
	};
}

describe('ChatMessage user editing', () => {
	it('enters edit mode when clicking Edit on a user message', async () => {
		const { container } = render(TestChatMessageWrapper, {
			target: document.body,
			props: {
				message: createUserMessage(),
				onDelete: vi.fn()
			}
		});

		const editButton = container.querySelector('button[aria-label="Edit"]') as HTMLButtonElement | null;
		expect(editButton).toBeTruthy();

		editButton?.click();
		await tick();

		const editInput = container.querySelector(
			'[placeholder="Edit your message..."]'
		) as HTMLElement | null;
		expect(editInput).toBeTruthy();
	});
});
