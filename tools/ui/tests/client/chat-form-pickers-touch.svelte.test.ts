// Guards that typing `/` in the chat form opens the command picker with a mouse and
// nothing with a finger, where the pickers are off until they can be placed reliably
// around the on-screen keyboard.

import ChatFormTestWrapper from './components/ChatFormTestWrapper.svelte';
import { EditorView } from '@codemirror/view';
import { deviceStore } from '$lib/stores';
import { tick } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { render } from 'vitest-browser-svelte';

async function typeSlash() {
	const { container } = render(ChatFormTestWrapper);

	await tick();

	const content = container.querySelector<HTMLElement>('.cm-content');
	const editor = content && EditorView.findFromDOM(content);

	if (!editor) throw new Error('editor not rendered');

	editor.dispatch({ changes: { from: 0, insert: '/' }, selection: { anchor: 1 } });
	await tick();
}

const commandPicker = () => document.querySelector('[data-slot="popover-content"]');

describe('ChatForm typed pickers', () => {
	afterEach(() => {
		vi.restoreAllMocks();
	});

	it('open the command picker with a mouse', async () => {
		vi.spyOn(deviceStore, 'isTouch', 'get').mockReturnValue(false);
		await typeSlash();

		await expect.poll(commandPicker).not.toBeNull();
	});

	it('open nothing with a finger', async () => {
		vi.spyOn(deviceStore, 'isTouch', 'get').mockReturnValue(true);
		await typeSlash();

		// long enough for the popover to have mounted, as it does with a mouse
		await new Promise((resolve) => setTimeout(resolve, 300));
		expect(commandPicker()).toBeNull();
	});
});
