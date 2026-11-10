// Guards that typing `/` in the chat form opens the command picker on a desktop and
// nothing on a phone, where the pickers are off until they can be placed reliably
// around the on-screen keyboard.

import ChatFormTestWrapper from './components/ChatFormTestWrapper.svelte';
import { EditorView } from '@codemirror/view';
import { tick } from 'svelte';
import { afterEach, describe, expect, it } from 'vitest';
import { page } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

const PHONE = { height: 844, width: 390 };
const DESKTOP = { height: 900, width: 1280 };

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

describe('ChatForm pickers on a phone', () => {
	afterEach(async () => {
		await page.viewport(DESKTOP.width, DESKTOP.height);
	});

	it('opens the command picker on a desktop', async () => {
		await page.viewport(DESKTOP.width, DESKTOP.height);
		await typeSlash();

		await expect.poll(commandPicker).not.toBeNull();
	});

	it('opens no picker on a phone', async () => {
		await page.viewport(PHONE.width, PHONE.height);
		await typeSlash();

		// long enough for the popover to have mounted, as it does on a desktop
		await new Promise((resolve) => setTimeout(resolve, 300));
		expect(commandPicker()).toBeNull();
	});
});
