// Guards the Enter-key contract of the chat form: Enter sends, and Shift+Enter or
// Ctrl/Cmd+Enter adds a line (continuing lists), in the small form and in the expanded
// editor alike. With sending on Enter turned off, or on a touch screen, the two swap.

import ChatFormTestWrapper from './components/ChatFormTestWrapper.svelte';
import { EditorView } from '@codemirror/view';
import { SETTINGS_KEYS } from '$lib/constants';
import { deviceStore } from '$lib/stores';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import { tick } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';
import { render } from 'vitest-browser-svelte';

async function renderForm({ expanded = false } = {}) {
	const onSubmit = vi.fn();
	const { container } = render(ChatFormTestWrapper, { expandable: true, onSubmit });

	await tick();

	if (expanded) {
		await userEvent.click(
			container.querySelector<HTMLElement>('button[aria-label="Expand editor"]')!
		);
		await tick();
	}

	const content = container.querySelector<HTMLElement>('.cm-content');
	const editor = content && EditorView.findFromDOM(content);

	if (!editor) throw new Error('editor not rendered');

	await userEvent.click(editor.contentDOM);

	return { editor, onSubmit };
}

describe('ChatForm Enter', () => {
	afterEach(() => {
		settingsStore.updateConfig(SETTINGS_KEYS.SEND_ON_ENTER, true);
		vi.restoreAllMocks();
	});

	it.each([false, true])('sends on Enter (expanded: %s)', async (expanded) => {
		const { editor, onSubmit } = await renderForm({ expanded });

		await userEvent.keyboard('hello{Enter}');
		await tick();

		expect(onSubmit).toHaveBeenCalledOnce();
		expect(editor.state.doc.toString()).toBe('hello');
	});

	it('sends on Enter inside a code block too', async () => {
		const { onSubmit } = await renderForm();

		await userEvent.keyboard('```js{Enter}');
		await tick();

		expect(onSubmit).toHaveBeenCalledOnce();
	});

	it.each(['Shift', 'Control', 'Meta'])('adds a line on %s+Enter', async (modifier) => {
		const { editor, onSubmit } = await renderForm();

		await userEvent.keyboard(`hello{${modifier}>}{Enter}{/${modifier}}world`);
		await tick();

		expect(onSubmit).not.toHaveBeenCalled();
		expect(editor.state.doc.toString()).toBe('hello\nworld');
	});

	it('continues a list on Shift+Enter', async () => {
		const { editor, onSubmit } = await renderForm({ expanded: true });

		await userEvent.keyboard('- one{Shift>}{Enter}{/Shift}two');
		await tick();

		expect(onSubmit).not.toHaveBeenCalled();
		expect(editor.state.doc.toString()).toBe('- one\n- two');
	});

	it('adds a line on Enter with a touch screen', async () => {
		vi.spyOn(deviceStore, 'isTouch', 'get').mockReturnValue(true);

		const { editor, onSubmit } = await renderForm({ expanded: true });

		await userEvent.keyboard('hello{Enter}world');
		await tick();

		expect(onSubmit).not.toHaveBeenCalled();
		expect(editor.state.doc.toString()).toBe('hello\nworld');
	});

	it('swaps the two with sending on Enter turned off', async () => {
		settingsStore.updateConfig(SETTINGS_KEYS.SEND_ON_ENTER, false);

		const { editor, onSubmit } = await renderForm();

		await userEvent.keyboard('hello{Enter}world');
		await tick();
		expect(onSubmit).not.toHaveBeenCalled();
		expect(editor.state.doc.toString()).toBe('hello\nworld');

		await userEvent.keyboard('{Control>}{Enter}{/Control}');
		await tick();
		expect(onSubmit).toHaveBeenCalledOnce();
	});
});
